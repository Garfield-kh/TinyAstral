"""VLABench episode wrapper for the TinyAstral harness (zero-shot evaluation).

Builds one official evaluation episode (VLABench/configs/evaluation/tracks/<track>.json, task, index) exactly like the
official evaluator (VLABench/evaluation/evaluator/base.py: load_env(task, episode_config=cfg, random_init=False,
eval=False, run_mode="eval"); env.reset()), and offers only robot-side primitives to the adapter:

  * proprioception: end-effector site pose, finger joint positions, finger body positions;
  * motion: straight-line Cartesian waypoints (1 cm, as VLABench's own SkillLib.interpolate_path) -> dm_control IK
    (robot.get_qpos_from_ee_pos, the call the official evaluator uses for ee-space policies) -> joint position targets;
  * gripper: finger position targets (0.04 = open, 0 = closed), stepped until the fingers stop.
  * cameras: RGB + metric depth of the fixed scene cameras (+ intrinsics / extrinsics from the camera model).

Success: the official criterion is the task's termination condition (timestep.last() of the dm_control env, i.e.
task.conditions.is_met) checked after EVERY env.step, exactly as the official evaluator does; the official episode
also stops as soon as it fires. We record it in self.success_flag and stop moving the robot from then on. It is
never returned to the planner/tool layer (adapter.success() is only read by the episode driver for the official
result / logging). progress_score / intention_score are the official post-episode metrics.
No object state, entity list, target names, conditions or segmentation renders are exposed here."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


def _vlabench_root():
    """the VLABench package dir: $VLABENCH_ROOT, else <clone>/VLABench with the clone at $VLABENCH_REPO, else the installed
    VLABench package, else $TINYASTRAL_THIRD_PARTY/VLABench/VLABench, else <repo>/third_party/VLABench/VLABench."""
    if os.environ.get("VLABENCH_ROOT"):
        return os.environ["VLABENCH_ROOT"]
    if not os.environ.get("VLABENCH_REPO"):
        import importlib.util
        spec = importlib.util.find_spec("VLABench")
        if spec is not None and spec.origin:
            return os.path.dirname(spec.origin)
    repo = os.environ.get("VLABENCH_REPO") or os.path.join(
        os.environ.get("TINYASTRAL_THIRD_PARTY", str(Path(__file__).resolve().parents[2] / "third_party")), "VLABench")
    return os.path.join(repo, "VLABench")


VLABENCH_ROOT = _vlabench_root()
os.environ.setdefault("VLABENCH_ROOT", VLABENCH_ROOT)      # read by VLABench itself and by run_vlabench.official_max_episode_length

WP_STEP = 0.01          # m per Cartesian waypoint (VLABench SkillLib: target_velocity 0.05 * 0.2)
ROT_STEP = np.radians(5.0)
TOL = 0.01              # joint tolerance (rad) for "waypoint reached" (official evaluator tolerance)
OPEN_Q = 0.04           # finger joint target when open (each finger, m)


def _q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _m2q(R):
    import transforms3d as t3d
    q = t3d.quaternions.mat2quat(R)
    return q if q[0] >= 0 else -q


def _rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def _axang(rv):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rv).as_matrix()


def load_track(track):
    with open(os.path.join(VLABENCH_ROOT, "configs/evaluation/tracks", f"{track}.json")) as f:
        return json.load(f)


class VLABenchEnv:
    def __init__(self, track, task, episode, img=512):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from VLABench.envs import load_env
        import VLABench.robots  # noqa: F401  (registers the robots)
        import VLABench.tasks  # noqa: F401  (registers the tasks)
        self.track, self.task, self.episode, self.img = track, task, int(episode), img
        cfg = load_track(track)[task][self.episode]
        self.env = load_env(task, episode_config=cfg, random_init=False, eval=False, run_mode="eval")
        self.env.reset()
        self.instruction = self.env.task.get_instruction()
        self.physics = self.env.physics
        self.robot = self.env.robot
        self.success_flag = False
        self.success_step = None
        self.n_steps = 0
        self.deadline = None
        self._ee = self.robot.end_effector_site
        self._lf = self.robot.mjcf_model.find("body", "left_finger")
        self._rf = self.robot.mjcf_model.find("body", "right_finger")
        self._fj = [self.robot.mjcf_model.find("joint", "finger_joint1"), self.robot.mjcf_model.find("joint", "finger_joint2")]
        self._qcmd = np.array(self.robot.get_qpos(self.physics), float).reshape(-1)[:7]
        self._gcmd = np.array([OPEN_Q, OPEN_Q])
        m = self.physics.model
        names = [m.id2name(i, "camera") for i in range(m.ncam)]
        self.cam_names = names
        # fixed scene cameras only (the wrist camera moves with the hand); index 2 is the task's own view
        self.fixed_cams = [i for i, n in enumerate(names) if "wrist" not in (n or "")]

    # ------------------------------------------------------------------ frames
    def _base(self):
        b = self.physics.bind(self.robot.link_base)
        return np.array(b.xpos, float), np.array(b.xmat, float).reshape(3, 3)

    def _to_base(self, pw, Rw=None):
        bp, bR = self._base()
        p = bR.T @ (np.asarray(pw, float) - bp)
        return (p, bR.T @ Rw) if Rw is not None else p

    def _native_tcp(self):
        s = self.physics.bind(self._ee)
        return self._to_base(np.array(s.xpos, float), np.array(s.xmat, float).reshape(3, 3))

    def _finger_points(self):
        return (self._to_base(np.array(self.physics.bind(self._lf).xpos, float)),
                self._to_base(np.array(self.physics.bind(self._rf).xpos, float)))

    def _gq(self):
        return np.array([float(np.ravel(self.physics.bind(j).qpos)[0]) for j in self._fj])

    def _opening(self):
        return float(np.clip(np.sum(np.abs(self._gq())) / (2 * OPEN_Q), 0, 1))

    # ------------------------------------------------------------------ stepping (official success check each step)
    def _step(self, qarm, g):
        if self.success_flag:
            return True
        ts = self.env.step(np.concatenate([qarm, g]))
        self.n_steps += 1
        if ts.last():
            self.success_flag = True
            self.success_step = self.n_steps
        return self.success_flag

    def _over(self):
        return self.success_flag or (self.deadline is not None and time.time() > self.deadline)

    def _track(self, qt, max_sub):
        for _ in range(max_sub):
            if self._step(qt, self._gcmd):
                return
            q = np.array(self.robot.get_qpos(self.physics), float).reshape(-1)[:7]
            if np.max(np.abs(q - qt)) < TOL:
                return

    def _ik(self, pw, Rw):
        ok, q = self.robot.get_qpos_from_ee_pos(physics=self.physics, pos=pw, quat=_m2q(Rw))
        return bool(ok), np.asarray(q, float)[:7]

    def _servo(self, p_base, R_native_base, max_sub=12):
        """straight line (position + slerp orientation) in <=1 cm / <=5 deg waypoints; stop on stall (blocked)."""
        if self._over():
            return 0
        bp, bR = self._base()
        pt, Rt = bp + bR @ np.asarray(p_base, float), bR @ np.asarray(R_native_base, float)
        s = self.physics.bind(self._ee)
        p0, R0 = np.array(s.xpos, float), np.array(s.xmat, float).reshape(3, 3)
        rv = _rotvec(Rt @ R0.T)
        n = int(max(np.ceil(np.linalg.norm(pt - p0) / WP_STEP), np.ceil(np.linalg.norm(rv) / ROT_STEP), 1))
        best, since = 1e9, 0
        for i in range(1, n + 1):
            if self._over():
                break
            a = i / n
            ok, q = self._ik(p0 + a * (pt - p0), _axang(rv * a) @ R0)
            self._qcmd = q
            self._track(q, max_sub)
            cur = np.array(self.physics.bind(self._ee).xpos, float)
            e = float(np.linalg.norm(cur - pt))
            if e < best - 1e-3:
                best, since = e, 0
            else:
                since += 1
                if since > 8:          # ~8 waypoints without progress: blocked by contact
                    break
        # settle on the final target
        ok, q = self._ik(pt, Rt)
        self._qcmd = q
        self._track(q, 60)
        return n

    def _set_gripper(self, value):
        """value 1 = open, 0 = closed; ramp the finger targets, then step until the fingers stop."""
        if self._over():
            return 0
        g0, g1 = self._gcmd.copy(), np.ones(2) * OPEN_Q * float(np.clip(value, 0, 1))
        n = 0
        for i in range(1, 11):
            self._gcmd = g0 + (g1 - g0) * i / 10
            for _ in range(3):
                n += 1
                if self._step(self._qcmd, self._gcmd):
                    return n
        prev, still = self._gq(), 0
        while n < 120:
            n += 1
            if self._step(self._qcmd, self._gcmd):
                break
            q = self._gq()
            still = still + 1 if np.max(np.abs(q - prev)) < 2e-4 else 0
            prev = q
            if still >= 4:
                break
        return n

    # ------------------------------------------------------------------ cameras (RGB + metric depth only)
    def render(self, cam, size=None, depth=False):
        size = size or self.img
        return self.physics.render(camera_id=cam, height=size, width=size, depth=depth)

    def cam_pose(self, cam):
        m, d = self.physics.model, self.physics.data
        return np.array(d.cam_xpos[cam], float), np.array(d.cam_xmat[cam], float).reshape(3, 3), float(m.cam_fovy[cam])

    # ------------------------------------------------------------------ official metrics (after the episode only)
    def official_metrics(self):
        out = {"success": bool(self.success_flag)}
        try:
            out["progress_score"] = float(self.env.get_task_progress())
        except Exception as e:  # noqa: BLE001
            out["progress_score"] = None
            out["progress_error"] = str(e)[:200]
        try:
            out["intention_score"] = float(self.env.get_intention_score(threshold=0.1))
        except Exception as e:  # noqa: BLE001
            out["intention_score"] = None
            out["intention_error"] = str(e)[:200]
        return out

    def close(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
