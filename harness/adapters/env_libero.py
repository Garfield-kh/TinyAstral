"""LIBERO / LIBERO-PRO port (MotorMind's Table 3 protocol) of the RoboDawn discrete-command interface.

Protocol reproduced from MotorMind (gitclone/motormind: sweep_missions.py, adapter/libero/adapter.py):
  * suites: LIBERO-PRO aggregate index (MotorMind `--suite libero_pro`): 0-49 spatial, 50-99 object, 100-149 goal,
    150-199 libero_10; inside each block 0-9 = ORIGINAL task, lan, object, swap, task.
    Table 3 base = tasks 0-9, 50-59, 100-109, seeds 0,1  (seed k = LIBERO's saved initial state k).
  * scene: MotorMind's vendored BDDL/init files (motormind/sim/libero) through its own use_repo_libero()+pro.register()
    (wrapped, not modified); OffScreenRenderEnv(horizon=300000) as MotorMind; reset -> set_init_state(states[seed])
    -> 32 settle ticks (MotorMind settle_steps=32).
  * budget: 450 s wall clock per episode (MotorMind --budget-s 450); success = BDDL predicate
    env._check_success(), read by the runner for logging. The runner determines termination.
    Elapsed time is exposed to the model as "steps used" = elapsed seconds / 450.
  * control: robosuite OSC_POSE (LIBERO's default controller, delta mode, 20 Hz); every command is servoed to its
    target pose with delta actions clipped to the controller's output_max (5 cm / 0.5 rad per tick).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import bp_core as C
from harness.core.env import EpisodeInfo

import paths as _paths  # noqa: E402
MM = _paths.MOTORMIND
WALL_S = 450.0
SETTLE = 32


def _libero():
    if str(MM) not in sys.path:
        sys.path.insert(0, str(MM))
    from adapter.libero.adapter import use_repo_libero
    use_repo_libero()
    from adapter.libero import pro
    pro.register()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    return pro, benchmark, get_libero_path, OffScreenRenderEnv


def split(index_or_name):
    """'libero_pro:57' / '57' -> (suite, task id) with MotorMind's aggregate indexing; 'libero_goal:3' also works."""
    s = str(index_or_name)
    pro = _libero()[0]
    if ":" in s:
        suite, i = s.split(":")
        if suite == "libero_pro":
            return pro.split_index(int(i))
        return suite, int(i)
    return pro.split_index(int(s))


class LiberoEnv(C.PortEnv):
    bench = "libero"
    overview_cam = "agentview"
    wrist_cam = "robot0_eye_in_hand"
    grid_x = (20, 80)
    grid_y = (-40, 40)
    overview_caption = ("overview (agentview camera: in front of the robot looking back at it, from above; the robot "
                        "base is at the TOP of the image, so +x (forward, away from the robot) points DOWN the image and +y "
                        "points to the RIGHT of the image; orange circle R = fingertips, line = direction the fingers point; "
                        "thin grey grid drawn ON THE TABLE TOP (z = 0) every 10 cm, labelled x.../y...; xyz legend)")

    def __init__(self, task: str, seed: int, img_size: int = 512):
        super().__init__()
        self.img_size = img_size
        self.suite_name, self.task_id = split(task)
        _, benchmark, get_libero_path, OffScreenRenderEnv = _libero()
        suite = benchmark.get_benchmark_dict()[self.suite_name]()
        t = suite.get_task(self.task_id)
        self.instruction = t.language
        self.task_name = f"{self.suite_name}:{self.task_id}"
        states = np.asarray(suite.get_task_init_states(self.task_id))
        if seed >= len(states):
            raise ValueError(f"{self.task_name} ships {len(states)} initial states, no seed {seed}")
        self.init_state = states[seed]
        self.seed = int(seed)
        bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview", "robot0_eye_in_hand"],
                                      camera_heights=128, camera_widths=128, horizon=300000)
        self.t0 = None

    # ------------------------------------------------------------------ episode
    def reset_episode(self, episode_index=None) -> EpisodeInfo:
        self.env.reset()
        self.env.set_init_state(self.init_state)
        e = self.env.env
        self.sim = e.sim
        self.robot = e.robots[0]
        self._g = -1.0
        self._bid = self.sim.model.body_name2id(self.robot.robot_model.root_body)
        c = self.robot.controller
        self.omax = np.asarray(c.output_max, float)
        for _ in range(SETTLE):
            self._step(np.zeros(6))
        self._calibrate_tool()
        self._initial_objects = {k: v["position_cm"] for k, v in self.object_states().items()}
        self.t0 = time.time()
        return EpisodeInfo(self.task_name, "libero_pro", self.seed, self.seed, self.instruction, int(WALL_S),
                           {"suite": self.suite_name, "task_id": self.task_id})

    def close_episode(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def steps_used(self):
        return 0 if self.t0 is None else int(time.time() - self.t0)

    @property
    def step_limit(self):
        return int(WALL_S)

    # ------------------------------------------------------------------ frames
    def _base(self):
        return self.sim.data.body_xpos[self._bid].copy(), self.sim.data.body_xmat[self._bid].reshape(3, 3).copy()

    def _to_base(self, pw, Rw=None):
        bp, bR = self._base()
        p = bR.T @ (np.asarray(pw) - bp)
        return p if Rw is None else (p, bR.T @ Rw)

    def _native_tcp(self):
        sid = self.robot.eef_site_id
        return self._to_base(self.sim.data.site_xpos[sid].copy(), self.sim.data.site_xmat[sid].reshape(3, 3).copy())

    def _finger_points(self):
        lf = C.find_body(self.sim, "leftfinger")
        rf = C.find_body(self.sim, "rightfinger")
        return self._to_base(self.sim.data.body_xpos[lf].copy()), self._to_base(self.sim.data.body_xpos[rf].copy())

    def table_z_base_m(self):
        bp, bR = self._base()
        return C.mj_table_z(self.sim, bp, bR, xs=(0.3, 0.8), ys=(-0.3, 0.3))

    def _cam_pose(self, cam):
        i = self.sim.model.camera_name2id(cam)
        p, R = self._to_base(self.sim.data.cam_xpos[i].copy(), self.sim.data.cam_xmat[i].reshape(3, 3).copy())
        return p, R, float(self.sim.model.cam_fovy[i])

    def _render(self, cam):
        return np.ascontiguousarray(self.sim.render(camera_name=cam, width=self.img_size, height=self.img_size)[::-1])

    def object_positions_base(self):
        out = {}
        for name, bid in self.env.env.obj_body_id.items():
            out[name] = self._to_base(self.sim.data.body_xpos[bid].copy())
        return out

    def check_success(self):
        try:
            return bool(self.env.env._check_success())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ control (robosuite OSC_POSE, delta)
    def _step(self, a6):
        self.env.step(np.concatenate([a6, [self._g]]))

    def _hold_action(self, pt, Rt):
        sid = self.robot.eef_site_id
        ep = pt - self.sim.data.site_xpos[sid]
        er = C.rotvec(Rt @ self.sim.data.site_xmat[sid].reshape(3, 3).T)
        return np.concatenate([np.clip(ep / self.omax[:3], -1, 1), np.clip(er / self.omax[3:], -1, 1)])

    def _here(self):
        sid = self.robot.eef_site_id
        return self.sim.data.site_xpos[sid].copy(), self.sim.data.site_xmat[sid].reshape(3, 3).copy()

    def _servo(self, p_base, R_native_base, max_ticks=300):
        """Large rotations are cut into <=10 deg / <=5 cm waypoints along the geodesic (OSC couples big orientation
        errors into large position transients); the last waypoint is the target."""
        bp, bR = self._base()
        pt, Rt = bp + bR @ p_base, bR @ R_native_base
        p0, R0 = self._here()
        rv = C.rotvec(Rt @ R0.T)
        n_wp = int(max(np.ceil(np.degrees(np.linalg.norm(rv)) / 10.0), np.ceil(np.linalg.norm(pt - p0) / 0.05), 1))
        ticks = 0
        if np.degrees(np.linalg.norm(rv)) > 15:
            for i in range(1, n_wp):
                s = i / n_wp
                ticks += self._servo1(p0 + s * (pt - p0), C.axang(rv, np.degrees(np.linalg.norm(rv)) * s) @ R0
                                      if np.linalg.norm(rv) > 1e-9 else R0, 60, loose=True)
        return ticks + self._servo1(pt, Rt, max_ticks)

    def _servo1(self, pt, Rt, max_ticks, loose=False):
        best, since, n = 1e9, 0, 0
        sid = self.robot.eef_site_id
        while n < max_ticks:
            pc = self.sim.data.site_xpos[sid].copy()
            Rc = self.sim.data.site_xmat[sid].reshape(3, 3)
            ep = pt - pc
            er = C.rotvec(Rt @ Rc.T)
            e = np.linalg.norm(ep) + 0.02 * np.linalg.norm(er)
            if np.linalg.norm(ep) < (0.01 if loose else 0.003) and np.linalg.norm(er) < (0.06 if loose else 0.03):
                break
            if e < best - 2e-4:
                best, since = e, 0
            else:
                since += 1
                if since > 40:
                    break
            a = np.concatenate([np.clip(ep / self.omax[:3], -1, 1), np.clip(er / self.omax[3:], -1, 1)])
            self._step(a)
            n += 1
        if loose:
            return n
        for _ in range(3):
            self._step(self._hold_action(pt, Rt))
        return n + 3

    def _gq(self):
        idx = self.robot._ref_gripper_joint_pos_indexes
        return self.sim.data.qpos[idx].copy()

    def _opening(self):
        q = self._gq()
        return float(np.clip((abs(q[0]) + abs(q[-1])) / 0.08, 0, 1))

    def _set_gripper(self, value):
        self._g = 1.0 - 2.0 * float(value)      # robosuite: -1 open, +1 close
        prev, still, n = self._gq(), 0, 0
        pt, Rt = self._here()
        while n < 60:
            self._step(self._hold_action(pt, Rt))
            n += 1
            q = self._gq()
            still = still + 1 if np.max(np.abs(q - prev)) < 2e-4 else 0
            prev = q
            if still >= 4 and n >= 8:
                break
        return n

    def _wait(self):
        pt, Rt = self._here()
        for _ in range(10):
            self._step(self._hold_action(pt, Rt))
        return 10
