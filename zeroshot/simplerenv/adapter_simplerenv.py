"""SimplerEnv (ManiSkill2_real2sim / SAPIEN 2.2.2; Google Robot or WidowX, one arm) adapter for the TinyAstral tool layer (harness/).

Same interface as harness/adapters/adapter_guava.py (hand_pose / tcp_point / run / opening / width / held / home_pose /
grasp_plan / place_plan / push_plan / objects / obj_pose / aabb / is_articulated / joints / table_z / snapshot /
register / perceive / get / masks / cam_dirs / rgb / refresh), so planner.UnifiedHarness (harness/single_arm) drives it unchanged.

Fairness rules (zero-shot, same information as the other harness/adapters/ adapters):
  * object geometry comes ONLY from the task camera: SAM3 text-prompt masks on the RGB the policy sees (visual-matching
    overlay included) + the camera's depth (SAPIEN "Position" buffer = per-pixel camera-space xyz, i.e. depth
    unprojected with the camera intrinsics) + the camera pose. The simulator segmentation buffer (needed by the official
    overlay) is never read here;
  * the table height is estimated from the same depth (dominant horizontal surface), not read from the scene;
  * robot state = proprioception only (TCP pose from forward kinematics in the robot base frame, finger joint positions);
  * motion = the official SimplerEnv controller of the robot (Google Robot: arm_pd_ee_delta_pose_align_interpolate_by_
    planner + gripper target-delta; WidowX: arm_pd_ee_target_delta_pose_align2 + gripper joint pos): closed-loop
    small end-effector deltas through env.step, so every control step goes through the official physics/controller and
    the official per-step evaluate();
  * no object poses, contacts, joint values of scene articulations or task success are read here. success() and
    official_info() exist only for the episode runner's logging and are never shown to the planner;
  * no task names anywhere.
Frame: "table frame" = robot base frame translated so that z = 0 is the estimated table top; metres. pose7 =
[x, y, z, qw, qx, qy, qz]: xyz = TCP, quaternion = TCP orientation in the base frame."""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.request

import numpy as np
import transforms3d as t3d

ARM = "right"                     # the single arm is called "right" for the shared tool layer
WMAX = 0.08                       # nominal "width" scale: width() = opening fraction * WMAX (same units as Guava)
OPEN_W, CLOSED_W = 0.07, 0.004    # opening >= 87.5 % = open; <= 5 % = closed on nothing
SAFE_Z = 0.15                     # default transit height of the TCP above the table (m)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _largest_components(p, voxel=0.015):
    from scipy import ndimage
    if len(p) < 3:
        return p
    v = np.floor(p / voxel).astype(int)
    v -= v.min(0)
    g = np.zeros(v.max(0) + 1, bool)
    g[tuple(v.T)] = True
    lab, n = ndimage.label(g, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return p
    li = lab[tuple(v.T)]
    cnt = np.bincount(li)
    cnt[0] = 0
    return p[np.isin(li, np.nonzero(cnt >= 0.15 * cnt.max())[0])]


def _pose_mat(pose):
    """sapien.Pose -> 4x4."""
    M = np.eye(4)
    M[:3, :3] = t3d.quaternions.quat2mat(np.asarray(pose.q, float))
    M[:3, 3] = np.asarray(pose.p, float)
    return M


def _b64png(a):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


class SimplerAdapter:
    """env = gym env built like simpler_env/evaluation/maniskill2_evaluator.py (obs_mode rgbd, official control mode)."""

    def __init__(self, env, cam, sam3_url, max_steps_cap=2000):
        self.env, self.U = env, env.unwrapped
        self.cam, self.sam3 = cam, sam3_url.rstrip("/")
        self.agent = self.U.agent
        self.ctrl = self.agent.controller.controllers["arm"]
        self.gctrl = self.agent.controller.controllers["gripper"]
        self.g_delta = bool(getattr(self.gctrl.config, "use_delta", False))     # Google Robot: relative gripper command
        self.names, self.cache, self.first = [], {}, {}
        self.n_render = self.n_steps = 0
        self.max_steps_cap = max_steps_cap
        self.inject = None
        self.last_info = {}
        self.info_trace = []          # per env.step official info (logging only)
        self.stats = {"seg_calls": 0, "seg_s": 0.0}
        self._home = None
        self._gcmd = 1.0              # absolute gripper command (WidowX): 1 open, -1 close
        # embodiment calibration (robot only, measured once by closing the EMPTY gripper): the Google Robot fingertips
        # touch at 17.4 % of the joint stroke before the joint limit; WidowX closes to its limit
        self.closed_empty = 0.174 if "google_robot" in str(getattr(self.U, "robot_uid", "")) else 0.0
        # Google Robot fingers rotate inward and meet AT the TCP (the open nails sit 3.4 cm behind it), so the closing point
        # has to be below the object's mid-height (30 % of its height above its bottom) for the curled nails to wrap it;
        # WidowX parallel fingers: TCP 1 cm below the top (Guava rule)
        self.grasp_frac = 0.7 if "google_robot" in str(getattr(self.U, "robot_uid", "")) else 0.0
        cfg = getattr(self.agent, "_config", None) or getattr(self.agent, "config", None)
        self.ctrl_freq = getattr(self.U, "_control_freq", 3)
        # per-control-step limits of the closed-loop EE servo (embodiment constants, not task parameters)
        self.step_p = 0.10 if self.g_delta else 0.05
        self.step_r = 0.50 if self.g_delta else 0.40

    # ---------------------------------------------------------------- episode
    def start_episode(self, obs=None):
        self.names, self.cache, self.first = [], {}, {}
        self._fingers_setup()
        self.refresh()
        self.tz = self._estimate_table()
        self._axes_setup()
        self._home = self.hand_pose().tolist()
        # the robot's own "down" = its approach direction at the start pose (Google Robot: ~18 deg forward tilt, the
        # posture its arm reaches comfortably; WidowX: straight down); grasps rotate it about the vertical axis
        self.R_home = t3d.quaternions.quat2mat(self._home[3:])
        self.yaw_home = self.quat_yaw(self._home[3:])

    def register(self, names):
        for n in names:
            n = str(n).strip()
            if n and n not in self.names:
                self.names.append(n)

    # ---------------------------------------------------------------- robot kinematics (proprioception)
    def _base_M(self):
        return _pose_mat(self.agent.robot.pose)

    def _ee_base(self):
        """TCP pose in the base frame (forward kinematics of the robot)."""
        return np.linalg.inv(self._base_M()) @ _pose_mat(self.ctrl.ee_link.pose)

    def _fingers_setup(self):
        R = self.agent.robot
        qn = [j.name for j in R.get_active_joints()]
        self.fidx = [i for i, n in enumerate(qn) if "finger" in n.lower()]
        lim = R.get_qlimits()[self.fidx]
        q0 = R.get_qpos()[self.fidx]
        # the episode starts with the gripper open: the closed end is the joint limit farther from the start position
        self.f_closed = np.where(np.abs(lim[:, 1] - q0) > np.abs(lim[:, 0] - q0), lim[:, 1], lim[:, 0])
        self.f_open = np.where(self.f_closed == lim[:, 1], lim[:, 0], lim[:, 1])

    def _axes_setup(self):
        """approach / closing axes of the gripper in the TCP frame, from the robot's own links (embodiment)."""
        R = self.agent.robot
        links = {l.name: l for l in R.get_links()}
        fl = [l for n, l in links.items() if "finger" in n.lower() and "left" in n.lower()]
        fr = [l for n, l in links.items() if "finger" in n.lower() and "right" in n.lower()]
        Minv = np.linalg.inv(_pose_mat(self.ctrl.ee_link.pose))
        pl = (Minv @ np.r_[np.mean([l.pose.p for l in fl], 0), 1])[:3]
        pr = (Minv @ np.r_[np.mean([l.pose.p for l in fr], 0), 1])[:3]
        c = pr - pl
        c = c / np.linalg.norm(c)
        # approach axis: the TCP-frame axis most orthogonal to the closing axis that points from the wrist to the tip
        par = self.ctrl.ee_link
        wrist = None
        for j in R.get_joints():
            if j.get_child_link() is not None and j.get_child_link().name == par.name and j.get_parent_link() is not None:
                wrist = j.get_parent_link()
        cand = []
        mid = (pl + pr) / 2
        for ax in np.eye(3):
            for s in (1, -1):
                v = s * ax
                if abs(v @ c) > 0.5:
                    continue
                cand.append(v)
        if wrist is not None:
            pw = (Minv @ np.r_[np.asarray(wrist.pose.p), 1])[:3]
            a = -pw / (np.linalg.norm(pw) + 1e-9)
        else:
            a = -mid / (np.linalg.norm(mid) + 1e-9)
        a = max(cand, key=lambda v: float(v @ a))
        c = c - (c @ a) * a
        c = c / np.linalg.norm(c)
        self.ax_a, self.ax_c = a, c
        self.ax_b = np.cross(a, c)

    def quat_for(self, approach_w, closing_w):
        """TCP quaternion (base frame) that points the approach axis along approach_w and closing axis along closing_w."""
        A = np.asarray(approach_w, float)
        A /= np.linalg.norm(A)
        C = np.asarray(closing_w, float)
        C = C - (C @ A) * A
        C /= np.linalg.norm(C)
        Wm = np.stack([A, C, np.cross(A, C)], 1)
        Tm = np.stack([self.ax_a, self.ax_c, self.ax_b], 1)
        return t3d.quaternions.mat2quat(Wm @ Tm.T)

    def yaw_quat(self, yaw):
        """the start-pose gripper orientation turned about the vertical so the closing axis points at yaw."""
        c, s_ = np.cos(yaw - self.yaw_home), np.sin(yaw - self.yaw_home)
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
        return t3d.quaternions.mat2quat(Rz @ self.R_home)

    def quat_yaw(self, q):
        Rm = t3d.quaternions.quat2mat(np.asarray(q, float))
        c = Rm @ self.ax_c
        return float(np.arctan2(c[1], c[0]))

    def hand_pose(self, arm=ARM):
        M = self._ee_base()
        return np.concatenate([M[:3, 3] - [0, 0, self.tz], t3d.quaternions.mat2quat(M[:3, :3])])

    def tcp_point(self, pose7):
        return np.asarray(pose7, float)[:3]

    def opening(self, arm=ARM):
        q = self.agent.robot.get_qpos()[self.fidx]
        f = float(np.mean((q - self.f_closed) / (self.f_open - self.f_closed)))
        return float(np.clip((f - self.closed_empty) / (1.0 - self.closed_empty), 0.0, 1.0))

    def width(self):
        return self.opening() * WMAX

    def home_pose(self, arm=ARM):
        return list(self._home)

    # ---------------------------------------------------------------- control (official controller, closed loop)
    def _step(self, dpos, drot, grip):
        act = np.concatenate([np.asarray(dpos, float), np.asarray(drot, float), [float(grip)]])
        # = env.step minus the camera render: same controller + physics substeps (step_action), same step counter and the
        # same official evaluate() (get_info); verified bit-identical qpos/info vs env.step. Images are rendered only when
        # the agent looks (refresh), which the planner never sees per control step anyway.
        U = self.U
        U.step_action(act)
        U._elapsed_steps += 1
        info = U.get_info(obs=None)
        self.n_steps += 1
        self.last_info = info
        self.info_trace.append({"step": self.n_steps, "success": bool(info.get("success", False))})
        if self.n_steps >= self.max_steps_cap:
            raise RuntimeError("STEP_CAP")

    def _grip_hold(self):
        return 0.0 if self.g_delta else self._gcmd

    def _ref_pose(self):
        """the pose the controller adds the delta to (current target for target-delta controllers)."""
        if getattr(self.ctrl.config, "use_target", False) and getattr(self.ctrl, "_target_pose", None) is not None:
            return _pose_mat(self.ctrl._target_pose)
        return self._ee_base()

    def _ik_ok(self, p_table, quat):
        """robot IK (the controller's own pinocchio model) reaches this TCP pose from the current joints."""
        import sapien.core as sapien
        pose = sapien.Pose(np.asarray(p_table, float)[:3] + [0, 0, self.tz], np.asarray(quat, float))
        return self.ctrl.compute_ik(pose, max_iterations=300) is not None

    def move(self, p_table, quat=None, max_steps=30, tol=0.012):
        tgt = np.asarray(p_table, float)[:3] + [0, 0, self.tz]
        Rt = t3d.quaternions.quat2mat(np.asarray(quat, float)) if quat is not None else self._ee_base()[:3, :3]
        best, stall = 9.0, 0
        for _ in range(max_steps):
            cur = self._ee_base()
            e = tgt - cur[:3, 3]
            er = t3d.axangles.mat2axangle(Rt @ cur[:3, :3].T)
            ang = abs(er[1])
            if np.linalg.norm(e) < tol and ang < 0.12:
                break
            ref = self._ref_pose()
            dp = tgt - ref[:3, 3]
            n = np.linalg.norm(dp)
            if n > self.step_p:
                dp *= self.step_p / n
            ax, an = t3d.axangles.mat2axangle(Rt @ ref[:3, :3].T)
            an = float(_wrap(an))
            drot = np.asarray(ax) * float(np.clip(an, -self.step_r, self.step_r))
            self._step(dp, drot, self._grip_hold())
            c2 = self._ee_base()
            d = float(np.linalg.norm(tgt - c2[:3, 3])) + 0.05 * abs(t3d.axangles.mat2axangle(Rt @ c2[:3, :3].T)[1])
            if d < best - 0.002:
                best, stall = d, 0
            else:
                stall += 1
                if stall >= 2:
                    break
        return float(np.linalg.norm(self.hand_pose()[:3] - np.asarray(p_table, float)[:3]))

    def _gripper(self, close, max_steps=12):
        prev = self.opening()
        still = 0
        for _ in range(max_steps):
            if self.g_delta:
                g = 1.0 if close else -1.0
            else:
                self._gcmd = -1.0 if close else 1.0
                g = self._gcmd
            self._step([0, 0, 0], [0, 0, 0], g)
            o = self.opening()
            still = still + 1 if abs(o - prev) < 0.01 else 0
            prev = o
            if still >= 2:
                break
        return self.opening()

    def run(self, items: dict) -> bool:
        """items {arm: [action dict]}; action types: move {pose}, grip {val}, home. Returns reachability."""
        ok = True
        for _, L in items.items():
            for d in L:
                if d["type"] == "move":
                    err = self.move(d["pose"][:3], d["pose"][3:7] if len(d["pose"]) >= 7 else None)
                    ok = ok and (err < 0.03 or bool(d.get("contact")))
                elif d["type"] == "home":
                    h = self.hand_pose()
                    self.move([h[0], h[1], max(h[2], self._home[2])], h[3:])
                    self.move(self._home[:3], self._home[3:])
                elif d["type"] == "grip":
                    self.last_close = self._gripper(d["val"] < 0.5)
        return ok

    # ---------------------------------------------------------------- sensing (camera only)
    def refresh(self):
        raw = self.U.get_obs()                 # renders the camera (no physics step); overlay applied as in the policy obs
        im = raw["image"][self.cam]
        self.last_rgb = np.clip(im["Color"][..., :3] * 255, 0, 255).astype(np.uint8)
        pos = im["Position"]
        self.last_xyz_gl = pos[..., :3]
        self.last_valid = pos[..., 3] < 1.0
        self.last_c2w = np.asarray(raw["camera_param"][self.cam]["cam2world_gl"], float)
        self.n_render += 1

    def rgb(self):
        return self.last_rgb

    def _to_table(self, pts_gl):
        Mb = np.linalg.inv(self._base_M()) @ self.last_c2w
        p = (Mb[:3, :3] @ pts_gl.T).T + Mb[:3, 3]
        if hasattr(self, "tz"):
            p[:, 2] -= self.tz
        return p

    def _estimate_table(self):
        """dominant horizontal surface below the gripper in front of the robot (camera depth only)."""
        v = self.last_valid
        p = self._to_table(self.last_xyz_gl[v].reshape(-1, 3))
        ee = self._ee_base()[:3, 3]
        sel = (p[:, 2] < ee[2] - 0.05) & (np.linalg.norm(p[:, :2] - ee[:2], axis=1) < 0.8)
        z = p[sel, 2] if sel.sum() > 500 else p[:, 2]
        h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.01, 0.01))
        i = int(np.argmax(h))
        return float((e[i] + e[i + 1]) / 2)

    def cam_dirs(self):
        """unit xy vectors of camera-right and camera-away (into the image) in the table frame."""
        Mb = np.linalg.inv(self._base_M()) @ self.last_c2w
        r = Mb[:3, 0][:2]                       # OpenGL camera: +x right, -z forward
        f = -Mb[:3, 2][:2]
        return r / np.linalg.norm(r), f / np.linalg.norm(f)

    def _seg(self, prompt, top_k=6, min_score=0.15):
        """SAM3 masks for one prompt; identical (image, prompt) requests are answered from a small cache (pure speed-up)."""
        import hashlib
        key = (hashlib.md5(self.last_rgb.tobytes()).hexdigest(), prompt, top_k, min_score)
        cache = self.__dict__.setdefault("_segcache", {})
        if key in cache:
            self.stats["seg_cached"] = self.stats.get("seg_cached", 0) + 1
            return cache[key]
        r = self._seg_call(prompt, top_k, min_score)
        if len(cache) > 64:
            cache.pop(next(iter(cache)))
        cache[key] = r
        return r

    def _seg_call(self, prompt, top_k=6, min_score=0.15):
        import time
        from PIL import Image
        t = time.time()
        body = json.dumps({"image_b64": _b64png(self.last_rgb), "prompts": [prompt], "top_k": top_k, "min_score": min_score}).encode()
        rq = urllib.request.Request(self.sam3 + "/seg", body, {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(rq, timeout=120).read())["results"][prompt]
        self.stats["seg_calls"] += 1
        self.stats["seg_s"] += time.time() - t
        return [(float(m["score"]), np.array(Image.open(io.BytesIO(base64.b64decode(m["mask"])))) > 127) for m in r]

    def masks(self, prompt, min_score=0.5):
        """all SAM3 instance masks for a prompt, sorted left-to-right in the image."""
        ms = [m for s, m in self._seg(prompt) if s >= min_score and m.any()]
        return sorted(ms, key=lambda m: np.nonzero(m)[1].mean())

    def mask(self, name):
        """'<noun> #i' = i-th instance (1-based, left to right in the image); otherwise the best-scoring mask."""
        m = re.fullmatch(r"(.+?)\s*#(\d+)", name.strip())
        if m:
            ms = self.masks(m.group(1))
            i = int(m.group(2)) - 1
            if not 0 <= i < len(ms):
                raise RuntimeError(f"only {len(ms)} instances of {m.group(1)}")
            return ms[i]
        r = [(s, mk) for s, mk in self._seg(name) if mk.any()]
        if not r:
            raise RuntimeError("not found")
        return max(r, key=lambda x: x[0])[1]

    def perceive(self, names, refresh=True):
        """{name: {'min','max','center','u','v','npix','pts'} | None}; table frame. SAM3 mask + camera depth."""
        if refresh:
            self.refresh()
        out = {}
        for n in names:
            try:
                mask = self.mask(n) & self.last_valid
                if mask.sum() < 10:
                    raise RuntimeError("few points")
                pcb = self._to_table(self.last_xyz_gl[mask].reshape(-1, 3))
                med = np.median(pcb, 0)
                d = np.linalg.norm(pcb - med, axis=1)
                pcb = pcb[d < np.percentile(d, 95) * 1.5 + 1e-6]
                pcb = _largest_components(pcb)
                lo, hi = np.percentile(pcb, 2, axis=0), np.percentile(pcb, 98, axis=0)
                ext = hi - lo
                if hi[2] < -0.15 or max(ext[0], ext[1]) > 0.6 or ext[2] > 0.6:
                    raise RuntimeError("implausible box")
                vs, us = np.nonzero(mask)
                o = {"min": lo, "max": hi, "center": (lo + hi) / 2, "u": float(us.mean()), "v": float(vs.mean()),
                     "npix": int(mask.sum()), "pts": pcb}
                out[n] = o
                self.cache[n] = o
                self.first.setdefault(n, o)
            except Exception:  # noqa: BLE001
                out[n] = None
        return out

    def get(self, name, refresh=True):
        o = self.perceive([name], refresh)[name]
        return o if o is not None else self.cache.get(name)

    # ---------------------------------------------------------------- scene interface for tools.Tools
    def objects(self):
        return list(self.names)

    def obj_pose(self, name):
        o = self.get(name)
        if o is None:
            raise RuntimeError(f"cannot see the {name}")
        return np.concatenate([o["center"], [1.0, 0, 0, 0]])

    def aabb(self, name):
        o = self.cache.get(name) or self.get(name)
        return np.asarray(o["min"]), np.asarray(o["max"])

    def is_articulated(self, name):
        return False            # no privileged joint information in this backend

    def joints(self, name):
        return []

    def table_z(self):
        return 0.0

    def held(self) -> dict:
        """perception + proprioception: fingers stalled part-open and the TCP inside an object's box (+3 cm)."""
        w = self.width()
        if w >= OPEN_W or w <= CLOSED_W:
            return {}
        tcp = self.hand_pose()[:3]
        best, bd = None, 9.0
        for n in self.names:
            o = self.cache.get(n)
            if o is None:
                continue
            if np.all(tcp >= o["min"] - 0.03) and np.all(tcp <= o["max"] + 0.03):
                d = float(np.linalg.norm(tcp - o["center"]))
                if d < bd:
                    best, bd = n, d
        return {ARM: best} if best else {}

    def snapshot(self, names=None):
        names = self.names if names is None else names
        obs = self.perceive(names) if names else {}
        objs = {}
        for n in names:
            o = obs.get(n) or self.cache.get(n)
            if o is None:
                continue
            objs[n] = {"pose": list(o["center"]) + [1.0, 0, 0, 0], "lo": o["min"].tolist(), "hi": o["max"].tolist(),
                       "q": [], "seen": obs.get(n) is not None}
        return {"hands": {ARM: self.hand_pose().tolist()}, "opening": {ARM: self.opening()}, "cmd": {ARM: None},
                "held": self.held(), "objects": objs}

    # ---------------------------------------------------------------- geometry plans (generic, perception based)
    def _transit(self, xy, z_goal, quat, extra_clear=0.0):
        tops = [float(o["max"][2]) for o in self.cache.values() if o is not None]
        hz = max([SAFE_Z, z_goal + 0.05] + [t + 0.08 + extra_clear for t in tops])
        hz = min(hz, self._home[2] + 0.10)       # stay in the arm's comfortable band (start-pose height + 10 cm)
        cur = self.hand_pose()
        acts = []
        if cur[2] < hz - 0.01:
            acts.append({"type": "move", "pose": [cur[0], cur[1], hz, *quat]})
        acts.append({"type": "move", "pose": [xy[0], xy[1], hz, *quat]})
        return acts

    def grasp_plan(self, obj, arm=ARM, pre_dist=0.1, depth=0.0, grip=0.0, contact=None):
        """top-down grasp from SAM3 + depth (same rule as adapter_guava): anchor = top-band point nearest the band
        centroid, closing axis = local PCA short axis, contact height = top - 1 cm (WidowX) / 30 % above the bottom (Google
        Robot, see grasp_frac) (- depth), clipped above the table.
        contact: optional yaw offset in degrees (re-grasp variation)."""
        o = self.get(obj)
        if o is None:
            return None
        pc = o["pts"]
        top_z = float(pc[:, 2].max())
        band = pc[pc[:, 2] > top_z - 0.015]
        if len(band) < 10:
            band = pc
        bxy = band[:, :2]
        dc = np.linalg.norm(bxy - bxy.mean(0), axis=1)
        near = band[dc < dc.min() + 0.005]
        cur = self.hand_pose()
        anchor = near[int(np.argmin(np.linalg.norm(near[:, :2] - cur[:2], axis=1)))]
        local = band[np.linalg.norm(bxy - anchor[:2], axis=1) < 0.025]
        if len(local) < 3:
            local = band
        xy = local[:, :2] - local[:, :2].mean(0)
        _, _, Vt = np.linalg.svd(xy, full_matrices=False)
        if Vt[1, 1] < 0 or (Vt[1, 1] == 0 and Vt[1, 0] < 0):
            Vt[1] = -Vt[1]
        yaw = float(np.arctan2(Vt[1, 1], Vt[1, 0])) + (np.radians(contact) if contact else 0.0)
        cy = self.quat_yaw(cur[3:])
        if abs(_wrap(yaw + np.pi - cy)) < abs(_wrap(yaw - cy)):
            yaw += np.pi
        # floor: above the estimated table, or just above the object's own bottom when it rests lower (e.g. in a sink)
        gz = max(top_z - max(0.01, self.grasp_frac * (top_z - float(o["min"][2]))) - depth, min(0.008, float(o["min"][2]) + 0.005))
        goal = np.array([float(anchor[0]), float(anchor[1]), gz])
        # the robot's own down posture turned to the grasp yaw if the arm reaches it, else tilt the approach away from the
        # robot base (IK-checked)
        q = self.yaw_quat(yaw)
        A_ = t3d.quaternions.quat2mat(q) @ self.ax_a
        r = goal[:2] / (np.linalg.norm(goal[:2]) + 1e-9)
        for tdeg in (0, 20, 35, 50, 65):
            t = np.radians(tdeg)
            if tdeg == 0:
                qq = self.yaw_quat(yaw)
                Av = t3d.quaternions.quat2mat(qq) @ self.ax_a
            else:
                Av = np.array([r[0] * np.sin(t), r[1] * np.sin(t), -np.cos(t)])
                Cv = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                if abs(Cv @ Av) > 0.9:
                    Cv = np.cross(Av, [0, 0, 1.0])
                qq = self.quat_for(Av, Cv)
            pre = goal - Av * (top_z + pre_dist - gz)
            if self._ik_ok(goal, qq) and self._ik_ok(pre, qq):
                q, A_ = qq, Av
                break
        q = q.tolist()
        pre = goal - A_ * (top_z + pre_dist - gz)
        acts = [{"type": "grip", "val": 1.0}] if self.width() < OPEN_W else []
        acts += self._transit(pre[:2], pre[2], q)
        acts += [{"type": "move", "pose": [*pre, *q]},
                 {"type": "move", "pose": [*goal, *q]},
                 {"type": "grip", "val": float(grip)}]
        return acts

    def place_plan(self, obj, arm, target7, fp=None, pre_dist=0.1, dist=0.02, release=True, extra=None):
        o = self.get(obj)
        hand = self.hand_pose()
        if o is None:
            off = np.array([0, 0, 0.0])
        else:
            off = hand[:3] - o["center"]
            off[:2] = 0.0
            off[2] = float(np.clip(off[2], 0.0, 0.15))
        tgt = np.asarray(target7, float)[:3] + off
        q = hand[3:].tolist()
        acts = self._transit(tgt[:2], tgt[2] + pre_dist, q, extra_clear=max(0.0, off[2]))
        acts += [{"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + pre_dist, *q]},
                 {"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + dist, *q]}]
        if release:
            acts += [{"type": "grip", "val": 1.0},
                     {"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + dist + 0.08, *q]}]
        return acts

    def push_plan(self, obj, direction_xy, distance, height=None, clearance=0.04):
        o = self.get(obj)
        if o is None:
            return None
        d = np.asarray(direction_xy, float)
        d = d / np.linalg.norm(d)
        pts = o["pts"][:, :2]
        proj = (pts - o["center"][:2]) @ d
        back = float(-proj.min())
        h = float(o["max"][2])
        zc = float(o["center"][2])
        z = float(np.clip(height if height is not None else zc, max(0.015, float(o["min"][2]) + 0.01), max(0.015, h - 0.01)))
        yaw = float(np.arctan2(d[1], d[0])) + np.pi / 2
        cy = self.quat_yaw(self.hand_pose()[3:])
        if abs(_wrap(yaw + np.pi - cy)) < abs(_wrap(yaw - cy)):
            yaw += np.pi
        q = self.yaw_quat(yaw).tolist()
        start = o["center"][:2] - d * (back + clearance)
        end = o["center"][:2] + d * (distance - back)
        acts = [{"type": "grip", "val": 0.0}] if self.width() > CLOSED_W + 0.004 else []
        acts += self._transit(start, z, q)
        acts += [{"type": "move", "pose": [start[0], start[1], z, *q]}]
        n = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.04)))
        for i in range(1, n + 1):
            p = start + (end - start) * i / n
            acts.append({"type": "move", "pose": [p[0], p[1], z, *q], "contact": True})
        back_p = end - d * 0.05
        acts += [{"type": "move", "pose": [back_p[0], back_p[1], z, *q]},
                 {"type": "move", "pose": [back_p[0], back_p[1], max(z + 0.10, h + 0.06), *q]}]
        return acts

    # ---------------------------------------------------------------- logging only (never shown to the planner)
    def success(self):
        return bool(self.last_info.get("success", False))

    def official_info(self):
        return {k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in self.last_info.items()}
