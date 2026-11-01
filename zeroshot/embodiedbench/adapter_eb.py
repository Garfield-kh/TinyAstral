"""EmbodiedBench EB-Manipulation (VLMbench / CoppeliaSim + PyRep, single Franka Panda) adapter for the TinyAstral tool layer (harness/).

Same interface as adapter_guava.GuavaAdapter (hand_pose / tcp_point / run / opening / held / home_pose / grasp_plan /
place_plan / push_plan / objects / obj_pose / aabb / is_articulated / joints / table_z / snapshot / masks / perceive).

Honesty rules (hard):
  * object geometry ONLY from the front camera: SAM3 mask (text prompt = the planner's own noun phrase) on front_rgb +
    the camera's own point cloud (front_point_cloud = depth unprojected with the camera calibration). The benchmark's
    privileged inputs are NOT used: no obs.object_informations, no *_mask (simulator segmentation), no task_low_dim_state,
    no YOLO/GT-projected boxes, no discretised GT object coordinates;
  * robot state = proprioception only (tip pose, finger joint positions) + robot geometry (finger axis at start);
  * motion goes through the OFFICIAL action interface: every arm motion is one EBManEnv.step() with a 7-d DISCRETE
    action [x, y, z, roll, pitch, yaw, gripper] (100 position bins, 120 rotation bins), i.e. exactly what GPT-4o etc.
    output in the paper; the env plans the path (ABS_EE_POSE_PLAN_WORLD_FRAME) and applies its own grasp attach rule;
  * success = the official per-step task_success (info from EBManEnv.step, incl. its stack special case); it is stored
    for logging (success_step) and NEVER returned to the planner (EBManEnv's env_feedback text with the reward is dropped);
  * no task names, no per-task branches.
Frame: table frame = world x, y and z - table_z (table height estimated from the camera point cloud at episode start);
pose7 = [x, y, z, qw, qx, qy, qz] of the tip (fingertip centre)."""
from __future__ import annotations

import base64
import io

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ARM = "right"
OPEN_W, CLOSED_W = 0.075, 0.004     # finger opening (sum of the two finger joints, m)
SAFE_Z = 0.22
BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])   # EB-Manipulation SCENE_BOUNDS (action discretisation box)
VOX, ROT_RES = 100, 3


def yaw_quat(yaw):
    """straight-down gripper (roll 180, pitch 0) with the given yaw; = euler xyz (180, 0, yaw) in wxyz."""
    return np.array([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])


def quat_yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _largest_components(p, voxel=0.01):
    from scipy import ndimage
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


def to_discrete(pos_world, quat_wxyz, grip):
    """continuous tip target -> the official 7-d discrete action (inverse of eb_man_utils.get_continous_action_from_discrete)."""
    res = (BOUNDS[3:] - BOUNDS[:3]) / VOX
    idx = np.clip(np.round((np.asarray(pos_world) - BOUNDS[:3]) / res - 0.5), 0, VOX - 1).astype(int)
    w, x, y, z = quat_wxyz
    e = Rot.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)
    r = np.round((e + 180.0) / ROT_RES).astype(int) % int(360 / ROT_RES)
    return [int(v) for v in idx] + [int(v) for v in r] + [1.0 if grip >= 0.5 else 0.0]


class EBAdapter:
    def __init__(self, env, sam3_url, max_env_steps=60):
        """env = embodiedbench EBManEnv after reset(); sam3_url = SAM3 server (POST /segment)."""
        self.env = env
        self.sam3 = sam3_url.rstrip("/")
        self.max_env_steps = max_env_steps
        self.names, self.cache, self.first = [], {}, {}
        self.n_render = self.n_perceive = 0
        self.inject = None
        self.trace = []

    # ---------------------------------------------------------------- episode
    def start_episode(self, obs):
        self.obs = obs if isinstance(obs, dict) else vars(obs)
        self.names, self.cache, self.first = [], {}, {}
        self.env_steps, self.success_step, self.invalid = 0, None, 0
        self.grip = 1.0 if float(self.obs["gripper_open"]) > 0.5 else 0.0
        self.table = self._estimate_table_z()
        self._home = self.hand_pose().tolist()
        # robot geometry: yaw offset between the tip's x axis and the finger closing axis (measured once)
        try:
            g = self.env.task._robot.gripper
            a, b = np.asarray(g.joints[0].get_position()), np.asarray(g.joints[1].get_position())
            d = (b - a)[:2]
            tip_yaw = quat_yaw(self.hand_pose()[3:])
            self.close_off = _wrap(float(np.arctan2(d[1], d[0])) - tip_yaw) if np.linalg.norm(d) > 1e-4 else 0.0
        except Exception:  # noqa: BLE001
            self.close_off = 0.0

    def register(self, names):
        for n in names:
            n = str(n).strip()
            if n and n not in self.names:
                self.names.append(n)

    # ---------------------------------------------------------------- sensing
    def refresh(self):
        self.n_render += 1          # self.obs is kept current by _step (env obs, re-rendered after gripper changes)

    def rgb(self):
        return np.asarray(self.obs["front_rgb"])

    def _estimate_table_z(self):
        pc = self._pc().reshape(-1, 3)
        m = np.all((pc >= BOUNDS[:3]) & (pc <= BOUNDS[3:]), axis=1)
        z = pc[m, 2]
        h, e = np.histogram(z, bins=np.arange(BOUNDS[2], BOUNDS[5], 0.005))
        return float(e[np.argmax(h)] + 0.0025)

    def cam_dirs(self):
        """unit xy vectors of image-right and image-away in the table frame (from the camera point cloud)."""
        pc = self._pc()
        H, W = pc.shape[:2]
        r = (pc[H // 2, int(W * 0.8)] - pc[H // 2, int(W * 0.2)])[:2]
        f = (pc[int(H * 0.35), W // 2] - pc[int(H * 0.75), W // 2])[:2]
        return r / np.linalg.norm(r), f / np.linalg.norm(f)

    VIEWS = ("front", "left_shoulder", "right_shoulder", "wrist")     # robot-mounted / fixed cameras of the env

    def _segment(self, prompt, view="front"):
        """all SAM3 masks for a text prompt on one camera: [(score, mask)] sorted by score (high first)."""
        import requests
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(self.obs[f"{view}_rgb"])).save(buf, format="PNG")
        r = requests.post(f"{self.sam3}/segment", json={"image_b64": base64.b64encode(buf.getvalue()).decode(),
                                                        "text_prompt": prompt}, timeout=60.0)
        r.raise_for_status()
        out = [(float(m.get("score", 0)), np.array(m["mask"], dtype=bool)) for m in r.json().get("masks", [])]
        return sorted([(sc, m) for sc, m in out if m.any()], key=lambda t: -t[0])

    def masks(self, prompt, min_score=0.5):
        """front-camera instance masks above min_score, left to right in the image (for '<noun> #i')."""
        ms = [m for sc, m in self._segment(prompt) if sc >= min_score]
        return sorted(ms, key=lambda m: np.nonzero(m)[1].mean())

    def _pc(self, view="front"):
        im = np.asarray(self.obs[f"{view}_rgb"])
        return np.asarray(self.obs[f"{view}_point_cloud"]).reshape(im.shape[0], im.shape[1], 3)

    def _points(self, mk, view):
        p = self._pc(view)[mk]
        p = p[np.all((p >= BOUNDS[:3]) & (p <= BOUNDS[3:]), axis=1)].copy()
        p[:, 2] -= self.table
        return p

    def _predicted(self, name):
        """where the object should be now: its last perceived centre, moved with the hand if the gripper closed on it
        since (perception + proprioception only)."""
        o = self.cache.get(name)
        if o is None:
            return None
        c = np.asarray(o["center"], float)
        tcp, w = self.hand_pose()[:3], self.width()
        if o.get("held_then"):
            return c + (tcp - np.asarray(o["hand_then"]))
        ca = o.get("closed_at")
        if ca is not None and o.get("grip_open_then") and CLOSED_W < w < OPEN_W and np.linalg.norm(np.asarray(ca)[:2] - c[:2]) < 0.05:
            return c + (tcp - np.asarray(ca))
        return c

    def _front_uv(self, p_table):
        """pixel of a table-frame point in the front camera (calibration only)."""
        E = np.asarray(self.obs["misc"]["front_camera_extrinsics"])
        K = np.asarray(self.obs["misc"]["front_camera_intrinsics"])
        pw = np.r_[p_table[0], p_table[1], p_table[2] + self.table, 1.0]
        pc = (np.linalg.inv(E) @ pw)[:3]
        uvw = K @ pc
        return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])

    def locate_mask(self, name):
        """(view, mask). '<noun> #i' = i-th front instance from the left. Otherwise the front camera's best SAM3 mask;
        once the object has been seen, the candidate nearest to where it should be now; if the front view has no
        plausible candidate (e.g. the object is inside the closed gripper and hidden by the hand) the shoulder / wrist
        cameras are tried the same way. Cameras + proprioception only."""
        import re
        m = re.fullmatch(r"(.+?)\s*#(\d+)", name.strip())
        if m:
            ms = self.masks(m.group(1))
            i = int(m.group(2)) - 1
            if not 0 <= i < len(ms):
                raise RuntimeError(f"only {len(ms)} instances of {m.group(1)}")
            return "front", ms[i]
        pred = self._predicted(name)
        fallback = None
        for view in self.VIEWS:
            cands = self._segment(name, view)
            if not cands:
                continue
            top = cands[0][0]
            if pred is None:
                if view == "front":
                    return view, cands[0][1]
                continue
            best, bd = None, 9.0
            for sc, mk in cands:
                if sc < 0.4 * top or sc < 0.1:
                    continue
                p = self._points(mk, view)
                if len(p) < 10:
                    continue
                d = float(np.linalg.norm(np.median(p, axis=0) - pred))
                if d < bd:
                    best, bd = mk, d
            if best is not None and bd < 0.08:
                return view, best
            if fallback is None and view == "front":
                fallback = (view, cands[0][1])
            if not (CLOSED_W < self.width() < OPEN_W):
                break                       # only a held object is expected to hide behind the hand
        if fallback is None:
            raise RuntimeError("not found")
        return fallback

    def mask(self, name):
        return self.locate_mask(name)[1]

    def perceive(self, names, refresh=True):
        if refresh:
            self.refresh()
        self.n_perceive += 1
        out = {}
        for n in names:
            try:
                view, mk = self.locate_mask(n)
                p = self._points(mk, view)
                if len(p) < 10:
                    raise RuntimeError("few points")
                p = _largest_components(p)
                lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
                ext = hi - lo
                if hi[2] < -0.03 or max(ext[0], ext[1]) > 0.5 or ext[2] > 0.4:
                    raise RuntimeError("implausible box")
                if view == "front":
                    vs, us = np.nonzero(mk)
                    u, v = float(us.mean()), float(vs.mean())
                else:
                    u, v = self._front_uv((lo + hi) / 2)
                o = {"min": lo, "max": hi, "center": (lo + hi) / 2, "u": u, "v": v, "npix": int(mk.sum()), "pts": p,
                     "view": view}
                tcp, w = self.hand_pose()[:3], self.width()
                o["hand_then"] = tcp
                o["held_then"] = bool(CLOSED_W < w < OPEN_W and np.all(tcp >= lo - 0.03) and np.all(tcp <= hi + 0.03))
                o["grip_open_then"] = bool(w >= OPEN_W)
                o["closed_at"] = None
                out[n] = o
                self.cache[n] = o
                self.first.setdefault(n, o)
            except Exception:  # noqa: BLE001
                out[n] = None
        return out

    def get(self, name, refresh=True):
        o = self.perceive([name], refresh)[name]
        return o if o is not None else self.cache.get(name)

    # ---------------------------------------------------------------- robot (proprioception)
    def hand_pose(self, arm=ARM):
        x, y, z, qx, qy, qz, qw = np.asarray(self.obs["gripper_pose"], float)
        return np.array([x, y, z - self.table, qw, qx, qy, qz])

    def tcp_point(self, pose7):
        return np.asarray(pose7, float)[:3]

    def width(self):
        return float(np.sum(self.obs["gripper_joint_positions"]))

    def opening(self, arm=ARM):
        return self.width() / 0.08

    def home_pose(self, arm=ARM):
        return list(self._home)

    def _step(self, pos_table, quat, grip):
        """one official env step; returns True if the env accepted the action."""
        if self.env_steps >= self.max_env_steps:
            self.trace.append({"skip": "env step budget"})
            return False
        p = np.asarray(pos_table, float).copy()
        p[2] += self.table
        q = np.asarray(quat if quat is not None else self.hand_pose()[3:], float)
        act = to_discrete(p, q, grip)
        changed = (1.0 if grip >= 0.5 else 0.0) != self.grip
        _, _, _, info = self.env.step(act)
        self.env_steps += 1
        self.obs = self.env.last_frame_obs
        if changed:     # the env captures its observation BEFORE actuating the gripper: re-render the cameras / proprio
            self.obs = vars(self.env.task._scene.get_observation())
            if grip < 0.5:
                tcp = self.hand_pose()[:3]
                for o in self.cache.values():
                    if o is not None and o.get("grip_open_then"):
                        o["closed_at"] = tcp
                        if np.linalg.norm(tcp[:2] - np.asarray(o["center"])[:2]) < 0.04:
                            o["grasp_off"] = tcp - np.asarray(o["center"])   # hand-to-object offset at closing
        ok = bool(info.get("action_success", 0.0) == 1.0)
        if not ok:
            self.invalid += 1
        if info.get("task_success", 0.0) == 1.0 and self.success_step is None:
            self.success_step = self.env_steps            # logging only
        self.grip = 1.0 if grip >= 0.5 else 0.0
        self.trace.append({"a": act, "ok": ok})
        return ok

    def move(self, p_table, quat=None):
        self._step(p_table, quat, self.grip)
        return float(np.linalg.norm(self.hand_pose()[:3] - np.asarray(p_table, float)[:3]))

    def run(self, items: dict) -> bool:
        """items {arm: [action dict]}; move {pose} / grip {val} / home. A grip right after a move is merged into that
        move's env step (the env closes/opens the gripper at the end of the planned path)."""
        ok = True
        for _, L in items.items():
            steps = []                          # [pose_table(3), quat(4) | None, grip, contact]
            g = self.grip
            for d in L:
                if d["type"] == "move":
                    pose = d["pose"]
                    steps.append([pose[:3], pose[3:7] if len(pose) >= 7 else None, g, bool(d.get("contact"))])
                elif d["type"] == "home":
                    h = self._home
                    steps.append([h[:3], h[3:], g, False])
                elif d["type"] == "grip":
                    g = 1.0 if d["val"] >= 0.5 else 0.0
                    if steps and steps[-1][2] != g and not steps[-1][3]:
                        steps[-1][2] = g
                    else:
                        cur = self.hand_pose() if not steps else np.r_[steps[-1][0], steps[-1][1] if steps[-1][1] is not None else self.hand_pose()[3:]]
                        steps.append([cur[:3], cur[3:], g, False])
            for p, q, gg, contact in steps:
                good = self._step(p, q, gg)
                err = float(np.linalg.norm(self.hand_pose()[:3] - np.asarray(p, float)))
                ok = ok and good and (err < 0.03 or contact)
        return ok

    # ---------------------------------------------------------------- scene interface for tools
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
        return False

    def joints(self, name):
        return []

    def table_z(self):
        return 0.0

    def held(self) -> dict:
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
        hz = max([0.12, z_goal + 0.04] + [t + 0.08 + extra_clear for t in tops])
        hz = min(hz, 0.40)
        cur = self.hand_pose()
        acts = []
        if cur[2] < hz - 0.02:
            acts.append({"type": "move", "pose": [cur[0], cur[1], hz, *quat]})
        acts.append({"type": "move", "pose": [xy[0], xy[1], hz, *quat]})
        return acts

    def _yaw_for_closing(self, closing_yaw):
        """tip yaw that puts the finger closing axis along closing_yaw (robot geometry offset), nearest to current."""
        yaw = closing_yaw - self.close_off
        cy = quat_yaw(self.hand_pose()[3:])
        if abs(_wrap(yaw + np.pi - cy)) < abs(_wrap(yaw - cy)):
            yaw += np.pi
        return _wrap(yaw)

    def grasp_plan(self, obj, arm=ARM, pre_dist=0.08, depth=0.0, grip=0.0, contact=None):
        """top-down grasp from SAM3 + depth (same rule as the Guava adapter): anchor = top-band point nearest the
        top-band centroid, closing axis = local PCA short axis, contact height = top - 1.5 cm (- depth)."""
        o = self.get(obj)
        if o is None:
            return None
        pc = o["pts"]
        top_z = float(pc[:, 2].max())
        band = pc[pc[:, 2] > top_z - 0.015]
        if len(band) < 10:
            band = pc
        bxy = band[:, :2]
        c = bxy.mean(0)
        dc = np.linalg.norm(bxy - c, axis=1)
        anchor = np.r_[c, top_z]
        local = band[dc < 0.03]
        if len(local) < 3:
            local = band
        xy = local[:, :2] - local[:, :2].mean(0)
        _, _, Vt = np.linalg.svd(xy, full_matrices=False)
        yaw = float(np.arctan2(Vt[1, 1], Vt[1, 0])) + (np.radians(contact) if contact else 0.0)
        q = yaw_quat(self._yaw_for_closing(yaw)).tolist()
        gz = max(top_z - 0.015 - depth, 0.01)
        goal = [float(anchor[0]), float(anchor[1]), gz]
        acts = [{"type": "grip", "val": 1.0}] if self.width() < OPEN_W else []
        acts += self._transit(goal[:2], top_z + pre_dist, q)
        acts += [{"type": "move", "pose": [*goal, *q]},
                 {"type": "grip", "val": float(grip)}]
        return acts

    def place_plan(self, obj, arm, target7, fp=None, pre_dist=0.08, dist=0.02, release=True, extra=None):
        """target7 = desired object-centre pose. Hand-to-object offset = the one measured when the gripper closed on it
        (perception before the grasp + proprioception), else from current perception. This simulator's grasp is an
        attach: opening the gripper drops the object straight down, so the release is done at the target itself
        (the target already carries the tool layer's clearance above the support); `dist` only shapes the approach."""
        c = self.cache.get(obj)
        hand = self.hand_pose()
        if c is not None and c.get("grasp_off") is not None:
            off = np.asarray(c["grasp_off"], float).copy()
        else:
            o = self.get(obj)
            off = np.zeros(3) if o is None else hand[:3] - o["center"]
        off[:2] = 0.0
        off[2] = float(np.clip(off[2], -0.03, 0.15))
        tgt = np.asarray(target7, float)[:3] + off
        q = hand[3:].tolist()
        acts = self._transit(tgt[:2], tgt[2] + pre_dist, q, extra_clear=max(0.0, off[2]))
        acts += [{"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + 0.005, *q]}]
        if release:
            acts += [{"type": "grip", "val": 1.0},
                     {"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + 0.08, *q]}]
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
        q = yaw_quat(self._yaw_for_closing(float(np.arctan2(d[1], d[0])) + np.pi / 2)).tolist()
        start = o["center"][:2] - d * (back + clearance)
        end = o["center"][:2] + d * (distance - back)
        acts = [{"type": "grip", "val": 0.0}] if self.width() > CLOSED_W + 0.004 else []
        acts += self._transit(start, z, q)
        acts += [{"type": "move", "pose": [start[0], start[1], z, *q]},
                 {"type": "move", "pose": [end[0], end[1], z, *q], "contact": True},
                 {"type": "move", "pose": [end[0], end[1], max(z + 0.10, h + 0.06), *q]}]
        return acts

    # ---------------------------------------------------------------- logging only (never shown to the planner)
    def success(self):
        return self.success_step is not None
