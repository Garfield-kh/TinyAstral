"""Guava (robosuite / MuJoCo, single Panda) adapter for the harness tool layer.

The ONLY harness module that touches the Guava simulator. Implements the adapter interface used by tools.Tools
(hand_pose / tcp_point / run / opening / held / home_pose / grasp_plan / place_plan / objects / obj_pose / aabb /
is_articulated / joints / table_z / snapshot) plus push_plan / slide_plan for the push / articulate primitives.

Fairness rules (same information as the Guava-4B harness):
  * object geometry comes ONLY from Guava's own perception chain: SAM3 mask (ctx.perception.get_mask) on the
    task camera RGB + metric depth unprojection (guava.tools.tools._unproject_pointclouds, the same outlier filter);
  * robot state = proprioception (TCP pose, finger width) only;
  * motion = Guava RobotArm.move (pybullet IK, same motion speed) / RobotArm.grasp (close-until-stall) /
    RobotArm.release; the grasp pose is Guava's own PCA top-grasp rule;
  * no simulator object state, contacts, joint values or task score is read here (success() exists only for the
    episode runner's logging and is never shown to the planner);
  * no task names anywhere.
Frame: table-aligned (x, y = robot base, z = 0 at the tabletop), metres; pose7 = [x, y, z, qw, qx, qy, qz] where xyz
is the TCP (fingertip centre) and the quaternion is the end-effector orientation in the base frame."""
from __future__ import annotations

import numpy as np

ARM = "right"                     # the single Panda arm is called "right" for the shared tool layer
OPEN_W, CLOSED_W = 0.075, 0.002   # finger width (m): >= open, <= closed-on-nothing
SAFE_Z = 0.22                     # default transit height of the TCP above the table (m)


def yaw_quat(yaw):
    """straight-down gripper (roll 180, pitch 0) with the given yaw (Guava's convention)."""
    return np.array([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])


def quat_yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _largest_components(p, voxel=0.02):
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


class GuavaAdapter:
    def __init__(self, engine):
        from guava.inference import sft_frame
        self.E, self.env, self.arm, self.ctx = engine, engine.env, engine.arm, engine.ctx
        self.off = sft_frame.measured_offset(self.env)          # table z = base z + off
        self.names = []                                         # objects the planner has referred to (open vocabulary)
        self.cache = {}                                         # name -> last perception (for held / occluded objects)
        self.first = {}                                         # name -> first perception of the episode
        self.n_render = 0
        self._home = None
        self.inject = None                                      # Optional perturbation hook; disabled by default.

    # ---------------------------------------------------------------- episode
    def start_episode(self):
        self.names, self.cache, self.first = [], {}, {}
        self.refresh()
        p, q = self.env.eef_pose_base()
        self._home = np.concatenate([np.asarray(p) + [0, 0, self.off], q]).tolist()

    def register(self, names):
        for n in names:
            n = str(n).strip()
            if n and n not in self.names:
                self.names.append(n)

    # ---------------------------------------------------------------- sensing
    def refresh(self):
        from guava.tools.tools import _refresh
        _refresh(self.ctx)
        self.n_render += 1

    def rgb(self):
        return self.ctx.last_rgb

    def cam_dirs(self):
        """unit xy vectors of camera-right and camera-away (into the image) in the table frame."""
        R = np.asarray(self.ctx.last_pose_mat)[:3, :3]
        r = R[:, 0][:2]
        f = R[:, 2][:2]
        return r / np.linalg.norm(r), f / np.linalg.norm(f)

    def masks(self, prompt, min_score=0.5):
        """all SAM3 instance masks for a prompt (same server / model as Guava), sorted left-to-right in the image."""
        import base64, io, requests
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(self.ctx.last_rgb).save(buf, format="PNG")
        r = requests.post(f"{self.ctx.perception.url}/segment", json={"image_b64": base64.b64encode(buf.getvalue()).decode(),
                                                                       "text_prompt": prompt}, timeout=30.0)
        r.raise_for_status()
        ms = [m for m in r.json().get("masks", []) if m.get("score", 0) >= min_score]
        ms = [np.array(m["mask"], dtype=bool) for m in ms]
        ms = [m for m in ms if m.any()]
        return sorted(ms, key=lambda m: np.nonzero(m)[1].mean())

    def mask(self, name):
        """'<noun> #i' = i-th instance (1-based, left to right in the image); otherwise Guava's best mask."""
        import re
        m = re.fullmatch(r"(.+?)\s*#(\d+)", name.strip())
        if m:
            ms = self.masks(m.group(1))
            i = int(m.group(2)) - 1
            if not 0 <= i < len(ms):
                raise RuntimeError(f"only {len(ms)} instances of {m.group(1)}")
            return ms[i]
        return self.ctx.perception.get_mask(name, self.ctx.last_rgb)

    def perceive(self, names, refresh=True):
        """{name: {'min','max','center','u','v','npix'} | None}; table frame. Guava SAM3 + depth chain."""
        from guava.tools.tools import _unproject_pointclouds, _remove_statistical_outliers
        if refresh:
            self.refresh()
        c = self.ctx
        out = {}
        for n in names:
            try:
                mask = self.mask(n)
                _, pc = _unproject_pointclouds(c.last_depth, c.last_K, mask)
                if pc.shape[0] < 10:
                    raise RuntimeError("few points")
                pc = _remove_statistical_outliers(pc, k=20, std_ratio=2.0)
                pcb = (c.last_pose_mat @ np.hstack([pc, np.ones((len(pc), 1))]).T).T[:, :3]
                pcb[:, 2] += self.off
                pcb = _largest_components(pcb)
                lo, hi = np.percentile(pcb, 2, axis=0), np.percentile(pcb, 98, axis=0)
                ext = hi - lo
                if hi[2] < -0.03 or max(ext[0], ext[1]) > 0.45 or ext[2] > 0.40:
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

    # ---------------------------------------------------------------- robot (proprioception)
    def hand_pose(self, arm=ARM):
        p, q = self.env.eef_pose_base()
        return np.concatenate([np.asarray(p, float) + [0, 0, self.off], np.asarray(q, float)])

    def tcp_point(self, pose7):
        return np.asarray(pose7, float)[:3]

    def width(self):
        return float(self.arm.gripper_width())

    def opening(self, arm=ARM):
        return self.width() / 0.08

    def home_pose(self, arm=ARM):
        return list(self._home)

    def move(self, p_table, quat=None):
        p = np.asarray(p_table, float).copy()
        p[2] -= self.off
        self.arm.move(p.tolist(), None if quat is None else np.asarray(quat, float))
        return float(np.linalg.norm(self.hand_pose()[:3] - np.asarray(p_table, float)[:3]))

    def run(self, items: dict) -> bool:
        """items {arm: [action dict]}; action types: move {pose}, grip {val}, home. Returns reachability."""
        ok = True
        for _, L in items.items():
            for d in L:
                if d["type"] == "move":
                    err = self.move(d["pose"][:3], d["pose"][3:7] if len(d["pose"]) >= 7 else None)
                    ok = ok and (err < 0.03 or bool(d.get("contact")))
                elif d["type"] == "home":
                    self.arm.home_pose()
                elif d["type"] == "grip":
                    if d["val"] < 0.5:
                        self.last_close = self.arm.grasp()
                    else:
                        self.arm.release()
        return ok

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
        """lift to a transit height that clears every known object, move over xy, return the action list."""
        tops = [float(o["max"][2]) for o in self.cache.values() if o is not None]
        hz = max([SAFE_Z, z_goal + 0.06] + [t + 0.10 + extra_clear for t in tops])
        hz = min(hz, 0.45)
        cur = self.hand_pose()
        acts = []
        if cur[2] < hz - 0.01:
            acts.append({"type": "move", "pose": [cur[0], cur[1], hz, *quat]})
        acts.append({"type": "move", "pose": [xy[0], xy[1], hz, *quat]})
        return acts

    def grasp_plan(self, obj, arm=ARM, pre_dist=0.1, depth=0.0, grip=0.0, contact=None):
        """Guava's PCA top-down grasp (guava.tools.tools.grasp, PCA path) computed from SAM3 + depth:
        anchor = top surface point closest to the top-band centroid, closing axis = local PCA short axis,
        contact height = top - 1 cm (- depth). contact: optional yaw offset in degrees (re-grasp variation)."""
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
        cy = quat_yaw(cur[3:])
        if abs(_wrap(yaw + np.pi - cy)) < abs(_wrap(yaw - cy)):
            yaw += np.pi
        q = yaw_quat(yaw).tolist()
        gz = max(top_z - 0.01 - depth, 0.008)
        goal = [float(anchor[0]), float(anchor[1]), gz]
        if self.inject and self.inject[0] == "grasp_offset":     # injected miss: descend beside the object
            goal[0] += float(self.inject[1][0])
            goal[1] += float(self.inject[1][1])
        acts = [{"type": "grip", "val": 1.0}] if self.width() < OPEN_W else []
        acts += self._transit(goal[:2], top_z + pre_dist, q, extra_clear=0.0)
        acts += [{"type": "move", "pose": [goal[0], goal[1], top_z + pre_dist, *q]},
                 {"type": "move", "pose": [*goal, *q]},
                 {"type": "grip", "val": float(grip)}]
        return acts

    def place_plan(self, obj, arm, target7, fp=None, pre_dist=0.1, dist=0.02, release=True, extra=None):
        """target7 = desired object-centre pose (table frame). The hand goes where the held object's centre lands on the
        target (TCP-to-object offset measured from perception at execution time), then optionally opens and retreats."""
        o = self.get(obj)
        hand = self.hand_pose()
        if o is None:
            off = np.array([0, 0, 0.0])
        else:
            off = hand[:3] - o["center"]
            off[:2] = 0.0                       # top grasp: the object hangs below the TCP
            off[2] = float(np.clip(off[2], 0.0, 0.15))
        tgt = np.asarray(target7, float)[:3] + off
        q = hand[3:].tolist()
        acts = self._transit(tgt[:2], tgt[2] + pre_dist, q, extra_clear=max(0.0, off[2]))
        acts += [{"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + pre_dist, *q]},
                 {"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + dist, *q]}]
        if release:
            acts += [{"type": "grip", "val": 1.0},
                     {"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + dist + 0.10, *q]}]
        return acts

    def push_plan(self, obj, direction_xy, distance, height=None, clearance=0.05):
        """contact the side of obj opposite to direction with the closed gripper (wide face toward the object), sweep
        `distance` along direction, back off and lift."""
        o = self.get(obj)
        if o is None:
            return None
        d = np.asarray(direction_xy, float)
        d = d / np.linalg.norm(d)
        pts = o["pts"][:, :2]
        proj = (pts - o["center"][:2]) @ d
        back = float(-proj.min())                    # extent behind the centre along -d
        h = float(o["max"][2])
        zc = float(o["center"][2])
        z = float(np.clip(height if height is not None else zc, max(0.015, float(o["min"][2]) + 0.01), max(0.015, h - 0.01)))
        yaw = float(np.arctan2(d[1], d[0])) + np.pi / 2   # closing axis perpendicular to the push => wide face forward
        cy = quat_yaw(self.hand_pose()[3:])
        if abs(_wrap(yaw + np.pi - cy)) < abs(_wrap(yaw - cy)):
            yaw += np.pi
        q = yaw_quat(yaw).tolist()
        start = o["center"][:2] - d * (back + clearance)
        end = o["center"][:2] + d * (distance - back)        # TCP stops when the object front moved `distance`
        acts = [{"type": "grip", "val": 0.0}] if self.width() > CLOSED_W + 0.004 else []
        acts += self._transit(start, z, q)
        acts += [{"type": "move", "pose": [start[0], start[1], z, *q]}]
        n = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.04)))
        for i in range(1, n + 1):
            p = start + (end - start) * i / n
            acts.append({"type": "move", "pose": [p[0], p[1], z, *q], "contact": True})
        back_p = end - d * 0.05
        acts += [{"type": "move", "pose": [back_p[0], back_p[1], z, *q]},
                 {"type": "move", "pose": [back_p[0], back_p[1], max(z + 0.12, h + 0.08), *q]}]
        return acts

    # ---------------------------------------------------------------- logging only (never shown to the planner)
    def success(self):
        try:
            return bool(self.env.rob.score()["success"])
        except Exception:  # noqa: BLE001
            return False
