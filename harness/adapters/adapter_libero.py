"""LIBERO / LIBERO-PRO (robosuite, single Panda, OSC_POSE) adapter for the harness tool layer.

Same interface as adapter_guava.GuavaAdapter (hand_pose / run / opening / held / home_pose / grasp_plan / place_plan /
push_plan / objects / obj_pose / aabb / perceive / masks / cam_dirs / snapshot ...), so tools.Tools and the Guava
harness logic run unchanged on top of it.

Fairness rules:
  * object geometry ONLY from cameras: SAM3 text-prompt masks (local sam3_srv.py) on the rendered RGB of the scene's
    fixed cameras (agentview + frontview, 512 px) + the rendered depth of the same cameras, back-projected
    with the camera intrinsics / extrinsics and fused across views (mask erosion, workspace filter, 3D clustering).
    Object names come from the planner (open vocabulary), never from the simulator's object list. The table height
    is estimated from the agentview depth (dominant plane);
  * robot state = proprioception only (end-effector site pose, finger joint positions, finger body positions for the
    one-off tool-axis calibration);
  * motion = robosuite OSC_POSE delta servoing (env_libero.py LiberoEnv._servo / _set_gripper)
  * no simulator object state, contacts, joint values or BDDL predicate is read here (success() exists only for the
    episode runner's logging and is never shown to the planner);
  * no task names anywhere.
Frame (= Guava adapter): robot-base axes (+x away from the robot, +y robot's left, +z up), origin shifted so that z = 0
is the table top; metres. pose7 = [x, y, z, qw, qx, qy, qz]: xyz = TCP (fingertip centre = robosuite grip site),
quaternion = orientation of that site in the base frame."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

import numpy as np

ARM = "right"
OPEN_W, CLOSED_W = 0.075, 0.004   # finger width (m): >= open, <= closed-on-nothing
SAFE_Z = 0.20                     # default transit height of the TCP above the table (m)
VIEWS = ("agentview", "frontview")
IMG = 512
SAM3 = os.environ.get("LIBERO_SAM3_URL", "http://127.0.0.1:8134")


# ---------------------------------------------------------------- small rotation helpers
def _axang(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _m2q(R):
    import transforms3d as t3d
    q = t3d.quaternions.mat2quat(R)
    return q if q[0] >= 0 else -q


def _frame(approach, finger):
    a = np.asarray(approach, float)
    a = a / np.linalg.norm(a)
    f = np.asarray(finger, float)
    f = f - a * (a @ f)
    f = f / np.linalg.norm(f)
    return np.stack([a, f, np.cross(a, f)], axis=1)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _largest_components(p, voxel=0.015):
    from scipy import ndimage
    if len(p) < 20:
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
    return p[li == int(np.argmax(cnt))]


def _b64png(a):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def segment(rgb, prompts, top_k=6, min_score=0.15):
    from PIL import Image
    body = json.dumps({"image_b64": _b64png(rgb), "prompts": list(prompts), "top_k": top_k, "min_score": min_score}).encode()
    req = urllib.request.Request(SAM3 + "/seg", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())["results"]
    out = {}
    for p, lst in r.items():
        out[p] = [{"score": float(m["score"]), "mask": np.array(Image.open(io.BytesIO(base64.b64decode(m["mask"])))) > 127}
                  for m in lst]
    return out


def base_noun(name):
    m = re.fullmatch(r"(.+?)\s*#(\d+)", str(name).strip())
    return (m.group(1).strip().lower(), int(m.group(2))) if m else (str(name).strip().lower(), None)


def _servo1_fast(self, pt, Rt, max_ticks, loose=False):
    """env_libero.LiberoEnv._servo1 with a shorter stall window (blocked motions give up after 20 ticks = 1 s) and a
    hard stop at the episode deadline."""
    import time
    from bp_core import rotvec
    best, since, n = 1e9, 0, 0
    sid = self.robot.eef_site_id
    max_ticks = min(max_ticks, 200)
    dl = getattr(self, "deadline", None)
    while n < max_ticks:
        if dl is not None and time.time() > dl:
            break
        pc = self.sim.data.site_xpos[sid].copy()
        Rc = self.sim.data.site_xmat[sid].reshape(3, 3)
        ep = pt - pc
        er = rotvec(Rt @ Rc.T)
        e = np.linalg.norm(ep) + 0.02 * np.linalg.norm(er)
        if np.linalg.norm(ep) < (0.01 if loose else 0.003) and np.linalg.norm(er) < (0.06 if loose else 0.03):
            break
        if e < best - 2e-4:
            best, since = e, 0
        else:
            since += 1
            if since > 20:
                break
        a = np.concatenate([np.clip(ep / self.omax[:3], -1, 1), np.clip(er / self.omax[3:], -1, 1)])
        self._step(a)
        n += 1
    if loose:
        return n
    for _ in range(3):
        self._step(self._hold_action(pt, Rt))
    return n + 3


class LiberoAdapter:
    def __init__(self, env):
        """env: bench_ports env_libero.LiberoEnv (scene construction + OSC servo only)."""
        self.E = env
        self.names = []
        self.cache, self.first = {}, {}
        self.track = {}           # name -> (centre, tcp at that time, gripper closed on something?)
        self.order = {}           # noun -> list of instance centres at the first look (numbering of '#i')
        self.n_render = 0
        self._frame_key = None
        self._views = None
        self._segs = {}
        self.inject = None
        self.last_close = None
        self.deadline = None              # wall-clock end of the episode (motions stop there)
        self.cur_side = None              # top grasp at the object's edge on this side (xy unit vector) or None
        self.cur_approach_vec = None      # None = top-down grasp; else horizontal unit vector the fingers point along
        self.stats = {"perceive": 0, "perceive_s": 0.0, "sam3_calls": 0}

    # ---------------------------------------------------------------- episode
    def start_episode(self):
        E = self.E
        E.env.reset()
        E.env.set_init_state(E.init_state)
        e = E.env.env
        E.sim, E.robot = e.sim, e.robots[0]
        E._g = -1.0
        E._bid = E.sim.model.body_name2id(E.robot.robot_model.root_body)
        E.omax = np.asarray(E.robot.controller.output_max, float)
        import types
        E._servo1 = types.MethodType(_servo1_fast, E)
        for _ in range(32):                         # MotorMind settle_steps = 32
            E._step(np.zeros(6))
        self.names, self.cache, self.first, self.track, self.order = [], {}, {}, {}, {}
        self._frame_key = None
        self.tz = self._estimate_table_z()
        # tool axes from the finger geometry (proprioception, once): approach = finger base -> tip, finger = closing axis
        p, Rn = E._native_tcp()
        lf, rf = E._finger_points()
        mid = (np.asarray(lf) + np.asarray(rf)) / 2
        Rt = _frame(p - mid, np.asarray(lf) - np.asarray(rf))
        self.L = Rn.T @ Rt                          # native -> tool
        self._home = self.hand_pose().tolist()
        self.refresh()

    def register(self, names):
        for n in names:
            n = str(n).strip()
            if n and n not in self.names and n.lower() not in ("table", "world"):
                self.names.append(n)

    # ---------------------------------------------------------------- cameras
    def _sim_key(self):
        return float(self.E.sim.data.time)

    def _render_views(self):
        sim, out = self.E.sim, []
        m = sim.model
        ext = float(m.stat.extent)
        near, far = float(m.vis.map.znear) * ext, float(m.vis.map.zfar) * ext
        bp, bR = self.E._base()
        for cam in VIEWS:
            i = m.camera_name2id(cam)
            rgb, d = sim.render(camera_name=cam, width=IMG, height=IMG, depth=True)
            rgb, d = np.ascontiguousarray(rgb[::-1]), d[::-1]
            z = near / (1.0 - d * (1.0 - near / far))
            f = 0.5 * IMG / np.tan(np.radians(float(m.cam_fovy[i])) / 2)
            u, v = np.meshgrid(np.arange(IMG), np.arange(IMG))
            pc = np.stack([(u - IMG / 2 + 0.5) / f * z, -(v - IMG / 2 + 0.5) / f * z, -z], -1)
            R = sim.data.cam_xmat[i].reshape(3, 3)
            pw = pc.reshape(-1, 3) @ R.T + sim.data.cam_xpos[i]
            pb = (pw - bp) @ bR
            out.append({"name": cam, "rgb": rgb, "xyz": pb.reshape(IMG, IMG, 3), "valid": d < 0.9999,
                        "R": bR.T @ R, "pos": bR.T @ (sim.data.cam_xpos[i] - bp), "f": f})
        self.n_render += 1
        return out

    def _estimate_table_z(self):
        v = self._render_views()[0]
        p = v["xyz"][v["valid"]]
        p = p[(p[:, 0] > 0.3) & (p[:, 0] < 0.8) & (np.abs(p[:, 1]) < 0.35)]
        zs = np.round(p[:, 2], 3)
        vals, cnt = np.unique(zs, return_counts=True)
        return float(vals[np.argmax(cnt)])

    def refresh(self):
        k = self._sim_key()
        if self._frame_key != k:
            self._views = self._render_views()
            self._segs = {}
            self._frame_key = k

    def rgb(self):
        self.refresh()
        return self._views[0]["rgb"]

    def project(self, p_table, view=0):
        """table-frame point -> (u, v) pixel in the given view (default agentview)."""
        V = self._views[view]
        pc = V["R"].T @ (np.asarray(p_table, float) + [0, 0, self.tz] - V["pos"])
        if pc[2] >= -1e-6:
            return None
        u = V["f"] * pc[0] / (-pc[2]) + IMG / 2 - 0.5
        v = -V["f"] * pc[1] / (-pc[2]) + IMG / 2 - 0.5
        return float(u), float(v)

    def annotated_rgb(self):
        """agentview image with a small label at every object the planner has named (last perceived centre)."""
        from PIL import Image, ImageDraw
        im = Image.fromarray(self.rgb().copy())
        dr = ImageDraw.Draw(im)
        for n in self.names:
            o = self.cache.get(n)
            if o is None or any(m.startswith(n + " #") for m in self.names):
                continue
            uv = self.project([o["center"][0], o["center"][1], o["max"][2]])
            if uv is None:
                continue
            u, v = uv
            dr.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(255, 255, 0), outline=(0, 0, 0))
            x, y = u + 5, v - 12
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                dr.text((x + dx, y + dy), n, fill=(0, 0, 0))
            dr.text((x, y), n, fill=(255, 255, 0))
        return np.asarray(im)

    def crops(self, names, size=1024, box=0.09):
        """close-up tiles of named objects from a high-resolution agentview render (labelled), for identification."""
        from PIL import Image, ImageDraw
        sim = self.E.sim
        hi = np.ascontiguousarray(sim.render(camera_name=VIEWS[0], width=size, height=size)[::-1])
        tiles = []
        for n in names:
            o = self.cache.get(n) or self.get(n, refresh=False)
            if o is None:
                continue
            c = np.asarray(o["center"], float)
            r = max(float(np.max(o["max"] - o["min"])) / 2 + 0.02, box / 2)
            pts = [c + [dx, dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (-r, r)]
            uv = [self.project(p) for p in pts]
            uv = [u for u in uv if u is not None]
            if not uv:
                continue
            k = size / IMG
            us, vs = [u[0] * k for u in uv], [u[1] * k for u in uv]
            u0, u1, v0, v1 = max(0, min(us)), min(size, max(us)), max(0, min(vs)), min(size, max(vs))
            if u1 - u0 < 8 or v1 - v0 < 8:
                continue
            t = Image.fromarray(hi[int(v0):int(v1), int(u0):int(u1)]).resize((224, 224))
            ImageDraw.Draw(t).text((4, 4), n, fill=(255, 255, 0))
            tiles.append(t)
        if not tiles:
            return None
        W = Image.new("RGB", (224 * min(4, len(tiles)), 224 * ((len(tiles) + 3) // 4)))
        for i, t in enumerate(tiles):
            W.paste(t, ((i % 4) * 224, (i // 4) * 224))
        return np.asarray(W)

    def cam_dirs(self):
        """unit xy vectors of camera-right and camera-away (into the image) for the agentview camera."""
        self.refresh()
        R = self._views[0]["R"]                  # MuJoCo camera: x right, y up, -z forward
        r = R[:, 0][:2]
        f = -R[:, 2][:2]
        return r / np.linalg.norm(r), f / np.linalg.norm(f)

    # ---------------------------------------------------------------- perception
    def _seg(self, nouns):
        need = [n for n in nouns if n not in self._segs]
        if need:
            for vi, v in enumerate(self._views):
                s = segment(v["rgb"], need)
                self.stats["sam3_calls"] += 1
                for n in need:
                    self._segs.setdefault(n, [None] * len(self._views))[vi] = s.get(n, [])
        return {n: self._segs[n] for n in nouns}

    def _instances(self, noun):
        """fused 3D instances of a noun across the views: list of dicts sorted by support (desc)."""
        segs = self._seg([noun])[noun]
        C = []
        for v, lst in zip(self._views, segs):
            from scipy.ndimage import binary_erosion
            for m in lst or []:
                mk = m["mask"]
                e = binary_erosion(mk, iterations=2)
                mk = (e if e.sum() >= 15 else mk) & v["valid"]
                pts = v["xyz"][mk]
                z = pts[:, 2] - self.tz
                pts = pts[(z > 0.002) & (z < 0.45) & (pts[:, 0] > 0.12) & (pts[:, 0] < 1.0) & (np.abs(pts[:, 1]) < 0.6)]
                if len(pts) < 15:
                    continue
                pts = _largest_components(pts)
                if len(pts) < 15:
                    continue
                C.append({"view": v["name"], "score": m["score"], "pts": pts, "c": np.median(pts, 0),
                          "npix": int(m["mask"].sum())})
        cl = []
        for c in sorted(C, key=lambda c: -c["score"]):
            best, bd = None, 0.05
            for g in cl:
                d = float(np.linalg.norm(g["c"][:2] - c["c"][:2]))
                if d < bd and abs(float(g["c"][2] - c["c"][2])) < 0.035 and c["view"] not in g["views"]:
                    best, bd = g, d
            if best is None:
                cl.append({"c": c["c"], "m": [c], "views": {c["view"]}})
            else:
                best["m"].append(c)
                best["views"].add(c["view"])
        out = []
        for g in cl:
            sup = sum(m["score"] for m in g["m"])
            top = max(m["score"] for m in g["m"])
            if top < 0.35 and len(g["views"]) < 2:
                continue
            if top < 0.2:
                continue
            pts = np.concatenate([m["pts"] for m in g["m"]])
            pts[:, 2] -= self.tz
            lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
            if lo[2] < 0.008:
                lo[2] = 0.0
            ext = hi - lo
            if max(ext[0], ext[1]) > 0.5 or ext[2] > 0.45:
                continue
            out.append({"min": lo, "max": hi, "center": (lo + hi) / 2, "pts": pts, "support": sup * (1 + 0.25 * (len(g["views"]) - 1)),
                        "views": sorted(g["views"]), "score": top})
        # one physical object -> one instance (drop weaker duplicates that overlap strongly)
        out.sort(key=lambda o: -o["support"])
        keep = []
        for o in out:
            if any(np.linalg.norm(o["center"][:2] - k["center"][:2]) < 0.03 and abs(o["center"][2] - k["center"][2]) < 0.03 for k in keep):
                continue
            keep.append(o)
        return keep

    @staticmethod
    def _lr_order(insts):
        """left-to-right in the agentview image (= +y in the base frame), stacked ones (same xy) top to bottom."""
        return sorted(insts, key=lambda o: (round(float(o["center"][1]) / 0.04), -float(o["center"][2])))

    def _expected(self, name):
        t = self.track.get(name)
        if t is None:
            return None
        c, tcp0, closed = t
        tcp = self.hand_pose()[:3]
        if closed and self.width() > CLOSED_W and np.linalg.norm(c - tcp0) < 0.12:
            return c + (tcp - tcp0)                     # carried by the gripper since last seen
        return c

    def perceive(self, names, refresh=True):
        """{name: {'min','max','center','pts','support','views'} | None}; table frame."""
        import time
        t0 = time.time()
        if refresh:
            self.refresh()
        out = {}
        nouns = sorted({base_noun(n)[0] for n in names})
        try:
            self._seg(nouns)
        except Exception:  # noqa: BLE001
            return {n: None for n in names}
        for n in names:
            noun, idx = base_noun(n)
            insts = self._instances(noun)
            insts = [i for i in insts if not self._taken_by_other(n, noun, i)]
            if noun not in self.order and insts:
                self.order[noun] = [o["center"].copy() for o in self._lr_order(insts)]
            o = None
            exp = self._expected(n)
            if exp is not None and insts:
                carried = self.track[n][2]
                d = [float(np.linalg.norm((i["center"] - exp)[:2 if carried else 3])) for i in insts]
                j = int(np.argmin(d))
                if d[j] < 0.10:
                    o = insts[j]
                elif carried:       # lost after being carried (released / dropped elsewhere): the instance no other
                    # name of the same noun claims, nearest to the gripper
                    claimed = [self._expected(m) for m in self.track if m != n and base_noun(m)[0] == noun]
                    free = [i for i in insts if all(c is None or np.linalg.norm(i["center"][:2] - c[:2]) > 0.05 for c in claimed)]
                    if free:
                        tcp = self.hand_pose()[:3]
                        o = min(free, key=lambda i: float(np.linalg.norm(i["center"][:2] - tcp[:2])))
                        if np.linalg.norm(o["center"][:2] - tcp[:2]) > 0.15 and len(free) > 1:
                            o = None
                else:               # not carried: it may have been bumped / its earlier estimate was occluded;
                    # never take an instance another name of the same noun is tracking
                    claimed = [self._expected(m) for m in self.track if m != n and base_noun(m)[0] == noun]
                    free = [(dd, i) for dd, i in zip(d, insts)
                            if all(c is None or np.linalg.norm(i["center"] - c) > 0.04 for c in claimed)]
                    if free and min(free, key=lambda x: x[0])[0] < 0.25:
                        o = min(free, key=lambda x: x[0])[1]
                    elif idx is None and len(insts) == 1:
                        o = insts[0]
            elif idx is not None and insts:
                ref = self.order.get(noun, [])
                if 1 <= idx <= len(ref):
                    d = [float(np.linalg.norm(i["center"] - ref[idx - 1])) for i in insts]
                    j = int(np.argmin(d))
                    if d[j] < 0.12:
                        o = insts[j]
            elif insts:
                o = insts[0]
            if o is not None:
                o = dict(o)
                self.cache[n] = o
                self.first.setdefault(n, o)
                self.track[n] = (o["center"].copy(), self.hand_pose()[:3].copy(), self.CLOSED_ON_SOMETHING())
            out[n] = o
        self.stats["perceive"] += 1
        self.stats["perceive_s"] += time.time() - t0
        return out

    def _taken_by_other(self, n, noun, inst):
        """one physical object -> one name: an instance that another, differently-named object already occupies with a
        better segmentation score belongs to that name (e.g. a bowl also scoring as 'plate')."""
        for m, o in self.cache.items():
            if m == n or base_noun(m)[0] == noun or o is None:
                continue
            if (np.linalg.norm(o["center"][:2] - inst["center"][:2]) < 0.03 and abs(o["center"][2] - inst["center"][2]) < 0.03
                    and o.get("score", 0) >= inst["score"]):
                return True
        return False

    def CLOSED_ON_SOMETHING(self):
        w = self.width()
        return CLOSED_W < w < OPEN_W

    def instances(self, noun):
        """all instances of a noun (numbered left to right as in the first look): [(i, inst)]."""
        self.refresh()
        noun = base_noun(noun)[0]
        insts = self._instances(noun)
        if noun not in self.order and insts:
            self.order[noun] = [o["center"].copy() for o in self._lr_order(insts)]
        ref = self.order.get(noun, [])
        res = []
        for o in insts:
            d = [float(np.linalg.norm(o["center"] - r)) for r in ref]
            res.append((int(np.argmin(d)) + 1 if d else 0, o))
        return sorted(res, key=lambda x: x[0])

    def masks(self, prompt, min_score=0.5):
        return [o for _, o in self.instances(prompt)]

    def get(self, name, refresh=True):
        o = self.perceive([name], refresh)[name]
        return o if o is not None else self.cache.get(name)

    # ---------------------------------------------------------------- robot (proprioception)
    def hand_pose(self, arm=ARM):
        p, Rn = self.E._native_tcp()
        return np.concatenate([np.asarray(p, float) - [0, 0, self.tz], _m2q(Rn)])

    def tool_R(self):
        return self.E._native_tcp()[1] @ self.L

    def tcp_point(self, pose7):
        return np.asarray(pose7, float)[:3]

    def width(self):
        return float(self.E._opening()) * 0.08

    def opening(self, arm=ARM):
        return float(self.E._opening())

    def home_pose(self, arm=ARM):
        return list(self._home)

    def quat_for(self, approach, closing):
        """native-site quaternion for a tool frame with this approach axis and closing (finger) axis."""
        Rt = _frame(approach, closing)
        return _m2q(Rt @ self.L.T)

    def down_quat(self, closing_yaw):
        return self.quat_for([0, 0, -1], [np.cos(closing_yaw), np.sin(closing_yaw), 0.0])

    def closing_yaw(self):
        f = self.tool_R()[:, 1]
        return float(np.arctan2(f[1], f[0]))

    def move(self, p_table, quat=None):
        p = np.asarray(p_table, float)[:3] + [0, 0, self.tz]
        R = self.E._native_tcp()[1] if quat is None else _q2m(np.asarray(quat, float))
        self.E._servo(p, R)
        return float(np.linalg.norm(self.hand_pose()[:3] - np.asarray(p_table, float)[:3]))

    def run(self, items: dict) -> bool:
        import time
        if self.deadline is not None and time.time() > self.deadline:
            return False
        ok = True
        for _, L in items.items():
            for d in L:
                if d["type"] == "move":
                    err = self.move(d["pose"][:3], d["pose"][3:7] if len(d["pose"]) >= 7 else None)
                    ok = ok and (err < 0.03 or bool(d.get("contact")))
                elif d["type"] == "home":
                    h = self.hand_pose()
                    self.move([h[0], h[1], max(h[2], SAFE_Z)])
                    self.move(self._home[:3], self._home[3:])
                elif d["type"] == "grip":
                    self.E._set_gripper(float(d["val"]))
                    if d["val"] < 0.5:
                        self.last_close = round(self.width(), 4)
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
        tops = [float(o["max"][2]) for o in self.cache.values() if o is not None
                and np.linalg.norm(np.asarray(o["center"][:2]) - np.asarray(xy[:2])) < 0.5]
        hz = max([SAFE_Z, z_goal + 0.04] + [t + 0.08 + extra_clear for t in tops])
        hz = min(hz, 0.42)
        cur = self.hand_pose()
        acts = []
        if cur[2] < hz - 0.01:
            acts.append({"type": "move", "pose": [cur[0], cur[1], hz, *cur[3:]]})
        acts.append({"type": "move", "pose": [xy[0], xy[1], hz, *quat]})
        return acts

    def _pick_yaw(self, yaw):
        """closing axis is symmetric: choose yaw or yaw+pi, whichever needs less wrist rotation."""
        cy = self.closing_yaw()
        return yaw + np.pi if abs(_wrap(yaw + np.pi - cy)) < abs(_wrap(yaw - cy)) else yaw

    def grasp_plan(self, obj, arm=ARM, pre_dist=0.1, depth=0.0, grip=0.0, contact=None, approach=None):
        """approach 'top': PCA top-down grasp (the Guava rule: anchor = top-band point closest to the band centroid,
        closing axis = local PCA short axis, contact a little below the top).
        approach = a horizontal xy unit vector (side grasp, e.g. a handle on a vertical face): the gripper comes along
        that vector toward the object's nearest face, closing axis = perpendicular to the object's long axis."""
        if approach is None or isinstance(approach, str):
            approach = self.cur_approach_vec
        o = self.get(obj)
        if o is None:
            return None
        pc = o["pts"]
        h = float(o["max"][2] - o["min"][2])
        if approach is None:
            top_z = float(np.percentile(pc[:, 2], 99))
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
            yaw = float(np.arctan2(Vt[1, 1], Vt[1, 0])) + (np.radians(contact) if contact else 0.0)
            if self.cur_side is not None:        # grasp the edge / rim on that side, fingers across the edge
                v = np.asarray(self.cur_side, float)[:2]
                pr = bxy @ v
                edge = band[pr >= np.percentile(pr, 97)]
                anchor = edge.mean(0)
                anchor[:2] -= v * 0.012
                yaw = float(np.arctan2(v[1], v[0])) + (np.radians(contact) if contact else 0.0)
            yaw = self._pick_yaw(yaw)
            q = self.down_quat(yaw).tolist()
            gz = max(top_z - float(np.clip(0.35 * h, 0.01, 0.035)) - depth, 0.008)
            goal = [float(anchor[0]), float(anchor[1]), gz]
            acts = [{"type": "grip", "val": 1.0}] if self.width() < OPEN_W else []
            acts += self._transit(goal[:2], top_z + pre_dist, q)
            acts += [{"type": "move", "pose": [goal[0], goal[1], top_z + pre_dist, *q]},
                     {"type": "move", "pose": [*goal, *q]},
                     {"type": "grip", "val": float(grip)}]
            return acts
        d = np.asarray(approach, float)[:2]
        d = d / np.linalg.norm(d)                       # direction the fingers point (toward the object)
        a3 = np.array([d[0], d[1], 0.0])
        c = np.median(pc, 0)
        X = pc - c
        X = X - np.outer(X @ a3, a3)                    # remove the approach component
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        long_ax = Vt[0]
        closing = np.cross(a3, long_ax)
        if np.linalg.norm(closing) < 1e-6:
            closing = np.array([0, 0, 1.0])
        if abs(closing[2]) < 0.5 and abs(np.cross(a3, [0, 0, 1])[2]) < 1e-9:
            pass
        if contact:
            closing = _axang(a3, np.radians(contact)) @ closing
        s = (pc - c) @ a3
        face = c + a3 * float(np.percentile(s, 5))      # the object's surface facing the gripper
        goal = face + a3 * 0.015                         # fingertips 1.5 cm past the near surface
        q = self.quat_for(a3, closing)
        Rcur = self.tool_R()
        if (Rcur[:, 1] @ closing) < 0:                   # symmetric gripper: keep the wrist turn small
            q = self.quat_for(a3, -closing)
        q = np.asarray(q).tolist()
        pre = goal - a3 * pre_dist
        acts = [{"type": "grip", "val": 1.0}] if self.width() < OPEN_W else []
        acts += self._transit(pre[:2], float(pre[2]), self.hand_pose()[3:].tolist())
        acts += [{"type": "move", "pose": [*pre, *q]},
                 {"type": "move", "pose": [*goal, *q], "contact": True},
                 {"type": "grip", "val": float(grip)}]
        return acts

    def place_plan(self, obj, arm, target7, fp=None, pre_dist=0.1, dist=0.02, release=True, extra=None):
        """target7 = desired object-centre pose (table frame); TCP-to-object offset measured from perception."""
        o = self.get(obj)
        hand = self.hand_pose()
        if o is None:
            off = np.array([0, 0, 0.0])
        else:
            off = hand[:3] - o["center"]
            if np.linalg.norm(off[:2]) > 0.08:          # implausible (occlusion): assume it hangs below the TCP
                off[:2] = 0.0
            off[2] = float(np.clip(off[2], 0.0, 0.15))
        tgt = np.asarray(target7, float)[:3] + off
        q = hand[3:].tolist()
        acts = self._transit(tgt[:2], tgt[2] + pre_dist, q, extra_clear=max(0.0, off[2]))
        acts += [{"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + pre_dist, *q]},
                 {"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + dist, *q], "contact": True}]
        if release:
            acts += [{"type": "grip", "val": 1.0},
                     {"type": "move", "pose": [tgt[0], tgt[1], tgt[2] + dist + 0.10, *q]}]
        return acts

    def push_plan(self, obj, direction_xy, distance, height=None, clearance=0.05):
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
        z = float(np.clip(height if height is not None else zc, max(0.006, float(o["min"][2]) + 0.006), max(0.006, h - 0.008)))
        q = self.down_quat(self._pick_yaw(float(np.arctan2(d[1], d[0])) + np.pi / 2)).tolist()
        start = o["center"][:2] - d * (back + clearance)
        end = o["center"][:2] + d * (distance - back)
        acts = [{"type": "grip", "val": 0.0}] if self.width() > CLOSED_W + 0.004 else []
        acts += self._transit(start, z, q)
        acts += [{"type": "move", "pose": [start[0], start[1], z, *q]}]
        n = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.04)))
        for i in range(1, n + 1):
            p = start + (end - start) * i / n
            acts.append({"type": "move", "pose": [p[0], p[1], z, *q], "contact": True})
        back_p = end - d * 0.04
        acts += [{"type": "move", "pose": [back_p[0], back_p[1], z, *q], "contact": True},
                 {"type": "move", "pose": [back_p[0], back_p[1], max(z + 0.12, h + 0.08), *q]}]
        return acts

    def twist_plan(self, degrees):
        """rotate the gripper about its approach axis (through the TCP) by `degrees` (positive = counter-clockwise
        seen from the side the gripper comes from, i.e. from above for a top-down gripper), in <= 15 deg steps."""
        h = self.hand_pose()
        Rn = _q2m(h[3:])
        a = self.tool_R()[:, 0]
        n = max(1, int(np.ceil(abs(degrees) / 15.0)))
        acts = []
        for i in range(1, n + 1):
            R = _axang(-a, np.radians(degrees) * i / n) @ Rn
            acts.append({"type": "move", "pose": [*h[:3], *_m2q(R)], "contact": True})
        return acts

    # ---------------------------------------------------------------- logging only (never shown to the planner)
    def success(self):
        return bool(self.E.check_success())
