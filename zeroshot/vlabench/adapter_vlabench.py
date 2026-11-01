"""VLABench (MuJoCo / dm_control, single Franka Panda) adapter for the TinyAstral single-arm harness (harness/).

Same interface as adapter_libero.LiberoAdapter (it IS a LiberoAdapter: perception fusion, grasp / place / push / twist
plans, held-object tracking and the tool-layer hooks are inherited unchanged); only the simulator access differs:

  * cameras: the fixed scene cameras of the VLABench arena (right, left and the task view "forward"; the wrist camera
    is not used), RGB + metric depth rendered by dm_control, back-projected with the camera fovy / pose;
  * robot: env_vlabench.VLABenchEnv (end-effector site pose, finger joints, IK + joint position targets);
  * frame (= Guava / LIBERO adapters): robot-base axes (Franka link0: +x forward, +y robot's left, +z up), origin
    shifted so z = 0 is the support surface in front of the robot (dominant depth plane), metres.

Fairness / zero-shot rules (hard):
  * object geometry ONLY from SAM3 text-prompt masks on the rendered RGB + rendered depth (no segmentation renders,
    no entity list, no target_entity / conditions from the episode config, no object poses);
  * object names come from the planner (open vocabulary); the planner sees only the official instruction string;
  * no task names anywhere in the adapter, the driver or the prompts;
  * success() = the official termination flag, read only by the episode driver after each call (never shown to the
    planner); progress / intention scores are computed after the episode."""
from __future__ import annotations

import numpy as np

import adapter_libero as AL
from adapter_libero import LiberoAdapter, _frame, _largest_components, base_noun  # noqa: F401

ARM = AL.ARM
OPEN_W, CLOSED_W, SAFE_Z = AL.OPEN_W, AL.CLOSED_W, AL.SAFE_Z
IMG = 512
# perception workspace in the table frame (wider and taller than LIBERO: shelves / fridges / counters)
WS_X = (0.05, 1.25)
WS_Y = 0.85
WS_ZMAX = 0.9
MAX_EXT_XY, MAX_EXT_Z = 0.6, 0.7


class VLABenchAdapter(LiberoAdapter):
    def __init__(self, env):
        super().__init__(env)
        self.views_idx = list(env.fixed_cams)
        # main (planner) view first: the task camera (index 2 in the VLABench arena), then the others
        if 2 in self.views_idx:
            self.views_idx.remove(2)
            self.views_idx.insert(0, 2)

    # ---------------------------------------------------------------- episode
    def start_episode(self):
        E = self.E
        self.names, self.cache, self.first, self.track, self.order = [], {}, {}, {}, {}
        self._frame_key = None
        E._set_gripper(1.0)
        self.tz = self._estimate_table_z()
        p, Rn = E._native_tcp()
        lf, rf = E._finger_points()
        mid = (np.asarray(lf) + np.asarray(rf)) / 2
        Rt = _frame(p - mid, np.asarray(lf) - np.asarray(rf))      # tool axes from finger geometry (proprioception)
        self.L = Rn.T @ Rt
        self._home = self.hand_pose().tolist()
        self.refresh()

    # ---------------------------------------------------------------- cameras
    def _sim_key(self):
        return (float(self.E.physics.data.time), self.E.n_steps)

    def _render_views(self):
        out = []
        bp, bR = self.E._base()
        u, v = np.meshgrid(np.arange(IMG), np.arange(IMG))
        for ci in self.views_idx:
            rgb = np.ascontiguousarray(self.E.render(ci, IMG))
            z = np.asarray(self.E.render(ci, IMG, depth=True), float)
            cp, cR, fovy = self.E.cam_pose(ci)
            f = 0.5 * IMG / np.tan(np.radians(fovy) / 2)
            pc = np.stack([(u - IMG / 2 + 0.5) / f * z, -(v - IMG / 2 + 0.5) / f * z, -z], -1)
            pw = pc.reshape(-1, 3) @ cR.T + cp
            pb = (pw - bp) @ bR
            out.append({"name": self.E.cam_names[ci] or f"cam{ci}", "rgb": rgb, "xyz": pb.reshape(IMG, IMG, 3),
                        "valid": (z > 0.05) & (z < 6.0), "R": bR.T @ cR, "pos": bR.T @ (cp - bp), "f": f, "ci": ci})
        self.n_render += 1
        return out

    def _estimate_table_z(self):
        """the support surface the robot works on: dominant horizontal plane in front of the robot within +-15 cm of
        the robot's mounting height (all fixed views; a shelf / fridge top higher up is not taken)."""
        p = np.concatenate([v["xyz"][v["valid"]] for v in self._render_views()])
        p = p[(p[:, 0] > 0.15) & (p[:, 0] < 1.0) & (np.abs(p[:, 1]) < 0.6) & (np.abs(p[:, 2]) < 0.15)]
        if len(p) < 100:
            return 0.0
        zs = np.round(p[:, 2], 2)
        vals, cnt = np.unique(zs, return_counts=True)
        z0 = float(vals[np.argmax(cnt)])
        near = p[np.abs(p[:, 2] - z0) < 0.006, 2]
        return float(np.median(near)) if len(near) else z0

    def crops(self, names, size=1024, box=0.09):
        """close-up tiles of named objects from a high-resolution render of the main view (labelled)."""
        from PIL import Image, ImageDraw
        g = self.E.physics.model.vis.global_
        size = int(min(size, g.offwidth, g.offheight))          # offscreen framebuffer limit of the scene (960 px)
        hi = np.ascontiguousarray(self.E.render(self.views_idx[0], size))
        tiles = []
        for n in names:
            o = self.cache.get(n) or self.get(n, refresh=False)
            if o is None:
                continue
            c = np.asarray(o["center"], float)
            r = max(float(np.max(o["max"] - o["min"])) / 2 + 0.02, box / 2)
            pts = [c + [dx, dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (-r, r)]
            uv = [self.project(p) for p in pts]
            uv = [x for x in uv if x is not None]
            if not uv:
                continue
            k = size / IMG
            us, vs = [x[0] * k for x in uv], [x[1] * k for x in uv]
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

    # ---------------------------------------------------------------- perception (LIBERO fusion, wider workspace)
    def _instances(self, noun):
        from scipy.ndimage import binary_erosion
        segs = self._seg([noun])[noun]
        C = []
        for v, lst in zip(self._views, segs):
            for m in lst or []:
                mk = m["mask"]
                e = binary_erosion(mk, iterations=2)
                mk = (e if e.sum() >= 15 else mk) & v["valid"]
                pts = v["xyz"][mk]
                z = pts[:, 2] - self.tz
                pts = pts[(z > 0.002) & (z < WS_ZMAX) & (pts[:, 0] > WS_X[0]) & (pts[:, 0] < WS_X[1]) & (np.abs(pts[:, 1]) < WS_Y)]
                if len(pts) < 15:
                    continue
                pts = _largest_components(pts)
                if len(pts) < 15:
                    continue
                C.append({"view": v["name"], "score": m["score"], "pts": pts, "c": np.median(pts, 0), "npix": int(m["mask"].sum())})
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
            if (top < 0.35 and len(g["views"]) < 2) or top < 0.2:
                continue
            pts = np.concatenate([m["pts"] for m in g["m"]])
            pts[:, 2] -= self.tz
            lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
            if lo[2] < 0.008:
                lo[2] = 0.0
            ext = hi - lo
            if max(ext[0], ext[1]) > MAX_EXT_XY or ext[2] > MAX_EXT_Z:
                continue
            out.append({"min": lo, "max": hi, "center": (lo + hi) / 2, "pts": pts, "support": sup * (1 + 0.25 * (len(g["views"]) - 1)),
                        "views": sorted(g["views"]), "score": top})
        out.sort(key=lambda o: -o["support"])
        keep = []
        for o in out:
            if any(np.linalg.norm(o["center"][:2] - k["center"][:2]) < 0.03 and abs(o["center"][2] - k["center"][2]) < 0.03 for k in keep):
                continue
            keep.append(o)
        return keep

    @staticmethod
    def _lr_order(insts):
        """left-to-right as seen from the main camera is resolved by the harness; keep LIBERO's base-frame order."""
        return sorted(insts, key=lambda o: (round(float(o["center"][1]) / 0.04), -float(o["center"][2])))

    # ---------------------------------------------------------------- robot
    def move(self, p_table, quat=None):
        p = np.asarray(p_table, float)[:3] + [0, 0, self.tz]
        R = self.E._native_tcp()[1] if quat is None else AL._q2m(np.asarray(quat, float))
        self.E._servo(p, R)
        return float(np.linalg.norm(self.hand_pose()[:3] - np.asarray(p_table, float)[:3]))

    def run(self, items: dict) -> bool:
        if self.E._over():
            return False
        return super().run(items)

    # ---------------------------------------------------------------- logging only (never shown to the planner)
    def success(self):
        return bool(self.E.success_flag)
