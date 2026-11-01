"""Shared part of the bench ports: a RoboDawn-style DiscreteEnvBase for a single-arm robosuite/MuJoCo Panda.

The benchmark adapters (env_libero.py, env_guava.py) only implement the low-level hooks below; this class turns the
RoboDawn command vocabulary (move axis cm / rotate axis deg / point preset / home / gripper / wait) into targets for
the benchmark's own controller (LIBERO: robosuite OSC_POSE; Guava: Guava's pybullet IK + joint controller).

Frame shown to the model ("F"): the robot base frame translated so that the TABLE TOP is z = 0, in cm.
+x = forward (away from the robot base), +y = the robot's left, +z = up.
Tool orientation is described the RoboDawn way: approach = R[:,0] (fingers point along it), finger axis = R[:,1].
Single arm: it is called "right" in commands/state (RoboDawn's grammar needs left|right); prompts are edited to say so.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import paths as _paths  # noqa: E402
P = _paths.ROOT
RD = _paths.ROBODAWN
if str(RD) not in sys.path:
    sys.path.insert(0, str(RD))

from harness.core.commands import Command  # noqa: E402
from harness.core.env import DiscreteEnvBase, EpisodeInfo, View  # noqa: E402

ARM = "right"


# ----------------------------------------------------------------------------- rotation helpers
def axang(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def rot_angle_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rotvec(R):
    """Axis-angle vector of rotation matrix R."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        w, V = np.linalg.eigh((R + R.T) / 2)
        ax = V[:, np.argmax(w)]
        return ax * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return v * th


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_wxyz(R):
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def frame_from(approach, finger):
    a = np.asarray(approach, float)
    a = a / np.linalg.norm(a)
    f = np.asarray(finger, float)
    f = f - a * (a @ f)
    f = f / np.linalg.norm(f)
    return np.stack([a, f, np.cross(a, f)], axis=1)


# canonical tool orientations in F (approach = col 0, finger axis = col 1)
PRESETS = {
    "down": frame_from([0, 0, -1], [0, 1, 0]),
    "forward": frame_from([1, 0, 0], [0, 1, 0]),
    "down45": frame_from([1, 0, -1], [0, 1, 0]),
}


@dataclass
class ArmState:
    position_cm: list
    approach: list
    finger_axis: list
    rpy_deg: list
    gripper_real: float


def _rpy(R):
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    pitch = math.degrees(math.asin(max(-1, min(1, -R[2, 0]))))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return [round(roll, 1), round(pitch, 1), round(yaw, 1)]


# ----------------------------------------------------------------------------- the env
class PortEnv(DiscreteEnvBase):
    """Single-arm port. Subclasses implement the hooks marked HOOK."""

    bench = "?"
    overview_cam = "?"
    wrist_cam = "robot0_eye_in_hand"
    img_size = 512
    grid_x = (0, 80)          # cm in F, grid drawn on the table
    grid_y = (-40, 40)
    POS_TOL_CM, ANG_TOL_DEG = 1.5, 8.0     # RoboDawn's "reached" tolerance

    def __init__(self):
        self.command_log = []
        self.L = np.eye(3)               # native rotation -> tool frame:  Rt = Rn @ L
        self.home_tool = None            # (p_F_m, Rt)
        self._initial_objects = {}
        self.ticks = 0

    # ---------------------------------------------------------------- HOOKS
    def _native_tcp(self):
        """HOOK -> (p_base_m (3,), R_native_base (3,3)) of the tool centre point in the robot base frame."""
        raise NotImplementedError

    def _finger_points(self):
        """HOOK -> (left_finger_base_m, right_finger_base_m) (used once per episode to find approach / finger axis)."""
        raise NotImplementedError

    def _servo(self, p_base_m, R_native) -> int:
        """HOOK: drive the TCP to the pose with the benchmark's controller; returns control ticks used."""
        raise NotImplementedError

    def _set_gripper(self, value: float) -> int:
        """HOOK: 1 = open, 0 = close; returns ticks."""
        raise NotImplementedError

    def _wait(self) -> int:
        raise NotImplementedError

    def _opening(self) -> float:
        """HOOK: measured finger opening 0 (closed) .. 1 (fully open)."""
        raise NotImplementedError

    def _render(self, cam: str) -> np.ndarray:
        """HOOK: upright uint8 HxWx3."""
        raise NotImplementedError

    def _cam_pose(self, cam: str):
        """HOOK -> (cam_pos_base_m, cam_R_base (MuJoCo convention: -z forward, +y up), fovy_deg)."""
        raise NotImplementedError

    def table_z_base_m(self) -> float:
        raise NotImplementedError

    def object_positions_base(self) -> dict:
        """HOOK -> {name: p_base_m} privileged object positions (oracle mode only)."""
        return {}

    def check_success(self) -> bool:
        """HOOK: the benchmark's own success predicate (logged, hidden from the model)."""
        raise NotImplementedError

    # ---------------------------------------------------------------- frames
    def to_F(self, p_base_m):
        return np.asarray(p_base_m, float) - np.array([0, 0, self._tz])

    def from_F(self, p_F_m):
        return np.asarray(p_F_m, float) + np.array([0, 0, self._tz])

    def _calibrate_tool(self):
        """Find approach / finger axis in the native TCP frame from the finger geometry (once per episode)."""
        self._tz = float(self.table_z_base_m())
        p, Rn = self._native_tcp()
        lf_, rf_ = self._finger_points()
        mid = (np.asarray(lf_) + np.asarray(rf_)) / 2
        approach = p - mid
        finger = np.asarray(lf_) - np.asarray(rf_)
        Rt = frame_from(approach, finger)
        self.L = Rn.T @ Rt
        self.home_tool = (self.to_F(p), Rt)

    def tool_pose(self):
        p, Rn = self._native_tcp()
        return self.to_F(p), Rn @ self.L

    # ---------------------------------------------------------------- state
    def arm_state(self) -> ArmState:
        p, Rt = self.tool_pose()
        return ArmState(position_cm=[round(float(v) * 100, 1) for v in p],
                        approach=[round(float(v), 2) for v in Rt[:, 0]],
                        finger_axis=[round(float(v), 2) for v in Rt[:, 1]],
                        rpy_deg=_rpy(Rt), gripper_real=round(float(self._opening()), 3))

    def object_states(self) -> dict:
        out = {}
        for name, pb in self.object_positions_base().items():
            out[name] = {"position_cm": [round(float(v) * 100, 1) for v in self.to_F(pb)], "model": name}
        return out

    @property
    def table_z_cm(self) -> float:
        return 0.0

    def prompt_context(self) -> dict:
        return {"table_z": 0}

    @property
    def success(self) -> bool:     # hidden from the model: never reported (commands are never refused)
        return False

    def state(self) -> dict:
        return {"steps_used": self.steps_used, "step_limit": self.step_limit, "success": False,
                "table_z_cm": 0.0, ARM: asdict(self.arm_state()), "objects": self.object_states()}

    def observe(self) -> dict:
        images = {"overview": self._render(self.overview_cam)}
        try:
            images["wrist"] = self._render(self.wrist_cam)
        except Exception:  # noqa: BLE001
            pass
        return {"images": images, "state": self.state()}

    # ---------------------------------------------------------------- projection / annotation
    def project(self, cam, pts_F_m):
        cp, cR, fovy = self._cam_pose(cam)
        H = W = self.img_size
        f = (H / 2) / math.tan(math.radians(fovy) / 2)
        out = []
        for p in np.atleast_2d(pts_F_m):
            pc = cR.T @ (self.from_F(p) - cp)
            if pc[2] > -1e-3:
                out.append(None)
                continue
            out.append((W / 2 + f * pc[0] / -pc[2], H / 2 - f * pc[1] / -pc[2]))
        return out

    def _annotate(self, img, cam):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img).convert("RGB")
        d = ImageDraw.Draw(im)
        gx0, gx1 = self.grid_x
        gy0, gy1 = self.grid_y
        grey = (150, 150, 150)
        for x in range(gx0, gx1 + 1, 10):
            a, b = self.project(cam, [[x / 100, gy0 / 100, 0], [x / 100, gy1 / 100, 0]])
            if a and b:
                d.line([a, b], fill=grey, width=1)
                d.text((b[0] + 2, b[1] - 6), f"x{x}", fill=(255, 255, 255))
        for y in range(gy0, gy1 + 1, 10):
            a, b = self.project(cam, [[gx0 / 100, y / 100, 0], [gx1 / 100, y / 100, 0]])
            if a and b:
                d.line([a, b], fill=grey, width=1)
                d.text((b[0] + 2, b[1] - 6), f"y{y}", fill=(255, 255, 255))
        p, Rt = self.tool_pose()
        tip, tail = self.project(cam, [p, p - Rt[:, 0] * 0.08])
        if tip:
            r = 7
            d.ellipse([tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r], outline=(255, 140, 0), width=3)
            d.text((tip[0] + 9, tip[1] - 9), "R", fill=(255, 140, 0))
            if tail:
                d.line([tail, tip], fill=(255, 140, 0), width=2)
        o = [gx0 / 100, gy1 / 100, 0]
        pts = self.project(cam, [o, [o[0] + .1, o[1], 0], [o[0], o[1] + .1, 0], [o[0], o[1], .1]])
        if all(pts):
            for q, col, lab in zip(pts[1:], [(255, 0, 0), (0, 200, 0), (0, 0, 255)], "xyz"):
                d.line([pts[0], q], fill=col, width=3)
                d.text((q[0] + 2, q[1] - 2), "+" + lab, fill=col)
        return im

    def views(self, obs: dict, cameras: list) -> list:
        from PIL import Image
        out = [View("overview", self._annotate(obs["images"]["overview"], self.overview_cam), self.overview_caption)]
        if "wrist" in obs["images"]:
            out.append(View("wrist", Image.fromarray(obs["images"]["wrist"]),
                            "wrist (camera on the gripper, looking along the fingers)"))
        return out

    overview_caption = "overview"

    # ---------------------------------------------------------------- actions
    def command_target(self, cmd: Command):
        p, Rt = self.tool_pose()
        if cmd.kind == "move":
            p = p.copy()
            p[{"x": 0, "y": 1, "z": 2}[cmd.axis]] += cmd.value / 100.0
        elif cmd.kind == "rotate":
            ax = {"roll": [1, 0, 0], "pitch": [0, 1, 0], "yaw": [0, 0, 1]}[cmd.axis]
            Rt = axang(ax, cmd.value) @ Rt
        elif cmd.kind == "point":
            # a parallel gripper is symmetric about its approach axis: pick the finger-axis sign that needs less turning
            cands = [PRESETS[cmd.axis], PRESETS[cmd.axis] @ np.diag([1, -1, -1])]
            Rt = min(cands, key=lambda R: rot_angle_deg(R, Rt))
        elif cmd.kind == "home":
            p, Rt = self.home_tool
        return p, Rt

    def execute(self, cmd: Command) -> dict:
        t0 = time.time()
        res = {"command": cmd.text(), "kind": cmd.kind, "arm": cmd.arm, "ok": True, "note": "",
               "steps_used_before": self.steps_used}
        if cmd.arm not in (None, ARM):
            snap = {ARM: asdict(self.arm_state())}
            res.update(ok=False, note="this robot has ONE arm, called 'right'; nothing moved", before=snap, after=snap)
            self.command_log.append(res)
            return res
        if cmd.kind == "done" or self.budget_exhausted:
            snap = {ARM: asdict(self.arm_state())}
            res.update(ok=cmd.kind == "done", note="agent declared done" if cmd.kind == "done" else "step budget exhausted",
                       before=snap, after=snap)
            self.command_log.append(res)
            return res
        before = self.arm_state()
        tgt = None
        try:
            if cmd.kind == "gripper":
                n = self._set_gripper(float(cmd.value))
            elif cmd.kind == "wait":
                n = self._wait()
            else:
                tgt = self.command_target(cmd)
                n = self._servo(self.from_F(tgt[0]), tgt[1] @ self.L.T)
            self.ticks += int(n or 0)
            res["ticks"] = int(n or 0)
        except Exception as exc:  # noqa: BLE001
            res.update(ok=False, note=f"simulator error: {type(exc).__name__}: {exc}")
        after = self.arm_state()
        if tgt is not None and res["ok"]:
            p, Rt = self.tool_pose()
            perr = float(np.linalg.norm(p - tgt[0]) * 100)
            aerr = rot_angle_deg(Rt, tgt[1])
            res["position_error_cm"], res["orientation_error_deg"] = round(perr, 2), round(aerr, 1)
            if perr > self.POS_TOL_CM or aerr > self.ANG_TOL_DEG:
                res["ok"] = False
                moved = np.asarray(after.position_cm) - np.asarray(before.position_cm)
                mcm = float(np.linalg.norm(moved))
                if cmd.kind in ("rotate", "point"):
                    what = f"the gripper is still {aerr:.0f} deg from the requested orientation"
                elif mcm < 0.5:
                    what = "the arm did NOT move"
                else:
                    what = f"the fingertips moved ({moved[0]:+.1f}, {moved[1]:+.1f}, {moved[2]:+.1f}) cm, i.e. only part of the way"
                res["note"] = (f"controller could not reach the target: {what}; remaining error {perr:.1f} cm / {aerr:.0f} deg "
                               "(see CURRENT STATE for the actual pose). Try a smaller step, a different direction, or move "
                               "away from the table / obstacles first.")
        if cmd.kind == "gripper" and res["ok"]:
            real = after.gripper_real
            if cmd.value < 0.5 and real > 0.08:
                res["note"] = f"fingers stopped at opening {real:.2f}: something is between them (probably grasped)"
            elif cmd.value < 0.5:
                res["note"] = "fingers closed fully: nothing between them"
            else:
                res["note"] = "gripper opened"
        if cmd.clipped:
            res["note"] = (res["note"] + " " if res["note"] else "") + "(magnitude was clipped to the limit)"
        res.update(steps_used_after=self.steps_used, before={ARM: asdict(before)}, after={ARM: asdict(after)},
                   seconds=round(time.time() - t0, 2))
        self.command_log.append(res)
        return res

    def should_stop(self) -> Optional[str]:
        for name, o in (self.object_states() or {}).items():
            p0 = self._initial_objects.get(name)
            if p0 and p0[2] > -5 and o["position_cm"][2] < -30:
                return "object_fell"
        return None


# ----------------------------------------------------------------------------- MuJoCo helpers
def mj_table_z(sim, base_pos, base_R, xs=(0.35, 0.75), ys=(-0.3, 0.3), n=9):
    """Table-top height (base frame) by casting rays straight down over the area in front of the robot."""
    import mujoco
    m, d = sim.model._model, sim.data._data
    zs = []
    gid = np.array([-1], dtype=np.int32)
    for x in np.linspace(*xs, n):
        for y in np.linspace(*ys, n):
            pw = base_pos + base_R @ np.array([x, y, 1.5])
            dist = mujoco.mj_ray(m, d, pw, np.array([0, 0, -1.0]), None, 1, -1, gid)
            if dist > 0:
                zs.append(pw[2] - dist - base_pos[2])
    zs = np.round(np.asarray(zs), 2)
    vals, cnt = np.unique(zs, return_counts=True)
    return float(vals[np.argmax(cnt)])


def find_body(sim, *subs):
    names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
    for s in subs:
        for i, nm in enumerate(names):
            if nm and nm.endswith(s):
                return i
    raise KeyError(subs)
