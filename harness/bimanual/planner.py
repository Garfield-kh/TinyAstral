#!/usr/bin/env python
"""H* on RoboTwin 2.0 (two arms, aloha): the final harness used for the paper's teacher and students.

The planner (any OpenAI-compatible VLM) only decides WHAT: which object / part, which arm, which reference and relation,
order and arm coordination, and when the task is complete. The tools solve HOW from the cameras:
  grasp(arm, object, part)       camera point cloud -> shape analysis (major axis, width profile, thin protrusions, rim)
                                 -> fingertip point + closing axis -> approach candidates checked with the arm's motion
                                 planner (IK + path) -> approach / close / lift test; automatic retry on a failed hold
  place(arm, object, ref, rel)   support surface / container floor / table point from the reference's point cloud;
                                 closed-loop xy correction from the camera before lowering; release and retreat
  handover(object, to_arm)       regrasp through the table between the arms
  bimanual([...])                two calls executed together (joint grasp + joint lift test for a shared object)
  press / push / lift / move / release / home / locate / verify / done / give_up
Perception: SAM3 + native head / wrist RGB-D only (perception.py). No task names, no asset contact / functional points,
no simulator object poses are used by any tool. Simulator success is never shown to the planner. Privileged adapter
queries (grasp/place plans, object poses, boxes, joints, ...) are sealed (_seal_privileged): calling them raises.

Code layout. The harness is a stack of layers: planner_base.py holds the base tool bodies; this module defines
the planner-facing SYSTEM prompt, the planner loop (main) and a series of `class Harness(<Parent>)` refinements, each
gated to the calls it changes (in-hand tracking and honest re-measurement, place/carry closed loop, transport fallbacks,
instance-locked grasps and honest completion, end polarity of long objects, release without sweeping, slender-object
grasps and push, held-object height, panel presses, drawers, lean placement, pouring over a container). A per-task note
(memory/robotwin/<task>.json) is shown at the first turn (HARNESS_MEMORY=0 disables it).

Usage: harness/run.py robotwin --task T --episode k --lane L --url U [--model qwen]
(episode k -> the k-th valid RoboTwin seed >= 100000 of the task, listed in harness/data/robotwin_seedcache)."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # harness/: paths, tools, geom
import paths as _paths  # noqa: E402
_paths.use("bimanual")
import scene_runner as R  # noqa: E402
import geom as Gm  # noqa: E402
import geometry as G  # noqa: E402
import perception as PC  # noqa: E402
from adapter_robotwin import RoboTwinAdapter  # noqa: E402

from request_trace import record_request  # noqa: E402

M = R.M
P = _paths.OUT
SEEDS = _paths.SEEDCACHE
import os as _os
CAP, WALL, MAX_REJECT, KEEP_IMG_TURNS = int(_os.environ.get("HARNESS_CAP", 30)), int(_os.environ.get("HARNESS_WALL", 900)), 2, 2   # CAP/WALL parameters
MEMDIR = _paths.MEMORY     # note sets
MEMSET = "robotwin"                                    # memory
GAP_MIN = 0.02
LIFT_TEST = 0.05
TOP_Q = np.array([0.5, -0.5, 0.5, 0.5])
ARMS = ("left", "right")
PRE = 0.08

SYSTEM = """You control a two-arm robot (left arm, right arm; parallel grippers) above a table. You decide WHAT to do: which object (and which part of it) each arm grasps, where it goes (reference object + relation), the order of steps, how the two arms cooperate, and when the task is complete. The tools compute HOW from the cameras: grasp points and gripper orientations from the object's 3-D shape, collision-free reachable approaches, placement heights from support surfaces, and small closed-loop corrections. Never compute coordinates yourself unless a tool asks for them.

Each turn you get the head camera image, both wrist camera images, the arm states and the result of your previous call. Reply with a very short reasoning in <think>...</think> followed by exactly one call:
<tool_call>{"name": "...", "arguments": {...}}</tool_call>

Objects: use the scene object names. A target that is not in the list (e.g. a coloured mat or marked area) can be named with a short visual description such as "blue mat"; it is found by camera segmentation. Directions: "left"/"right" = the robot's left/right (image left/right in the head camera), "front" = toward the robot, "back" = away. Positions are cm in the table frame, z = height above the table. Each arm reaches its own side and the middle: use the arm on the object's side (x < 0 -> left, x > 0 -> right).

Tools:
- locate(objects: [str], descriptions=null) -> centre, size [x, y, z], top z and visibility of each object. If a scene object is found at an implausible place or size (compare with the images), pass descriptions={name: "short visual phrase, e.g. brown wicker basket"}; the name keeps working in all tools and the phrase is used for detection from then on.
- grasp(arm, object, part=null) -> the tool analyses the object's shape, chooses the grasp point and gripper orientation, plans a reachable approach, closes and does a 5 cm lift test (automatic retries with other grasps). code OK = held. part (optional): "handle" (a thin protruding handle), "left handle" / "right handle" (the handle on that side of the object), "left end" / "right end" (ends of a long object), "rim" (edge of a wide bowl/container). Without part the tool grasps narrow objects across their narrow side and wide objects at a handle or rim automatically.
- place(arm, object, ref, relation="on", xy=null, align=false, turn_deg=null) -> carry the HELD object to the reference and release it. align=true turns the object so its long side is parallel to the long side of ref (use it when the object must line up with a pad, mat, tray or slot). turn_deg turns the held object about the vertical axis by that many degrees before putting it down (positive = counter-clockwise seen from above); judge the needed turn from the images. With a reference object, xy=[x, y] (cm, table frame) picks the point on / in ref (e.g. one half of a tray); without xy the tool picks the centre or a free part. relation: "on" (on top of ref, e.g. stacking or onto a pad/mat/coaster), "in" (inside a container: basket, pan, bowl, bin), "left_of" / "right_of" / "front_of" / "behind" (on the table beside ref on that side, a clear gap of about 5 cm, same row), or ref="table" with xy=[x, y] cm. The tool centres the object over ref with camera feedback, lowers it and opens the gripper. relation="against" stands the held object up and leans it against a tall support (a stand or rack); this is chosen automatically when ref is tall and its top is too small to hold the object. After releasing, the tool measures where the object ended up: if it is off the target it re-grasps and places once more, and if it is still off it returns code OFF_TARGET with the measured offset (dx_cm, dy_cm, dyaw_deg) instead of OK.
- carry(arm, object, ref, relation="on", height_cm=10, turn_deg=null) -> like place but keeps holding the object height_cm above the target.
  place and carry also take face="front" / "back" / "left" / "right": the tool turns the held object so that its flat face (its thinnest side, e.g. a sign, a board, a fan) faces that direction ("front" = toward the robot). If the images show the wrong side would face that way, also pass turn_deg=180. Better: also pass face_now="front" / "back" / "left" / "right" = the direction that flat face (the side that must end up facing) points to now, as seen in the images; the tool then turns that side to face (up to a full half turn) instead of guessing which of the two flat sides is meant. Whenever an object must end up facing a direction (facing the robot, facing left ...), use face instead of estimating turn_deg yourself; a task note that says "turn" or "rotate" to face something means face.
  place and carry also take held_part="left end" / "right end" / "front end" / "back end": that end of the held object (as it points now, see the images) is brought over the target instead of its middle (e.g. the head of a held tool onto a target).
- pour(arm, direction="left", degrees=100, shake=true, return_upright=false) -> tip the HELD object (a container) about a horizontal axis so its top tilts toward direction (left/right/front/back), optionally shake it a little, and keep holding it (tilted unless return_upright=true). Carry it over the destination first.
- handover(object, to_arm) -> passes the object held by the other arm to to_arm (puts it down between the arms, the other arm backs off, to_arm grasps it). Result OK = to_arm holds it.
- lift(arm, height_cm) -> raise that gripper (and what it holds) to this height above the table (if it is already that high it stays; use move dz to go further up). arm may be "both" to lift both grippers together (for an object held with two hands).
- articulate(arm="auto", object, motion="open", degrees=80) -> opens (or closes) a hinged front door (microwave, cabinet): the tool finds the handle and hinge from the cameras, picks the arm on the hinge side when arm="auto" (a named arm is always used as given), grasps the handle and swings it along the door arc, then releases. For a low flat object with a lid on a horizontal hinge it pinches the lid edge, lifts the lid about the hinge and keeps holding it. If the swing is SHORT, look at the images and repeat with arm="auto" or the arm named in the hint.
- press(arm, object, direction=null) -> press the object with the closed gripper and come back (the gripper opens again afterwards). direction "down" presses its top (buttons, staplers); "from_front" / "from_left" / "from_right" / "from_back" pushes horizontally on its face toward that side (a switch or button facing sideways). Without direction a top press that cannot be reached falls back to tilted and horizontal pushes.
- push(arm, object, direction, distance_cm) -> push the object along a direction with the closed gripper.
- move(arm, dx, dy, dz) -> move that gripper by an offset in cm (anything held moves with it).
- release(arm) -> open that gripper and back off upward.
- home(arm) -> return that arm to its raised observation pose (do this to get an arm out of the way).
- bimanual(calls: [call, call]) -> run two calls at the same time, one per arm (each {"name", "arguments"} of grasp/place/lift/move/release/home). Use it to grasp one large object with both hands (e.g. two grasps with part "handle" or "left end"/"right end"), then lift(arm="both").
- verify(checks) / done(checks) -> checks are [[predicate, args...], ...] evaluated from the cameras and gripper sensors. Predicates: on(a, b), in(a, b), on_table(a), left_of/right_of/front_of/behind(a, b), near(a, b), same_row(a, b), at(a, x, y, radius_cm), upright(a), height(a, cm) (bottom of a above the table), lifted(a, cm) (held and bottom at least cm high), holding(arm, a), released(arm), pressed(a). done ends the episode if all checks pass. When no predicate measures the goal, judge the images and call done(checks=[]). A check with ok=null could not be measured reliably (not seen, or an implausible measurement): judge that part from the images and leave it out of done if the images show it is fine.
- give_up(reason).

Strategies: to put something into a container that must end up lifted (pan, basket, tray), you may hold the container by its handle with the arm on its side, carry it to the middle a few cm above the table (carry with ref "table", xy=[0, -6], height_cm=6) and then place the object into it with the other arm. Each arm reaches its own side and the middle of the table. If place/carry/grasp says UNREACHABLE, bring the reference object to the middle first (grasp it with the arm on its side and place it on ref "table" with xy=[0, -8]), use the other arm, or pass the held object with handover. To stack several objects, first bring the bottom one to the middle, then put the others on it one by one, moving each used arm home afterwards so it does not block the view. To lift a container with something in it, put the object in first, move that arm home, then grasp the container with the arm on its side (part "handle" if it has a handle) and lift it.

If a task note is given with the task, it was learned from earlier practice on this kind of task: it says what must be true at the end (positions, heights, orientation, holding or released) even when the task sentence does not say it. Follow it unless the images clearly contradict it, and make your done checks cover it.

Rules: one call per turn. A grasp holds only if its code is OK. If a step fails, read the result and change something (other arm, other part, re-locate, move the other arm home) instead of repeating it unchanged. Release the gripper(s) when the task says to put/place/stack something. For lift tasks keep holding and lift at least 10 cm before done. Call done as soon as the goal is visibly achieved."""


def _implausible(o, size_cm):
    """a measurement that does not look like the same object resting somewhere: bottom well above the table, or a footprint
    much larger / smaller than the object's size when it was held."""
    if o is None:
        return False
    if float(o["lo"][2]) > 3.0:
        return True
    s0 = np.asarray(size_cm if size_cm is not None else [0, 0, 0], float)
    a, b = float(np.max(np.asarray(o["size"], float)[:2])), float(np.max(s0[:2]))
    return b > 1.0 and not (0.55 <= a / b <= 1.8)


class _Unsure(Exception):
    """a predicate's measurement is implausible (reported as ok=null)."""


def r1(v):
    return [round(float(x), 1) for x in v]


def footprint(pts):
    """long / short side (cm) and long axis of the minimum-area rectangle around the points' xy (not the axis-aligned box
    which makes a diagonal long object look square)."""
    xy = np.asarray(pts, float)[:, :2]
    if len(xy) < 5:
        return 0.0, 0.0, np.array([1.0, 0.0])
    c, u, m = G.pca2(xy)
    return float(G.extent((xy - c) @ u)) * 100, float(G.extent((xy - c) @ m)) * 100, np.asarray(u, float)


class Harness:
    def __init__(self, sim, names):
        self.sim, self.A = sim, RoboTwinAdapter(sim)
        self.names = list(names)
        self.queries = []
        self.descriptions = {}
        self.perc, self.first, self.pressed = {}, {}, set()
        self.holding = {}
        self.hold = {}
        self._place_arm = None
        self._drop = True          # drop points on the robot body (not while something is in a gripper)
        self.tz = table_z_from_depth(sim)           # camera, not the randomised simulator table height
        _seal_privileged(self.A)
        self.ever = False
        self.trace = []
        _run = self.A.run

        def run(items):
            ok = _run(items)
            self.ever = self.ever or self.A.success()
            return ok
        self.A.run = run

    # ------------------------------------------------------------ observation pose
    def observe_pose(self, arm):
        """work-ready pose: gripper raised over its side of the table, wrist camera pitched 50 deg down so the wrist
        cameras see the table area in front of the robot that the head camera misses."""
        import os
        sx = -1 if arm == "left" else 1
        b = np.radians(float(os.environ.get("V2_OBS_PITCH", 50)))
        a = np.array([0.0, np.cos(b), -np.sin(b)])
        return self.A.hand_from_tcp(np.array([sx * 0.24, float(os.environ.get("V2_OBS_Y", -0.25)), self.tz + 0.25]), G.hand_quat(a, [-1.0, 0, 0]))

    def start(self):
        free = [a for a in ARMS if not self.holding.get(a)]
        if free:
            self.run_seq({a: [{"type": "move", "pose": self.observe_pose(a).tolist()}] for a in free})

    # ------------------------------------------------------------ perception
    def perceive(self, labels=None):
        for lb in ([labels] if isinstance(labels, str) else (labels or [])):
            if lb not in self.names and lb not in self.queries:
                self.queries.append(lb)
        q = list(dict.fromkeys([*self.names, *self.queries]))
        try:
            new = PC.perceive(self.sim.rd, q, self.tz, priors=self.first, pos_prior=self.pos_prior(), prompt_overrides=dict(self.descriptions),
                              drop_robot=self._drop and not self.holding)
        except Exception as e:  # noqa: BLE001
            self.trace.append({"perceive_error": repr(e)[:200]})
            new = {}
        for n, o in new.items():
            self.perc[n] = o
            self.first.setdefault(n, o)
        return new

    def pos_prior(self):
        pp = {n: self.world(o["center"])[:2] for n, o in self.perc.items()}
        for a, o in self.holding.items():
            est = self.held_est(a)
            if est is not None:
                pp[o] = est[0][:2]
        return pp

    def held_est(self, arm):
        """object centre / bottom (world) predicted from the gripper pose and the in-hand offset measured at grasp time."""
        h = self.hold.get(arm)
        if not h:
            return None
        hp = self.A.hand_pose(arm)
        R = Gm.mat(hp)[:3, :3]
        t = self.A.tcp_point(hp)
        return t + R @ h["c"], float((t + R @ h["b"])[2])

    def see(self, n):
        if not getattr(self, "_raw", False):              # a held object follows its gripper (no re-segmentation)
            for a, ob in self.holding.items():
                if ob == n and a in self.hold and "pts_h" in self.hold[a]:
                    return self._tracked(a, n)
        return self.perceive([n]).get(n)

    def _tracked(self, arm, n):
        hp = self.A.hand_pose(arm)
        R = Gm.mat(hp)[:3, :3]
        t = self.A.tcp_point(hp)
        pts = (R @ self.hold[arm]["pts_h"].T).T + t
        lo = np.percentile(pts, 2, axis=0)
        hi = np.percentile(pts, 98, axis=0)
        lo_c = (lo - [0, 0, self.tz]) * 100
        hi_c = (hi - [0, 0, self.tz]) * 100
        o = {"center": (lo_c + hi_c) / 2, "size": hi_c - lo_c, "lo": lo_c, "hi": hi_c, "pts": pts, "tracked": True}
        self.perc[n] = o
        return o

    def world(self, p_cm):
        p = np.asarray(p_cm, float) / 100
        return np.array([p[0], p[1], p[2] + self.tz])

    def cm(self, p):
        p = np.asarray(p, float)
        return r1([p[0] * 100, p[1] * 100, (p[2] - self.tz) * 100])

    def describe(self, objs):
        new = self.perceive(objs)
        out = {}
        for n in objs:
            o = new.get(n)
            out[n] = ({"visible": True, "center": r1(o["center"]), "size": r1(o["size"]), "top_z": round(float(o["hi"][2]), 1)}
                      if o is not None else {"visible": False})
        return out

    def tcp(self, arm):
        return self.A.tcp_point(self.A.hand_pose(arm))

    def arm_text(self):
        parts = []
        for a in ARMS:
            t = self.tcp(a)
            parts.append(f"{a}: gripper at {self.cm(t)} cm, opening {self.A.opening(a) * 100:.0f}%, holding {self.holding.get(a) or 'nothing'}")
        return "Arms: " + "; ".join(parts) + "."

    # ------------------------------------------------------------ motion helpers
    def planner(self, arm):
        r = self.A.T.robot
        return r.left_plan_path if arm == "left" else r.right_plan_path

    def feasible(self, arm, poses):
        """plan the chain of hand poses from the current joint state; True if every segment plans."""
        last = None
        for p in poses:
            try:
                res = self.planner(arm)(list(map(float, p)), last_qpos=last) if last is not None else self.planner(arm)(list(map(float, p)))
            except Exception:  # noqa: BLE001
                return False
            if not isinstance(res, dict) or res.get("status") != "Success":
                return False
            pos = res.get("position")
            if pos is not None and len(pos):
                rb = self.A.T.robot
                pl = rb.left_planner if arm == "left" else rb.right_planner
                cur = (rb.left_entity if arm == "left" else rb.right_entity).get_qpos()
                full = np.asarray(cur if last is None else last, float).copy()
                try:
                    idx = [pl.all_joints.index(n) for n in pl.active_joints_name if n in pl.all_joints]
                    full[idx] = np.asarray(pos[-1], float)[: len(idx)]
                    last = full
                except Exception:  # noqa: BLE001
                    last = None
                    break
        return True

    def run_seq(self, seqs):
        """seqs: {arm: [action dicts]} executed index-paired. Returns {arm: code}."""
        codes = {a: "OK" for a in seqs}
        n = max(len(v) for v in seqs.values())
        for k in range(n):
            items = {a: [v[k]] for a, v in seqs.items() if k < len(v) and codes[a] == "OK"}
            if not items:
                break
            ok = self.A.run(items)
            if not ok:
                ok = self.A.run(items)
            for a, L in items.items():
                if L[0]["type"] == "move":
                    pe = Gm.pos_err(self.A.hand_pose(a), L[0]["pose"])
                    if not ok or pe > 0.03:
                        codes[a] = "PLAN_FAIL" if not ok else "POSE_ERR"
        return codes

    def retreat(self, arm, total=0.10):
        """back straight up in short steps (a long joint-space move can sweep the open fingers through what was just released)."""
        for dz in (0.015, 0.025, total - 0.04):
            self.run_seq({arm: [self.move_tcp(arm, self.tcp(arm) + [0, 0, dz])]})

    SOFT_GRIP = 0.15            # after the fingers stop on the object, hold this far (normalised) below the contact opening

    def soften(self, arms):
        """a full-close target squeezes light objects deep into the fingers, and the stored penetration throws them off
        the table when the gripper later opens or the arm moves. After contact, re-target the
        fingers just below the contact opening."""
        items = {}
        for a in arms:
            o = float(self.A.opening(a))
            if o >= GAP_MIN:
                items[a] = [{"type": "grip", "val": min(max(0.0, o - self.SOFT_GRIP), 0.18)}]   # command stays below 0.2 (= "closed")
        if items:
            self.run_seq(items)

    def open_gently(self, arm):
        """open in steps (contact opening, a bit wider, fully) instead of one jump to fully open."""
        o = float(self.A.opening(arm))
        for v in (min(1.0, o + 0.05), min(1.0, o + 0.25), 1.0):
            self.run_seq({arm: [{"type": "grip", "val": v}]})

    def move_tcp(self, arm, tcp, q=None):
        q = self.A.hand_pose(arm)[3:] if q is None else q
        return {"type": "move", "pose": self.A.hand_from_tcp(tcp, q).tolist()}

    # ------------------------------------------------------------ grasp
    def grasp_plan(self, arm, obj, part=None, exclude=(), own_side=False):
        o = self.see(obj)
        if o is None:
            return None, {"code": "NOT_SEEN", "why": f"the cameras cannot find {obj}; try locate with a visual description"}
        side_word = None
        pl = str(part or "").lower().replace("_", " ").replace("-", " ").strip()
        if pl in ("left handle", "right handle"):
            side_word, part = pl.split()[0], "handle"
        specs = G.grasp_specs(o["pts"], self.tz, arm, part)
        if part and "handle" in str(part).lower() and specs and specs[0]["why"].startswith("no thin protrusion"):
            hs = self.part_specs(obj, o, arm, "handle")
            if hs:
                specs = hs + specs
        if side_word:                    # the handle on the requested side of the object (world x), by geometry
            cx = float(G.body_center(o["pts"])[0])
            sg = -1.0 if side_word == "left" else 1.0
            dh = self._side_handle_spec(obj, o, side_word)
            specs = ([dh] if dh else []) + self.part_specs(obj, o, arm, "handle") + specs   # thin handles are often missing from the body mask
            hs = [sp for sp in specs if ("handle" in sp["why"] or "protrusion" in sp["why"]) and (sp["p"][0] - cx) * sg > 0.02]
            hs.sort(key=lambda sp: -sg * sp["p"][0])
            specs = hs or [dict(sp, why=f"no {side_word} handle found; {side_word} end of the object") for sp in G.grasp_specs(o["pts"], self.tz, arm, f"{side_word} end")]
        if own_side:                     # two hands on one object: each hand takes a grasp on its own half
            cx = self.world(o["center"])[0]
            sd = -1 if arm == "left" else 1
            specs = [s for s in specs if (s["p"][0] - cx) * sd > 0.01]
        tried = 0
        for si, s in enumerate(specs):
            sz = np.asarray(o["size"], float)
            tall = sz[2] >= 10.0 and sz[2] >= 2.0 * max(sz[0], sz[1])
            if tall and s.get("out") is None:            # tall thin object: side grasp on the upper part (tilted/top grasps topple it)
                s = dict(s)
                s["p"] = np.r_[s["p"][:2], self.world(o["lo"])[2] + 0.72 * (self.world(o["hi"])[2] - self.world(o["lo"])[2])]
                cands = self._sl_tall_cands(s, arm)          # hook (parent: G.candidates(s, arm, tilts=(90, 75, 60)))
            else:
                cands = G.candidates(s, arm)
            for al, a, c in cands:
                if (si, al) in exclude:
                    continue
                q = G.hand_quat(a, c)
                p = s["p"]
                pre = p - a * PRE
                hp, hg = self.A.hand_from_tcp(pre, q), self.A.hand_from_tcp(p, q)
                tried += 1
                if self.feasible(arm, [hp]) and self.feasible(arm, [hg]):
                    return {"arm": arm, "obj": obj, "pre": hp, "grasp": hg, "q": q, "key": (si, al), "why": s["why"], "obs": o, "spec_p": p,
                            "pregrip": float(s.get("pregrip", 1.0)),
                            "tilt": al, "point_cm": self.cm(p), "shape": s["shape"]}, None
                if tried >= 24:
                    break
        g = self._gm_grasp_more(arm, obj, o, specs, exclude)   # only after every candidate above failed to plan
        if g is not None:
            return g, None
        return None, {"code": "UNREACHABLE", "why": f"no reachable grasp of {obj} for the {arm} arm ({tried} approaches planned); "
                                                    "try the other arm, move the object closer, or another part"}

    def _side_handle_spec(self, obj, o, side):
        """a side handle from raw depth points just outside the object's body on the requested side (thin handles
        are usually missing from the segmentation mask). Points of other detected objects are excluded."""
        try:
            views = PC.capture(self.sim.rd)
        except Exception as e:  # noqa: BLE001
            self.trace.append({"side_handle_error": repr(e)[:120]})
            return None
        body = np.asarray(o["pts"], float)
        c = np.asarray(G.body_center(body), float)
        sg = -1.0 if side == "left" else 1.0
        edge = float(np.percentile(body[:, 0], 98 if sg > 0 else 2))
        top = float(np.percentile(body[:, 2], 97))
        P_ = np.concatenate([v["xyz"][v["valid"]] for v in views])
        sel = ((P_[:, 0] - edge) * sg > -0.005) & ((P_[:, 0] - edge) * sg < 0.07) & (np.abs(P_[:, 1] - c[1]) < 0.05) \
            & (P_[:, 2] > self.tz + 0.015) & (P_[:, 2] < top + 0.015)
        reg = P_[sel]
        for n, ob in self.perc.items():                 # drop points of other objects
            if n == obj or ob is None or "pts" not in ob or len(reg) == 0:
                continue
            lo, hi = np.asarray(ob["pts"]).min(0) - 0.01, np.asarray(ob["pts"]).max(0) + 0.01
            reg = reg[~np.all((reg >= lo) & (reg <= hi), axis=1)]
        rob = [self.tcp(a) for a in ARMS]
        reg = reg[np.min([np.linalg.norm(reg - r_, axis=1) for r_ in rob], axis=0) > 0.06] if len(reg) else reg
        if len(reg) < 15:
            self.trace.append({"side_handle": side, "points": int(len(reg))})
            return None
        xo = float(np.percentile(reg[:, 0], 98 if sg > 0 else 2))       # outer bar of the handle (a loop has a hole inside)
        bar = reg[np.abs(reg[:, 0] - xo) < 0.012]
        bar = bar if len(bar) >= 8 else reg
        ztop = float(np.percentile(bar[:, 2], 90))
        p = np.r_[xo - sg * 0.004, float(np.median(bar[:, 1])), max(ztop - 0.006, self.tz + 0.01)]
        sp = {"p": p, "m": np.array([1.0, 0.0]), "why": f"{side} handle (outer bar) from depth points beside the body", "top": ztop,
              "out": np.array([sg, 0.0]), "pregrip": 0.45}   # half-open: the inner finger must fit inside the handle loop
        sp["shape"] = {"handle_points": int(len(reg))}
        self.trace.append({"side_handle": side, "points": int(len(reg)), "p_cm": self.cm(sp["p"])})
        return sp

    def part_specs(self, obj, o, arm, word):
        """text-prompted part segmentation: '<object> handle' clusters near the object (thin parts are often missing
        from the whole-object mask)."""
        base = re.sub(r"_\d+$", "", obj)
        names = [f"{base}_{word}_1", f"{base}_{word}_2"]
        try:
            res = PC.perceive(self.sim.rd, names, self.tz)
        except Exception as e:  # noqa: BLE001
            self.trace.append({"part_error": repr(e)[:200]})
            return []
        c = self.world(o["center"])
        bc = G.body_center(o["pts"])
        out = []
        for n, r in res.items():
            pc = self.world(r["center"])
            if np.linalg.norm(pc[:2] - c[:2]) > 0.18 or len(r["pts"]) < 10:
                continue
            reg = r["pts"]
            _, u, m = G.pca2(reg[:, :2])
            sp = G._spec(reg, self.tz, m, f"{word} found by part segmentation")
            sp["out"] = G.unit(sp["p"][:2] - bc)
            sp["shape"] = {"part_size_cm": r1(r["size"])}
            out.append(sp)
        side = -1 if arm == "left" else 1
        out.sort(key=lambda s: -side * s["p"][0])
        self.trace.append({"part_specs": [self.cm(s["p"]) for s in out]})
        return out

    # grasp-candidate widening
    GRASP_MORE_BUDGET = {"turned": 8, "from_home": 0, "pre_short": 0}
    GRASP_MORE_FAR = 0.75             # m from the arm base: beyond this no candidate is tried (clearly out of reach)
    GRASP_MORE_PRE = (0.05, 0.03)

    def _gm_plan_q(self, arm, poses, start=None):
        """plan the chain of hand poses (from `start`, a full joint vector, or the current state) like feasible().
        Returns (final active joints, start active joints, final full vector) or None."""
        rb = self.A.T.robot
        pl = rb.left_planner if arm == "left" else rb.right_planner
        cur = np.asarray((rb.left_entity if arm == "left" else rb.right_entity).get_qpos(), float)
        idx = [pl.all_joints.index(n) for n in pl.active_joints_name if n in pl.all_joints]
        last, qa = (None if start is None else np.asarray(start, float)), None
        q0 = (cur if start is None else np.asarray(start, float))[idx]
        for p in poses:
            try:
                res = self.planner(arm)(list(map(float, p)), last_qpos=last) if last is not None else self.planner(arm)(list(map(float, p)))
            except Exception:  # noqa: BLE001
                return None
            if not isinstance(res, dict) or res.get("status") != "Success":
                return None
            pos = res.get("position")
            if pos is None or not len(pos):
                return None
            qa = np.asarray(pos[-1], float)[: len(idx)]
            full = (cur if last is None else last).copy()
            full[idx] = qa
            last = full
        return None if qa is None else (qa, q0, last)

    def _gm_margin(self, arm, qa):
        """smallest distance (rad) of a joint vector to its limits in the planner's model; None if unknown."""
        try:
            rb = self.A.T.robot
            pl = rb.left_planner if arm == "left" else rb.right_planner
            lim = pl.motion_gen.kinematics.get_joint_limits().position.detach().cpu().numpy()
            n = min(len(qa), lim.shape[1])
            return float(np.min(np.minimum(qa[:n] - lim[0, :n], lim[1, :n] - qa[:n])))
        except Exception:  # noqa: BLE001
            return None

    def _gm_pairs(self, s, arm, tall, stage):
        """candidate pairs [(al, a, c, pre), (al, a, -c, pre)] for one spec, in the order they are tried."""
        from transforms3d.axangles import axangle2mat
        if tall and s.get("out") is None:
            base = self._sl_tall_cands(s, arm)            # hook
        else:
            base = G.candidates(s, arm)
        pairs = {}
        if stage in ("from_home", "pre_short"):
            for pre in ((PRE,) if stage == "from_home" else self.GRASP_MORE_PRE):
                for al, a, c in base:
                    pairs.setdefault((round(al), pre, tuple(np.round(a, 3))), []).append((al, a, c, pre))
            return list(pairs.values())
        m = np.r_[np.asarray(s["m"], float)[:2], 0.0]
        dirs = []
        if not tall and s.get("out") is None:
            dirs += [(al, a, 0.0) for al, a, _ in G.candidates(s, arm, tilts=(60.0, 75.0))]
        horiz = []
        for al, a, _ in base + ([] if tall or s.get("out") is not None else G.candidates(s, arm, tilts=(60.0, 75.0))):
            if al > 0 and (round(al), tuple(np.round(a, 3))) not in horiz:
                horiz.append((round(al), tuple(np.round(a, 3))))
                dirs += [(al, a, th) for th in (30.0, -30.0)]
        dirs += [(al, np.asarray(a), th) for al, a in horiz for th in (60.0, -60.0)]
        for al, a, th in dirs:
            a2 = axangle2mat([0, 0, 1.0], np.radians(th)) @ np.asarray(a, float)
            c = self._sl_turn_c(s, a2, m, tall)            # hook (parent expression inside)
            if np.linalg.norm(c) < 0.5:
                continue
            key = (round(al), round(th), tuple(np.round(a2, 3)))
            if key not in pairs:
                pairs[key] = [(al, a2, G.unit(c), PRE), (al, a2, -G.unit(c), PRE)]
        return list(pairs.values())

    def _gm_grasp_more(self, arm, obj, o, specs, exclude):
        """entered only when no parent grasp candidate planned (the parent returns UNREACHABLE right after).
        'turned'   : steeper tilts (60/75 deg) and tilted approaches turned about the vertical by +-30 / +-60 deg;
        'from_home': when the arm is away from home, the parent's candidates planned from the home joint state; if one plans,
                     the arm goes home first;
        'pre_short': the parent's approaches with a 5 / 3 cm pre-grasp back-off.
        Of a symmetric pair (fingers swapped) that both plan, keep more joint-limit margin, then less joint travel."""
        sz = np.asarray(o["size"], float)
        tall = sz[2] >= 10.0 and sz[2] >= 2.0 * max(sz[0], sz[1])
        log = {}
        home_full = None
        if float(np.linalg.norm(np.asarray(specs[0]["p"], float)[:2] - G.ARM_BASE[arm])) > self.GRASP_MORE_FAR if specs else True:
            self.trace.append({"grasp_more": "skip_far"})
            return None
        for stage in ("turned",):
            start = None
            if stage == "from_home":
                hp0 = np.asarray(self.A.home_pose(arm), float)
                if Gm.pos_err(self.A.hand_pose(arm), hp0) < 0.05:
                    continue
                rh = self._gm_plan_q(arm, [hp0])
                if rh is None:
                    continue
                start = home_full = rh[2]
            tried = 0
            for si, s in enumerate(specs[:2]):
                s = dict(s)
                if tall and s.get("out") is None:
                    s["p"] = np.r_[s["p"][:2], self.world(o["lo"])[2] + 0.72 * (self.world(o["hi"])[2] - self.world(o["lo"])[2])]
                for pi, pair in enumerate(self._gm_pairs(s, arm, tall, stage)):
                    if tried >= self.GRASP_MORE_BUDGET[stage]:
                        break
                    best = None
                    for j, (al, a, c, pre) in enumerate(pair):
                        k = (si, f"x_{stage}_{pi}_{j}")
                        if k in exclude:
                            continue
                        q = G.hand_quat(a, c)
                        p = s["p"]
                        hp, hg = self.A.hand_from_tcp(p - a * pre, q), self.A.hand_from_tcp(p, q)
                        tried += 1
                        r1 = self._gm_plan_q(arm, [hp], start)
                        r2 = self._gm_plan_q(arm, [hg], start) if r1 is not None else None
                        if r2 is None:
                            if best is None:
                                continue
                            break
                        mg = self._gm_margin(arm, r1[0])
                        trav = float(np.abs(r1[0] - r1[1]).sum())
                        score = (mg if mg is not None else 0.0, -trav)
                        if best is None or score > best[0]:
                            best = (score, k, hp, hg, q, al, p, pre, mg, trav)
                    if best is None:
                        continue
                    _, k, hp, hg, q, al, p, pre, mg, trav = best
                    if stage == "from_home":
                        if self.run_seq({arm: [{"type": "move", "pose": list(self.A.home_pose(arm))}]})[arm] == "PLAN_FAIL":
                            log[stage] = "home_failed"
                            break
                    self.trace.append({"grasp_more": stage, "tried": tried, "tilt": round(float(al)), "pre_cm": round(pre * 100),
                                       "margin": None if mg is None else round(mg, 2), "travel": round(trav, 2), "log": log})
                    return {"arm": arm, "obj": obj, "pre": hp, "grasp": hg, "q": q, "key": k, "why": s["why"], "obs": o, "spec_p": p,
                            "pregrip": float(s.get("pregrip", 1.0)), "tilt": al, "point_cm": self.cm(p), "shape": s["shape"], "ext": stage}
            log[stage] = tried
        self.trace.append({"grasp_more": "none", "log": log})
        return None

    def _forget_release(self, obj):
        """an object (or the receptacle it was put on / in) was touched again: its release position no longer holds."""
        ra = getattr(self, "_rel_at", {})
        todo, gone = [obj], set()
        while todo:                                       # also what rests on / in it, recursively
            x = todo.pop()
            if x in gone:
                continue
            gone.add(x)
            todo += [n for n, v in ra.items() if v.get("ref") == x]
        for n in gone:
            ra.pop(n, None)

    def _forget_held_contents(self, arm):
        """an arm that holds a container moves it: whatever was released into / onto it moves too."""
        held = self.holding.get(arm)
        if held:
            ra = getattr(self, "_rel_at", {})
            for n in [n for n, v in ra.items() if v.get("ref") == held]:
                self._forget_release(n)

    def grasp(self, arm, obj, part=None):
        self._touched = getattr(self, "_touched", set()) | {obj}
        self._forget_release(obj)
        if self.holding.get(arm):
            raise ValueError(f"{arm} already holds {self.holding[arm]}; place or release it first")
        exclude, last = set(), None
        for attempt in range(3):
            g, err = self.grasp_plan(arm, obj, part, exclude)
            if g is None:
                return {arm: err if last is None else {**err, "previous_attempt": last}}
            exclude.add(g["key"])
            codes = self.run_seq({arm: [{"type": "grip", "val": g.get("pregrip", 1.0)}, {"type": "move", "pose": g["pre"].tolist()},
                                        {"type": "move", "pose": g["grasp"].tolist()}, {"type": "grip", "val": 0.0}]})
            self.soften([arm])
            g["grasp_actual"] = np.asarray(self.A.hand_pose(arm), float)
            code, ex = self.hold_test({arm: obj}) if codes[arm] in ("OK", "POSE_ERR") else ({arm: codes[arm]}, {})
            info = {"code": code[arm], "grasp": g["why"], "grasp_point_cm": g["point_cm"], "approach": "top-down" if g["tilt"] == 0 else ("tilted" if g["tilt"] < 100 else "from the side"), **ex.get(arm, {})}
            if code[arm] == "OK":
                self.holding[arm] = obj
                self.set_hold(arm, g)
                info["attempts"] = attempt + 1
                return {arm: info}
            last = info
            self.open_gently(arm)
            self.run_seq({arm: [self.move_tcp(arm, self.tcp(arm) + [0, 0, 0.06])]})
        return {arm: {**last, "attempts": 3}}

    def set_hold(self, arm, g):
        o = g["obs"]
        R = Gm.mat(g["grasp"])[:3, :3]
        t = self.A.tcp_point(g["grasp"])
        c = self.world(o["center"])
        if str(g.get("why", "")).startswith("body centre"):
            c[:2] = g["spec_p"][:2]               # parallel fingers centre the body between them
        b = np.r_[c[:2], self.world(o["lo"])[2]]
        _, u, _ = G.top_axes(o["pts"])
        bc = G.body_center(o["pts"])
        fz = G.floor_z(o["pts"], bc)
        fl = np.r_[bc, fz if fz is not None else self.world(o["lo"])[2]]
        fL, fW, _ = footprint(o["pts"])
        self.hold[arm] = {"c": R.T @ (c - t), "b": R.T @ (b - t), "size": np.asarray(o["size"], float), "u": R.T @ np.r_[u, 0.0],
                          "floor": R.T @ (fl - t), "top": float(self.world(o["hi"])[2] - t[2]), "fp": (fL, fW)}
        pts = np.asarray(o["pts"], float)
        if len(pts) > 1500:
            pts = pts[np.random.default_rng(0).choice(len(pts), 1500, replace=False)]
        self.hold[arm]["pts_h"] = (R.T @ (pts - t).T).T              # object points in the hand frame for tracking

    def refresh_hold(self, arm, obj):
        """after the lift test the object hangs in the gripper; measure it with the cameras and move the in-hand
        estimate to the measured footprint centre when the two disagree by 1-8 cm (the object shifted while closing)."""
        est = self.held_est(arm)
        if est is None:
            return
        self._drop = False
        try:
            o = self.see(obj)
        finally:
            self._drop = True
        if o is None or len(o["pts"]) < 30:
            self.trace.append({"hold_refresh": "object not measured"})
            return
        cam = np.asarray(G.body_center(o["pts"]), float)[:2]
        d = float(np.linalg.norm(cam - est[0][:2]))
        rec = {"cam_xy_cm": r1(cam * 100), "est_xy_cm": r1(est[0][:2] * 100), "diff_cm": round(d * 100, 1), "applied": False}
        if 0.01 < d < 0.08:
            hp = self.A.hand_pose(arm)
            R = Gm.mat(hp)[:3, :3]
            t = self.A.tcp_point(hp)
            self.hold[arm]["c"] = R.T @ (np.r_[cam, est[0][2]] - t)
            self.hold[arm]["b"] = R.T @ (np.r_[cam, est[1]] - t)
            rec["applied"] = True
        self.trace.append({"hold_refresh": rec})

    def hold_test(self, pairs):
        """pairs {arm: obj}: closed fingers not fully shut, then a joint 5 cm lift; the object must rise in the cameras."""
        codes, ex = {}, {}
        for a in pairs:
            gap = self.A.opening(a)
            ex[a] = {"finger_opening": round(gap, 3)}
            codes[a] = "EMPTY_GRASP" if gap < GAP_MIN else ("NOT_CLOSED" if gap > 0.97 else "OK")   # fingers still fully open = wedged, not held
        live = {a: o for a, o in pairs.items() if codes[a] == "OK"}
        if not live:
            return codes, ex
        objs = set(live.values())
        self._drop = False
        self._raw = True                                  # the lift test must look with the cameras
        o0 = {o: self.see(o) for o in objs}
        self.run_seq({a: [self.move_tcp(a, self.tcp(a) + [0, 0, LIFT_TEST])] for a in live})
        o1 = self.perceive(list(objs))
        self._drop = True
        self._raw = False
        for a, o in live.items():
            if self.A.opening(a) < GAP_MIN:
                codes[a] = "SLIPPED"
                continue
            if o0.get(o) is None or o1.get(o) is None:
                ex[a]["lift_test"] = "object not visible; judged by finger opening"
                continue
            dz = float(o1[o]["center"][2] - o0[o]["center"][2])
            ex[a]["rose_cm"] = round(dz, 1)
            if dz < LIFT_TEST * 100 * 0.5:
                codes[a] = "SLIPPED"
        return codes, ex

    def dbg_img(self, tag):
        import os
        if not os.environ.get("V2_DBG"):
            return
        from PIL import Image
        im = self.sim.rd.observe()["images"]
        self._dbg_k = getattr(self, "_dbg_k", 0) + 1
        Image.fromarray(np.concatenate([np.asarray(im[k]) for k in ("head_camera", "left_camera", "right_camera")], 1)).save(f"/tmp/dbgp_{self._dbg_k:02d}_{tag}.jpg")

    # ------------------------------------------------------------ place / carry
    def place_target(self, obj, ref, relation, xy):
        rel = str(relation or "on").lower().replace(" ", "_")
        if ref in (None, "table") or rel == "table":
            if xy is None:
                raise ValueError("ref 'table' needs xy=[x, y] in cm")
            return np.array([xy[0] / 100, xy[1] / 100]), self.tz + 0.005, "table point"
        for ha, ho in self.holding.items():           # reference held by the other hand: use its in-hand geometry
            if ho == ref and ha in self.hold and rel in ("in", "into", "inside", "on", "onto", "on_top_of", "at"):
                hp = self.A.hand_pose(ha)
                fl = self.A.tcp_point(hp) + Gm.mat(hp)[:3, :3] @ self.hold[ha]["floor"]
                return fl[:2], fl[2] + 0.015, "held container floor"
        r = self.see(ref)
        f0 = self.first.get(ref)
        if (f0 is not None and ref not in getattr(self, "_touched", set()) and ref not in self.holding.values()
                and (r is None or np.linalg.norm(np.asarray(r["center"][:2]) - np.asarray(f0["center"][:2])) > 6.0)):
            # a reference no tool has touched does not move; a detection far from its first observation is a mix-up
            # (e.g. with the held object or a same-coloured object), so keep the first observation
            self.trace.append({"static_ref": ref, "now_cm": None if r is None else r1(r["center"]), "first_cm": r1(f0["center"])})
            r = f0
        if self._frozen(ref) is not None:
            r = self._frozen(ref)
        if r is None:
            raise RuntimeError(f"cannot see {ref}; locate it with a visual description")
        pts = r["pts"]
        uxy = None                                        # the planner's point on / in the reference
        if xy is not None:
            p_ = np.asarray(xy, float)[:2] / 100
            lo_, hi_ = np.asarray(r["lo"][:2], float) / 100 - 0.03, np.asarray(r["hi"][:2], float) / 100 + 0.03
            if np.all(p_ >= lo_) and np.all(p_ <= hi_):
                uxy = p_
            else:
                self.trace.append({"xy_ignored": list(xy), "ref_box_cm": [r1(r["lo"][:2]), r1(r["hi"][:2])]})
        if rel in ("on", "onto", "on_top_of") and r["size"][2] >= 2.0:
            c0 = G.body_center(pts)
            f0 = G.floor_z(pts, c0, r=0.025)
            if f0 is not None and f0 < self.world(r["hi"])[2] - 0.02 and min(r["size"][:2]) > 7.0 and self._rimmed(pts, c0, f0):
                rel = "in"                                 # concave reference (bowl, basket, pan): "on" means nest inside
        if rel in ("in", "into", "inside"):
            c = G.body_center(pts)
            ar = G.arch(pts, self.tz)
            why_in = "container floor"
            if uxy is not None:
                c, why_in = uxy, "container floor at the given xy"
            elif ar is None:
                c, why_in = self._slot(obj, ref, r, c, why_in, "in")       # free part of the receptacle
            if ar is not None and uxy is None:          # keep clear of a handle arching over the middle
                m = np.asarray(ar["m"], float)
                w = G.extent((pts[:, :2] - c) @ m)
                hold_xy = self.held_est(self._place_arm)[0][:2] if self._place_arm in self.hold else c
                sgn = 1.0 if float((hold_xy - c) @ m) >= 0 else -1.0
                c = c + sgn * m * 0.26 * w
                why_in = "container floor beside its handle"
            f = G.floor_z(pts, c)
            fz = f if f is not None else self.world(r["lo"])[2] + 0.01
            rim = float(np.percentile(pts[:, 2], 70 if ar is not None else 95))
            if rim - fz > 0.06:                          # deep container: release just above the rim and let it drop in
                return c, rim + 0.01, why_in + " (released above the rim)"
            hfp = (self.hold.get(self._place_arm) or {}).get("fp") if self._place_arm else None
            if hfp is not None and rim - fz > 0.012:     # an object about as large as the opening nests; pressing it to the floor flings things
                rL, rW, _ = footprint(pts)
                if hfp[0] >= 0.75 * rL and hfp[1] >= 0.75 * rW:
                    return c, rim + 0.01, why_in + " (nested: released at the rim)"
            return c, fz + 0.012, why_in
        if rel in ("on", "onto", "on_top_of", "at", "over", "above"):
            c, top = G.top_center(pts)
            why_on = "top surface"
            if r["size"][2] < 1.5 or (r["size"][2] < 5.0 and min(r["size"][:2]) > 8.0):
                c = G.body_center(pts)  # flat or wide low target (mat, pad, plate with a rim): centre of the whole footprint
            if uxy is not None:
                c, why_on = uxy, "top surface at the given xy"
            else:
                c, why_on = self._slot(obj, ref, r, c, why_on, "on")         # free part only when something is already on it
            return c, top + 0.008, why_on
        word = {"left_of": [-1, 0], "right_of": [1, 0], "front_of": [0, -1], "in_front_of": [0, -1], "behind": [0, 1]}.get(rel)
        if word is None:
            raise ValueError(f"unknown relation {relation}")
        o = self.perc.get(obj)
        half = (np.asarray(o["size"][:2]) / 200) if o is not None else np.array([0.03, 0.03])
        v = np.asarray(word, float)
        d = float(np.abs(np.asarray(r["size"][:2]) / 200) @ np.abs(v)) + float(half @ np.abs(v)) + 0.05     # a clear 5 cm gap
        return np.asarray(r["center"][:2]) / 100 + v * d, self.tz + 0.005, "beside"

    def _rimmed(self, pts, c, fz):
        """a container has a raised edge all around: in at least 3 of 4 sectors the outer points rise well above the floor
        (a scale with a raised display or a coaster with one lip is not a container)."""
        try:
            d = pts[:, :2] - np.asarray(c, float)[:2]
            rr = np.linalg.norm(d, axis=1)
            R = float(np.percentile(rr, 95))
            ang = np.arctan2(d[:, 1], d[:, 0])
            ok = 0
            for k in range(4):
                a0 = -np.pi + k * np.pi / 2
                s_ = (ang >= a0) & (ang < a0 + np.pi / 2) & (rr > 0.6 * R)
                if s_.sum() >= 5 and float(np.percentile(pts[s_, 2], 80)) > fz + 0.015:
                    ok += 1
            self.trace.append({"rimmed_sectors": ok})
            return ok >= 3
        except Exception:  # noqa: BLE001
            return True

    def _slot(self, obj, ref, r, c, why, rel="on"):
        """free part of a receptacle along its long side. Something already on / in it -> the half away from it (when two
        fit).  Nothing on it: "on" -> the centre; "in" with room for two -> the half nearer this arm (leaves room for a second)."""
        try:
            pts = np.asarray(r["pts"], float)
            cc, u, m = G.pca2(pts[:, :2])
            L = G.extent((pts[:, :2] - cc) @ u)
            hfp = (self.hold.get(self._place_arm) or {}).get("fp") if self._place_arm else None
            ob = (float(hfp[0]) / 100) if hfp is not None else 0.06
            u = np.asarray(u, float)
            lo, hi = np.asarray(r["lo"][:2], float) / 100, np.asarray(r["hi"][:2], float) / 100
            occ = []
            for n, o in self.perc.items():
                if n in (obj, ref) or o is None or n in self.holding.values():
                    continue
                oc = np.asarray(o["center"][:2], float) / 100
                if np.all(oc > lo - 0.01) and np.all(oc < hi + 0.01):
                    occ.append(oc)
            slots = [np.asarray(c, float)[:2] + sg * u * L / 4 for sg in (-1.0, 1.0)]
            if occ:
                if L < 1.8 * ob or L < 0.10:
                    return c, why
                sl = max(slots, key=lambda v: min(float(np.linalg.norm(v - oc)) for oc in occ))
                tag = "free slot beside what is already there"
            else:
                if rel != "in" or L < 2.2 * ob or L < 0.12:
                    return c, why
                twin = False                         # reserve room only when another object of about this size still lies outside it
                for n, o in self.perc.items():
                    if n in (obj, ref) or o is None or n in self.holding.values() or "pts" not in o:
                        continue
                    oc = np.asarray(o["center"][:2], float) / 100
                    if np.all(oc > lo - 0.01) and np.all(oc < hi + 0.01):
                        continue
                    if 0.6 * ob * 100 <= footprint(o["pts"])[0] <= 1.6 * ob * 100:
                        twin = True
                        break
                if not twin:
                    return c, why
                base = G.ARM_BASE.get(self._place_arm, np.zeros(2))
                sl = min(slots, key=lambda v: float(np.linalg.norm(v - base)))
                tag = "half nearer this arm, room left for another"
            side = "left" if sl[0] < c[0] - 0.01 else ("right" if sl[0] > c[0] + 0.01 else ("front" if sl[1] < c[1] else "back"))
            self.trace.append({"slot": [round(float(x) * 100, 1) for x in sl], "occupied": len(occ), "rel": rel})
            return sl, f"{why}, {side} half ({tag})"
        except Exception as e:  # noqa: BLE001
            self.trace.append({"slot_error": repr(e)[:120]})
            return c, why

    def _face_turn(self, arm, obj, face, face_now=None):
        """yaw (deg, |yaw| <= 90) that turns the held object's flat face (its thinnest horizontal side, from the tracked
        in-hand points) toward the world direction `face`; the opposite side then faces away (add 180 to flip)."""
        d = {"front": [0, -1.0], "back": [0, 1.0], "left": [-1.0, 0], "right": [1.0, 0]}.get(str(face).lower())
        if d is None or arm not in self.hold or "pts_h" not in self.hold[arm]:
            return None
        P_ = self._tracked(arm, obj)["pts"]
        _, u, m = G.pca2(P_[:, :2])
        L0, W0 = G.extent(P_[:, :2] @ u), G.extent(P_[:, :2] @ m)
        best, sl = (L0 / max(W0, 1e-3), u, m), 0.0
        z0, z1 = float(np.percentile(P_[:, 2], 2)), float(np.percentile(P_[:, 2], 98))
        for frac in (0.35, 0.55):                          # a stand / base widens the whole footprint; the face is thinnest higher up
            Q = P_[P_[:, 2] >= z0 + frac * (z1 - z0)]
            if len(Q) < 40:
                continue
            _, uq, mq = G.pca2(Q[:, :2])
            r = G.extent(Q[:, :2] @ uq) / max(G.extent(Q[:, :2] @ mq), 1e-3)
            if r > 1.15 * best[0]:
                best, sl = (r, uq, mq), frac
        _, u, m = best
        cur = np.degrees(np.arctan2(m[1], m[0]))
        want = np.degrees(np.arctan2(d[1], d[0]))
        y = (want - cur + 90) % 180 - 90
        dn = {"front": [0, -1.0], "back": [0, 1.0], "left": [-1.0, 0], "right": [1.0, 0]}.get(str(face_now or "").lower())
        if dn is not None:                                 # the side that points toward face_now now is the one to turn (signed, full range)
            n = np.asarray(m[:2], float) * (1.0 if float(np.dot(m[:2], dn)) >= 0 else -1.0)
            cur = np.degrees(np.arctan2(n[1], n[0]))
            y = (want - cur + 180) % 360 - 180
        self.trace.append({"face_turn": round(float(y), 1), "thin_axis_deg": round(float(cur), 1), "face_now": face_now, "face_slice": sl,
                           "face_aspect": round(float(best[0]), 2), "full_aspect": round(float(L0 / max(W0, 1e-3)), 2)})
        return float(y)

    def place(self, arm, obj, ref, relation="on", xy=None, release=True, height_cm=10.0, align=False, turn_deg=None, held_part=None, face=None, face_now=None):
        self._forget_held_contents(arm)                    # a carried container takes its contents along
        if ref not in (None, "table"):
            self._forget_release(ref)                      # putting something onto / into ref may nudge it
        if face and self.holding.get(arm) == obj:          # face -> a turn about the vertical (plus the planner's extra turn_deg)
            ft = self._face_turn(arm, obj, face, face_now)
            if ft is not None:
                turn_deg = ft + float(turn_deg or 0.0)
                turn_deg = (turn_deg + 180) % 360 - 180
        """v2h wrapper: lean/insert mode for tall supports with a small top; one automatic re-grasp and re-place when
        the released object is measured off the target; OFF_TARGET is reported instead of OK when it stays off."""
        if self.holding.get(arm) != obj:
            raise ValueError(f"{obj} is not held by the {arm} arm (grasp it first)")
        if release and self._lean_needed(arm, obj, ref, relation):
            r = self._place_lean(arm, obj, ref)
            self._mark_touched(ref, release)
            return r
        r = self._place_once(arm, obj, ref, relation, xy, release, height_cm, align, turn_deg, held_part)
        self._mark_touched(ref, release)          # AFTER the target was computed (the static-reference guard stays active for this place)
        out = r.get(arm) or {}
        off = out.get("off_target")
        if release and off and off.get("dx_cm") is not None and str(out.get("target", "")).startswith(("top surface", "table point")) \
                and 1.5 <= float(np.hypot(off["dx_cm"], off["dy_cm"])) <= 10.0 and not self.holding.get(arm):
            g = self.grasp(arm, obj)
            if (g.get(arm) or {}).get("code") == "OK":
                td2 = None                                   # the object already carries the first turn; face is re-solved from the new hold
                if face:
                    td2 = self._face_turn(arm, obj, face, face if face_now else None)
                if td2 is None and (face or turn_deg is not None):
                    td2 = 0.0                                # keep the reached orientation (no free re-turn, no tilt)
                r2 = self._place_once(arm, obj, ref, relation, xy, release, height_cm, align, td2, held_part)
                r2[arm]["first_attempt_offset_cm"] = [off["dx_cm"], off["dy_cm"]]
                r2[arm]["regrasped"] = True
                return r2
            out["regrasp"] = (g.get(arm) or {}).get("code")
        return r

    def _frozen(self, ref):
        """a flat reference (pad, mat; under 1.5 cm at its first observation) that no gripper grasped keeps its first
        observation: once something lies on it, its segmentation shrinks or drifts."""
        f0 = self.first.get(ref)
        if (f0 is not None and float(f0["size"][2]) < 1.5 and ref not in getattr(self, "_touched", set())
                and ref not in self.holding.values()):
            return f0
        return None

    def _refv(self, ref):
        return self._frozen(ref) or self.perc.get(ref)

    def _mark_touched(self, ref, release):
        """a receptacle that received an object may move afterwards. flat references (pads, mats) do not move when
        something is put on them, so they keep their first observation unless a gripper grasped them."""
        if not release or ref in (None, "table"):
            return
        f0 = self.first.get(ref)
        if f0 is not None and float(f0["size"][2]) < 1.5:
            return
        self._touched = getattr(self, "_touched", set()) | {ref}

    LEAN_RELATIONS = ("against", "lean", "lean_against", "in_slot", "insert")
    LEAN_DEG, LEAN_DY, LEAN_DZ, LEAN_PRESS = (25.0, 15.0, 35.0), -0.01, -0.01, 0.0

    def _ref_obs(self, ref):
        r = self.perc.get(ref)
        f0 = self.first.get(ref)
        if f0 is not None and ref not in getattr(self, "_touched", set()):
            r = f0 if r is None or np.linalg.norm(np.asarray(r["center"][:2]) - np.asarray(f0["center"][:2])) > 6.0 else r
        return r

    def _lean_needed(self, arm, obj, ref, relation):
        """the support is tall and its top is much smaller than the held object's footprint, so the object cannot
        lie on it; stand the object up and lean it against the support (phone on a stand, plate in a rack)."""
        rel = str(relation or "on").lower().replace(" ", "_")
        if ref in (None, "table") or arm not in self.hold:
            return False
        if rel in self.LEAN_RELATIONS:
            return True
        if rel not in ("on", "onto", "on_top_of", "in", "into", "inside"):
            return False
        r = self._ref_obs(ref)
        if r is None or len(r["pts"]) < 30:
            return False
        pts = r["pts"]
        h = float(r["size"][2])
        z = pts[:, 2]
        top = pts[z >= np.percentile(z, 97) - 0.02]
        ext = float(np.max(np.ptp(top[:, :2], axis=0))) * 100 if len(top) >= 5 else 0.0
        L = float(np.max(np.asarray(self.hold[arm]["size"], float)[:2]))
        dec = h >= 8.0 and ext < 0.6 * L and L >= 8.0
        self.trace.append({"lean_check": {"ref_h_cm": round(h, 1), "top_extent_cm": round(ext, 1), "obj_long_cm": round(L, 1), "lean": dec}})
        return dec

    def _place_lean(self, arm, obj, ref, lean_deg=None):
        lean_deg = lean_deg or self.LEAN_DEG
        from transforms3d.quaternions import mat2quat
        r = self._ref_obs(ref)
        if r is None:
            raise RuntimeError(f"cannot see {ref}; locate it with a visual description")
        est = self.held_est(arm)
        if est is None:
            return {arm: {"code": "NOT_SEEN", "why": f"no in-hand estimate of the held {obj}"}}
        ow, obot = est
        c = (np.asarray(r["lo"][:2], float) + np.asarray(r["hi"][:2], float)) / 200   # footprint box centre (world, m)
        T = np.r_[c[0], c[1] + self.LEAN_DY, self.world(r["hi"])[2] + self.LEAN_DZ]   # object centre near the top of the support
        tcp = self.tcp(arm)
        hp = self.A.hand_pose(arm)
        q0 = hp[3:]
        uw = Gm.mat(hp)[:3, :3] @ self.hold[arm]["u"]
        uw = G.unit(np.r_[uw[:2], 0.0])

        def rot_between(a, b):
            a, b = G.unit(a), G.unit(b)
            v = np.cross(a, b); cth = float(a @ b)
            if np.linalg.norm(v) < 1e-8:
                return np.eye(3) if cth > 0 else -np.eye(3)
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + K + K @ K * (1.0 / (1.0 + cth))
        choice = None
        for ld in lean_deg:
            v = np.array([0.0, np.sin(np.radians(ld)), np.cos(np.radians(ld))])   # long axis up, top leaning away from the robot
            for sgn in (1.0, -1.0):
                Rm = rot_between(sgn * uw, v)
                q = G.quat_mul(np.asarray(mat2quat(Rm)), q0)
                off = Rm @ (ow - tcp)
                dest = T - off
                app = dest + np.array([0.0, -0.06, 0.05])
                if self.feasible(arm, [self.A.hand_from_tcp(dest, q)]) and self.feasible(arm, [self.A.hand_from_tcp(app, q)]):
                    choice = (ld, q, dest, app)
                    break
            if choice:
                break
        if choice is None:
            return {arm: {"code": "UNREACHABLE", "why": f"the {arm} arm cannot stand {obj} up against {ref}; "
                                                         "bring the support closer to this arm, use the other arm, or handover"}}
        ld, q, dest, app = choice
        hi_z = max(tcp[2], app[2] + 0.06)
        codes = self.run_seq({arm: [self.move_tcp(arm, np.r_[tcp[:2], hi_z]), self.move_tcp(arm, np.r_[app[:2], hi_z], q),
                                    self.move_tcp(arm, app, q), self.move_tcp(arm, dest, q)]})
        self.trace.append({"place_lean": codes[arm], "lean_deg": ld, "target_cm": self.cm(T), "dest": self.cm(dest), "tcp": self.cm(self.tcp(arm))})
        out = {"code": "OK" if codes[arm] != "PLAN_FAIL" else "PLAN_FAIL", "target": "stood up leaning against the support",
               "target_cm": self.cm(T), "xy_corrections_cm": [], "turned_deg": f"stand up, lean {ld:.0f}"}
        if self.LEAN_PRESS:
            self.run_seq({arm: [self.move_tcp(arm, self.tcp(arm) - [0, 0, self.LEAN_PRESS], q)]})
        self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        t = self.tcp(arm)
        back = -Gm.mat(self.A.hand_pose(arm))[:3, 0]          # straight out along the fingers, then up
        self.run_seq({arm: [self.move_tcp(arm, t + 0.05 * back), self.move_tcp(arm, t + 0.06 * back + [0, 0, 0.06])]})
        self.holding.pop(arm, None)
        self.hold.pop(arm, None)
        o3 = self.see(obj)
        L = float(np.max(np.asarray(r["size"], float)[:2]))
        if o3 is None:
            out.update(code="OFF_TARGET", off_target={"dx_cm": None, "dy_cm": None, "dyaw_deg": None},
                       why="released, but the cameras no longer see the object at the support; it may have fallen")
            return {arm: out}
        oc = np.asarray(G.body_center(o3["pts"]), float)[:2]
        d = oc - T[:2]
        upright = not (float(o3["lo"][2]) < 1.5 and float(o3["size"][2]) < 5.0)     # lying flat on the table = fell
        out["object_now"] = {"center": r1(o3["center"]), "bottom_z": round(float(o3["lo"][2]), 1)}
        out["placed_offset_cm"] = r1(d * 100)
        if not upright or float(np.linalg.norm(d)) > 0.06:
            out.update(code="OFF_TARGET", off_target={"dx_cm": round(float(d[0]) * 100, 1), "dy_cm": round(float(d[1]) * 100, 1), "dyaw_deg": None},
                       why="released, but the object is not standing at the support (" + ("fell over" if not upright else "off to the side") + ")")
        return {arm: out}

    def _held_end(self, arm, obj, part):
        """one end of the held object (left / right / front / back end, by the current world direction): its xy centre
        and bottom z, from the tracked in-hand points."""
        want = {"left": [-1.0, 0], "right": [1.0, 0], "front": [0, -1.0], "back": [0, 1.0]}.get(str(part).lower().split()[0])
        if want is None or arm not in self.hold or "pts_h" not in self.hold[arm]:
            return None
        P_ = self._tracked(arm, obj)["pts"]
        c, u, _ = G.pca2(P_[:, :2])
        u = np.asarray(u, float)
        if float(u @ np.asarray(want)) < 0:
            u = -u
        t = (P_[:, :2] - c) @ u
        L = G.extent(t)
        reg = P_[t >= np.percentile(t, 97) - min(0.05, 0.22 * L)]
        if len(reg) < 5:
            return None
        return np.r_[np.median(reg[:, :2], 0), float(np.median(reg[:, 2]))], float(np.percentile(reg[:, 2], 5))

    def _place_once(self, arm, obj, ref, relation="on", xy=None, release=True, height_cm=10.0, align=False, turn_deg=None, held_part=None):
        if self.holding.get(arm) != obj:
            raise ValueError(f"{obj} is not held by the {arm} arm (grasp it first)")
        self._place_arm = arm
        txy, tbot, why = self.place_target(obj, ref, relation, xy)
        est = self.held_est(arm)
        if est is not None:
            ow, obot = est
        else:
            o = self.see(obj)
            if o is None:
                return {arm: {"code": "NOT_SEEN", "why": f"cannot see the held {obj}"}}
            ow = self.world(o["center"])
            obot = self.world(o["lo"])[2]
        anchor = self._held_end(arm, obj, held_part) if held_part else None
        if anchor is not None:                           # that end of the held object goes to the target
            ow, obot = anchor
            why = f"{why}; {held_part} of {obj} over it"
        tcp = self.tcp(arm)
        if not release:
            tbot = max(tbot, self.tz + float(height_cm) / 100)
        q0 = self.A.hand_pose(arm)[3:]
        dz_ = tbot - obot
        rp = self._refv(ref) if ref not in (None, "table") else None
        clear = (self.world(rp["hi"])[2] + (tcp[2] - obot) + 0.04) if rp is not None else 0.0   # carry above the reference
        choice = None
        yaws = [0, 20, -20, 40, -40, 60, -60, 90, -90]
        auto_align = False
        if (not align and release and arm in self.hold and ref not in (None, "table") and self._refv(ref) is not None
                and str(relation or "on").lower() in ("on", "onto", "on_top_of")):
            rs = np.asarray(self._refv(ref)["size"], float)
            rL, rW, _ = footprint(self._refv(ref)["pts"])
            hL, hW = self.hold[arm].get("fp") or tuple(np.sort(np.asarray(self.hold[arm]["size"], float)[:2])[::-1])
            if rs[2] < 1.5 and rL / max(rW, 0.5) >= 1.4 and hL / max(hW, 0.5) >= 1.4:
                align = auto_align = True          # elongated object onto an elongated flat target (min-area rectangle) -> line up long sides
        if align and arm in self.hold and ref not in (None, "table") and self._refv(ref) is not None:
            _, ur, _ = G.pca2(self._refv(ref)["pts"][:, :2])
            uo = (Gm.mat(self.A.hand_pose(arm))[:3, :3] @ self.hold[arm]["u"])[:2]
            dl = np.degrees(np.arctan2(ur[1], ur[0]) - np.arctan2(uo[1], uo[0]))
            dl = (dl + 90) % 180 - 90                         # long sides parallel (either direction)
            yaws = [dl + e for e in (0, 180, -180, 8, -8)]
            yaws = [y for y in yaws if -200 <= y <= 200]
        if turn_deg is not None:                               # the planner's turn about the vertical (never replaced by a tilt)
            td = float(turn_deg)
            yaws = [td + e for e in (0, 8, -8, 15, -15)]
        from transforms3d.axangles import axangle2mat
        from transforms3d.quaternions import mat2quat
        rdir = G.unit(np.r_[txy - G.ARM_BASE[arm], 0.0])
        kax = G.unit(np.cross([0, 0, 1.0], rdir))
        rots = [(y, 0.0) for y in yaws] + ([] if turn_deg is not None else [(0.0, t) for t in (20.0, -20.0, 35.0, -35.0, 45.0, -45.0)])   # tilting is the last resort
        bpt = np.r_[ow[:2], obot]
        for yaw, tilt in rots:   # turning about the vertical keeps the object upright
            Rm = axangle2mat([0, 0, 1.0], np.radians(yaw)) if not tilt else axangle2mat(kax, np.radians(tilt))
            q = G.quat_mul(np.asarray(mat2quat(Rm)), q0)
            off = Rm @ (ow - tcp)
            boff = Rm @ (bpt - tcp)
            dest = np.r_[txy - off[:2], tbot - boff[2]]
            safe = max(dest[2] + 0.07, clear)
            hd, hs = self.A.hand_from_tcp(dest, q), self.A.hand_from_tcp(np.r_[dest[:2], safe], q)
            if not self.feasible(arm, [hd]):
                continue
            if not self.feasible(arm, [hs]):
                safe = dest[2] + 0.04
                if not self.feasible(arm, [self.A.hand_from_tcp(np.r_[dest[:2], safe], q)]):
                    continue
            choice = (yaw if not tilt else f"tilt {tilt}", q, dest, safe)
            break
        if choice is None:
            return {arm: {"code": "UNREACHABLE", "why": f"the {arm} arm cannot reach over the target ({why}) with {obj}" +
                          (f" turned by {turn_deg} deg; try a smaller turn or turn it in two steps" if turn_deg is not None else "") +
                          "; bring the target closer to this arm, use the other arm, or handover"}}
        yaw, q, dest, safe = choice
        codes = self.run_seq({arm: [self.move_tcp(arm, np.r_[tcp[:2], max(tcp[2], safe)]), self.move_tcp(arm, np.r_[tcp[:2], max(tcp[2], safe)], q),
                                    self.move_tcp(arm, np.r_[dest[:2], safe], q)]})
        self.dbg_img("over")
        self.trace.append({"place_over": codes[arm], "yaw": yaw, "tcp": self.cm(self.tcp(arm)), "dest": self.cm(dest), "obj_est": self.cm(ow), "tcp0": self.cm(tcp)})
        if codes[arm] == "PLAN_FAIL":                    # other waypoint sequences to the same verified destination pose
            c2, st = self._transport(arm, dest[:2], safe, q)
            self.trace.append({"transport_fallback": c2, "strategies": st, "tcp": self.cm(self.tcp(arm))})
            codes[arm] = "OK" if c2 == "OK" else "PLAN_FAIL"
        if codes[arm] == "PLAN_FAIL":
            return {arm: {"code": "UNREACHABLE", "why": f"cannot carry {obj} over the target ({why}); try the other arm or handover"}}
        corr = []
        for _ in range(0 if arm in self.hold else 2):     # closed-loop xy correction from the cameras (no in-hand estimate)
            o2 = self.see(obj)
            pred = self.held_est(arm)
            if o2 is None:
                break
            seen = self.world(o2["center"])[:2]
            if pred is not None and np.linalg.norm(seen - pred[0][:2]) > 0.03:
                break                                     # camera view of the held object disagrees with the hand: trust the hand
            err = txy - seen
            if np.linalg.norm(err) < 0.004 or np.linalg.norm(err) > 0.05:
                break
            corr.append(round(float(np.linalg.norm(err)) * 100, 1))
            t = self.tcp(arm)
            dest[:2] += err
            self.run_seq({arm: [self.move_tcp(arm, np.r_[t[:2] + err, t[2]])]})
        codes = self.run_seq({arm: [self.move_tcp(arm, dest)]})
        if codes[arm] == "PLAN_FAIL":                    # lower in smaller steps
            codes[arm] = self._move_steps(arm, dest)
            self.trace.append({"lower_in_steps": codes[arm], "tcp": self.cm(self.tcp(arm))})
        self.dbg_img("down")
        if release and codes[arm] == "PLAN_FAIL" and float(self.tcp(arm)[2] - dest[2]) > 0.04:   # never let go high above the target
            self.trace.append({"place_down_failed_high": self.cm(self.tcp(arm))})
            return {arm: {"code": "PLAN_FAIL", "why": f"could not lower {obj} onto the target ({why}); still holding it "
                          f"{(self.tcp(arm)[2] - dest[2]) * 100:.0f} cm above; try the other arm, another xy or handover", "target": why}}
        self.trace.append({"place_down": codes[arm], "tcp": self.cm(self.tcp(arm)), "dest": self.cm(dest)})
        out = {"code": "OK" if codes[arm] != "PLAN_FAIL" else "PLAN_FAIL", "target": why, "target_cm": self.cm(np.r_[txy, tbot]),
               "xy_corrections_cm": corr}
        if yaw:
            out["turned_deg"] = yaw if isinstance(yaw, str) else round(float(yaw))
        if release:
            est = self.held_est(arm)
            hs_ = np.asarray(self.hold.get(arm, {}).get("size", [0, 0, 0]), float)
            self._held_h = float(hs_[2]) if (hs_[2] >= 10.0 and hs_[2] >= 2.0 * max(hs_[0], hs_[1])) or (hs_[2] >= 5.0 and hs_[2] >= 1.2 * min(hs_[0], hs_[1])) else None
            self.open_gently(arm)
            self.dbg_img("open")
            self.retreat(arm)
            self.dbg_img("up")
            self.holding.pop(arm, None)
            self.hold.pop(arm, None)
            if est is not None and obj in self.perc:      # dead-reckoned new position, used as the perception prior
                p = dict(self.perc[obj])
                p["center"] = np.r_[est[0][0] * 100, est[0][1] * 100, p["center"][2] - p["lo"][2] + (tbot - self.tz) * 100]
                self.perc[obj] = p
            o3 = self.see(obj)
            if not hasattr(self, "_rel_at"):
                self._rel_at = {}
            jump = None if (o3 is None or est is None) else float(np.linalg.norm(self.world(o3["center"])[:2] - est[0][:2]))
            if o3 is not None and jump is not None and jump > 0.10 and _implausible(o3, hs_):     # implausible re-measurement (another object / a fragment)
                out["object_now"] = {"center": r1(o3["center"]), "bottom_z": round(float(o3["lo"][2]), 1)}
                if out.get("code") == "OK":
                    out["code"] = "MEASURE_UNSURE"
                out["why"] = (f"released; the cameras now report {obj} {jump * 100:.0f} cm from where it was let go: either a mis-detection or it "
                              f"slid / bounced away; look at the head image: if {obj} sits on the target, the placement worked")
                self._rel_at[obj] = {"xy": np.asarray(est[0][:2], float), "ref": ref, "size": hs_}
                self.trace.append({"measure_unsure_jump_cm": round(jump * 100, 1)})
            elif o3 is not None:
                out["object_now"] = {"center": r1(o3["center"]), "bottom_z": round(float(o3["lo"][2]), 1)}
                self._check_placed(out, obj, ref, o3, txy, why, align)
                self._rel_at[obj] = {"xy": np.asarray(self.world(o3["center"])[:2], float), "ref": ref, "size": np.asarray(o3["size"], float)}
            elif out.get("code") == "OK":               # not seen after release -> not verified (it may have bounced away)
                out["code"] = "RELEASED_NOT_SEEN"
                out["why"] = f"released, but the cameras cannot find {obj} now; check the images (it may have fallen or bounced away)"
        return {arm: out}

    # transport fallbacks
    @staticmethod
    def _slerp(q0, q1, f):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        d = float(np.dot(q0, q1))
        if d < 0:
            q1, d = -q1, -d
        if d > 0.9995:
            q = q0 + f * (q1 - q0)
        else:
            th = np.arccos(np.clip(d, -1, 1))
            q = (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)
        return q / np.linalg.norm(q)

    def _transport(self, arm, dxy, safe, q):
        """bring the TCP to (dxy, safe) with orientation q after the waypoints failed (the arm is wherever
        they left it).  Each strategy starts from the current pose; a waypoint that does not plan ends the strategy.
        Returns (code, strategy)."""
        tried = []
        for strat in ("diagonal", "direct", "turn_at_target", "diagonal_low"):
            p0 = self.tcp(arm)
            q0 = self.A.hand_pose(arm)[3:]
            end = np.r_[np.asarray(dxy, float)[:2], safe]
            if strat == "diagonal":
                z1 = min(max(float(p0[2]), safe), float(p0[2]) + 0.06)
                wps = [(np.r_[p0[:2], z1], q0)]
                for f in (1 / 3, 2 / 3, 1.0):
                    wps.append((np.r_[p0[:2] + f * (end[:2] - p0[:2]), z1 + f * (safe - z1)], self._slerp(q0, q, f)))
            elif strat == "direct":
                wps = [(end, q)]
            elif strat == "turn_at_target":
                wps = [(np.r_[p0[:2], min(max(float(p0[2]), safe), float(p0[2]) + 0.04)], q0), (end, q0), (end, q)]
            else:                                         # through the middle of the table, closer to the robot
                mid = np.r_[0.5 * (p0[:2] + end[:2]) + np.array([0.0, -0.06]), max(float(p0[2]), safe) - 0.03]
                wps = [(mid, self._slerp(q0, q, 0.5)), (end, q)]
            code = "OK"
            for p, qq in wps:
                c = self.run_seq({arm: [self.move_tcp(arm, p, qq)]})[arm]
                if c == "PLAN_FAIL":
                    code = "PLAN_FAIL"
                    break
            tried.append(strat)
            if code == "OK" and Gm.pos_err(self.A.hand_pose(arm), self.A.hand_from_tcp(end, q)) < 0.03:
                return "OK", tried
        return "PLAN_FAIL", tried

    def _move_steps(self, arm, target, q=None, n=(2, 3)):
        """a straight TCP move that did not plan, retried as n equal steps (the arm stops where a step fails)."""
        for k in n:
            p0 = self.tcp(arm)
            ok = True
            for i in range(1, k + 1):
                c = self.run_seq({arm: [self.move_tcp(arm, p0 + (np.asarray(target, float) - p0) * i / k, q)]})[arm]
                if c == "PLAN_FAIL":
                    ok = False
                    break
            if ok:
                return "OK"
        return "PLAN_FAIL"

    def _residual(self, arm, obj, txy, lo=0.004, hi=0.06):
        """target xy minus the camera-measured footprint centre of the held object (None if negligible or implausible)."""
        o = self.see(obj)
        if o is None or len(o["pts"]) < 30:
            self.trace.append({"residual": "held object not measured"})
            return None
        seen = np.asarray(G.body_center(o["pts"]), float)[:2]
        e = np.asarray(txy, float)[:2] - seen
        pred = self.held_est(arm)
        rec = {"seen_cm": r1(seen * 100), "target_cm": r1(np.asarray(txy)[:2] * 100), "err_cm": round(float(np.linalg.norm(e)) * 100, 1)}
        if pred is not None:
            rec["pred_cm"] = r1(pred[0][:2] * 100)
        self.trace.append({"residual": rec})
        n = float(np.linalg.norm(e))
        if n < lo or n > hi:
            return None
        if pred is not None and float(np.linalg.norm(seen - pred[0][:2])) > 0.06:
            return None                                   # camera and hand disagree grossly: do not act on it
        return e

    def _check_placed(self, out, obj, ref, o3, txy, why, align):
        """after release compare the object footprint with the intended target; report OFF_TARGET instead of OK."""
        hz = getattr(self, "_held_h", None)
        if hz and float(o3["size"][2]) < 0.65 * hz and not str(why).startswith(("container", "held container")):              # an object that lost a third of its height has fallen over
            out["code"] = "OFF_TARGET"
            out["off_target"] = {"dx_cm": None, "dy_cm": None, "dyaw_deg": None}
            out["why"] = f"released, but the object fell over (height {float(o3['size'][2]):.0f} cm, was {hz:.0f} cm)"
            return
        oc = np.asarray(G.body_center(o3["pts"]), float)[:2]
        d = oc - np.asarray(txy, float)[:2]
        off, dyaw = False, None
        w = str(why)
        if w.startswith("held container"):
            return
        if w.startswith("container floor"):
            r = self._refv(ref)
            if r is not None and r is not o3:
                lo = np.asarray(r["lo"][:2], float) / 100 + 0.01
                hi = np.asarray(r["hi"][:2], float) / 100 - 0.01
                off = bool(np.any(oc < lo) or np.any(oc > hi))
        elif w.startswith("top surface") or w.startswith("table point"):
            off = float(np.linalg.norm(d)) > 0.03
        else:
            return
        if align and ref not in (None, "table") and self._refv(ref) is not None:
            try:
                _, uo, _ = G.top_axes(o3["pts"])
                _, ur, _ = G.pca2(self._refv(ref)["pts"][:, :2])
                a = np.degrees(np.arctan2(uo[1], uo[0]) - np.arctan2(ur[1], ur[0]))
                dyaw = round(float((a + 90) % 180 - 90), 1)
                off = off or abs(dyaw) > 12
            except Exception:  # noqa: BLE001
                dyaw = None
        out["placed_offset_cm"] = r1(d * 100)
        if off:
            out["code"] = "OFF_TARGET"
            out["off_target"] = {"dx_cm": round(float(d[0]) * 100, 1), "dy_cm": round(float(d[1]) * 100, 1), "dyaw_deg": dyaw}
            out["why"] = "released, but the cameras measure the object away from the target; re-grasp and place again if the task needs it"

    # ------------------------------------------------------------ other skills
    def handover(self, obj, to_arm):
        """mid-air hand-over first: the giver brings the object to the middle, the receiver grasps the half of it
        away from the giver's fingers with a side approach, the giver lets go. Fallback: put it down between the
        arms and regrasp from the table."""
        frm = "left" if to_arm == "right" else "right"
        if self.holding.get(frm) != obj:
            raise ValueError(f"the {frm} arm does not hold {obj}")
        o = self.see(obj)
        y = float(o["center"][1]) if o is not None else 0.0
        mid = [4.0 if to_arm == "right" else -4.0, min(max(y, -10.0), -2.0)]
        r1_ = self._place_once(frm, obj, "table", "table", xy=mid)
        if r1_[frm]["code"] not in ("OK", "OFF_TARGET", "MEASURE_UNSURE"):          # the hand-over put-down point need not be exact
            return {"code": "HANDOVER_FAIL", "put_down": r1_[frm]}
        self.call("home", {"arm": frm})
        r2 = self.grasp(to_arm, obj)
        return {"code": r2[to_arm]["code"], "mode": "via the table", "put_down_at_cm": mid, to_arm: r2[to_arm]}

    def handover_air(self, frm, to, obj):
        hd = self.hold.get(frm)
        est = self.held_est(frm)
        if hd is None or est is None:
            return None
        size = np.asarray(hd["size"], float) / 100
        hgt = float(size[2])
        tcp = self.tcp(frm)
        # bring the object to the middle, bottom 12 cm above the table
        c0, bot0 = est
        sx = 1.0 if to == "right" else -1.0
        goal_c = np.array([-sx * 0.02, -0.06, self.tz + 0.12 + (c0[2] - bot0)])
        d = goal_c - c0
        c1 = self.run_seq({frm: [self.move_tcp(frm, np.r_[tcp[:2], max(tcp[2], goal_c[2] + 0.05)]), self.move_tcp(frm, tcp + d)]})
        if c1[frm] == "PLAN_FAIL":
            return None
        c, bot = self.held_est(frm)
        t_g = self.tcp(frm)
        # receiver point: on the object's receiver side / lower half, away from the giver's fingers
        cand = []
        lowz = bot + 0.3 * hgt if hgt > 0.07 else c[2]
        for gp in (np.r_[c[:2], lowz], np.r_[c[0] + sx * 0.25 * size[0], c[1], c[2]]):
            for tilt in (0.0, 25.0, 45.0):
                tl = np.radians(tilt)
                a = G.unit(np.array([-sx * np.cos(tl), 0.0, -np.sin(tl)]))
                for cv in ([0, 1.0, 0], [0, -1.0, 0]):
                    cand.append((gp, a, np.asarray(cv)))
        for gp, a, cv in cand:
            if np.linalg.norm(gp - t_g) < 0.04:
                continue
            q = G.hand_quat(a, cv)
            pre, hg = self.A.hand_from_tcp(gp - a * 0.08, q), self.A.hand_from_tcp(gp, q)
            if not (self.feasible(to, [pre]) and self.feasible(to, [hg])):
                continue
            self.run_seq({to: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}, {"type": "move", "pose": hg.tolist()}]})
            self.run_seq({to: [{"type": "grip", "val": 0.0}]})
            if self.A.opening(to) < GAP_MIN:
                self.run_seq({to: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}]})
                continue
            self.run_seq({frm: [{"type": "grip", "val": 1.0}]})
            self.retreat(frm, 0.06)
            self.holding.pop(frm, None)
            self.hold.pop(frm, None)
            self.call("home", {"arm": frm})
            if self.A.opening(to) < GAP_MIN:
                return {"code": "DROPPED", "why": "the receiving gripper closed fully after the giver let go"}
            self.holding[to] = obj
            R = Gm.mat(self.A.hand_pose(to))[:3, :3]
            tt = self.tcp(to)
            self.hold[to] = {"c": R.T @ (c - tt), "b": R.T @ (np.r_[c[:2], bot] - tt), "size": hd["size"]}
            return {"code": "OK", "mode": "mid-air", to: {"code": "OK", "finger_opening": round(self.A.opening(to), 3)}}
        return None

    def pour(self, arm, direction="left", degrees=100.0, shake=True, back=False):
        """tip the held object about the horizontal axis through its centre so that its top turns toward direction."""
        from transforms3d.axangles import axangle2mat
        from transforms3d.quaternions import mat2quat
        obj = self.holding.get(arm)
        est = self.held_est(arm)
        if not obj or est is None:
            return {arm: {"code": "NOT_HOLDING", "why": f"the {arm} arm holds nothing to pour"}}
        d = {"left": [-1.0, 0, 0], "right": [1.0, 0, 0], "front": [0, -1.0, 0], "back": [0, 1.0, 0]}.get(str(direction).lower())
        if d is None:
            return {arm: {"code": "BAD_ARG", "why": "direction must be left, right, front or back"}}
        k = G.unit(np.cross([0, 0, 1.0], d))                 # rotating the up axis about k turns it toward d
        c = est[0]
        t0, q0 = self.tcp(arm), self.A.hand_pose(arm)[3:]

        def pose(th):
            Rk = axangle2mat(k, np.radians(th))
            return self.move_tcp(arm, c + Rk @ (t0 - c), G.quat_mul(np.asarray(mat2quat(Rk)), q0))
        done_ = 0
        steps_ = list(np.arange(15.0, float(degrees), 15.0)) + [float(degrees)]   # the last step is the requested angle itself
        for th in steps_:
            if self.run_seq({arm: [pose(th)]})[arm] == "PLAN_FAIL":
                break
            done_ = float(th)
        if shake and done_ >= 30:
            for th in (done_ - 15, done_, done_ - 15, done_):
                self.run_seq({arm: [pose(th)]})
        if back:
            for th in np.arange(done_ - 15.0, -0.1, -15.0):
                self.run_seq({arm: [pose(max(th, 0.0))]})
        return {arm: {"code": "OK" if done_ >= float(degrees) - 10.0 else "SHORT", "tipped_deg": done_, "still_holding": obj,
                      "upright_again": bool(back), "note": "whether the contents fell out is not measured; judge it from the images"}}

    def lift(self, arm, h):
        arms = list(ARMS) if arm == "both" else [arm]
        seq = {}
        for a in arms:
            t = self.tcp(a)
            z = self.tz + float(h) / 100
            if arm == "both":                            # both hands move together to one height (a shared object stays level)
                zs = [float(self.tcp(x)[2]) for x in ARMS]
                if min(zs) >= z - 0.01:
                    continue
                z = max(z, max(zs))
            elif z < t[2] + 0.01:                        # absolute height; already that high -> stay (never lower a load)
                continue
            seq[a] = [self.move_tcp(a, np.r_[t[:2], z])]
        codes = self.run_seq(seq) if seq else {}
        drawn = {}
        if arm != "both":
            for a in [x for x in seq if codes.get(x) == "PLAN_FAIL"]:     # smaller steps, then a little toward the arm base
                z = self.tz + float(h) / 100
                codes[a] = self._move_steps(a, np.r_[self.tcp(a)[:2], z])
                for back in (0.04, 0.08):
                    if codes[a] != "PLAN_FAIL":
                        break
                    t0 = self.tcp(a)
                    d = G.ARM_BASE[a] - t0[:2]
                    d = d / max(np.linalg.norm(d), 1e-6) * back
                    codes[a] = self._move_steps(a, np.r_[t0[:2] + d, z], n=(1, 2))
                    if codes[a] != "PLAN_FAIL":
                        drawn[a] = round(back * 100)
        out = {}
        for a in arms:
            out[a] = {"code": codes.get(a, "OK"), "gripper_z_cm": round((self.tcp(a)[2] - self.tz) * 100, 1)}
            if a in drawn:
                out[a]["moved_toward_robot_cm"] = drawn[a]
            if a not in seq:
                out[a]["note"] = f"already at or above {float(h):.0f} cm; not moved"
        return out

    PRESS_DIRS = {"from_front": [0, 1.0, 0], "from_back": [0, -1.0, 0], "from_left": [1.0, 0, 0], "from_right": [-1.0, 0, 0]}

    def press(self, arm, obj, direction=None):
        """press the object with the closed gripper. direction "down" (default: on its top), or "from_front"
        "from_left" / "from_right" / "from_back" (a horizontal push on the face toward that side, e.g. a switch or button
        facing sideways); with direction=None a top press that cannot be reached falls back to tilted and horizontal pushes.
        The gripper opens again afterwards when it holds nothing."""
        self._forget_release(obj)
        o = self.see(obj)
        if o is None:
            return {arm: {"code": "NOT_SEEN"}}
        pts = np.asarray(o["pts"], float)
        c, top = G.top_center(pts)
        dirs = [direction] if direction else ["down", "tilted", "from_front", "from_left" if arm == "right" else "from_right"]
        if direction and direction not in ("down", "tilted", *self.PRESS_DIRS):
            return {arm: {"code": "BAD_ARG", "why": "direction must be down, from_front, from_left, from_right or from_back"}}
        tried, res = [], None
        for dname in dirs:
            if dname in ("down", "tilted"):
                p = np.r_[c, top]
                if dname == "down":
                    a = np.array([0, 0, -1.0]); q = TOP_Q
                else:
                    b = np.radians(50)
                    a = G.unit(np.array([0.0, np.cos(b), -np.sin(b)]))      # pitched down, coming from the robot side
                    q = G.hand_quat(a, [-1.0, 0, 0])
                start, end = p - a * 0.08, p + a * 0.02
            else:
                a = np.asarray(self.PRESS_DIRS[dname], float)
                sproj = pts @ a
                near = pts[sproj <= np.percentile(sproj, 1) + 0.004]     # the most protruding patch (a toggle's raised end), not the whole face
                if len(near) < 5:
                    near = pts[sproj <= np.percentile(sproj, 4) + 0.008]
                p = np.median(near, 0)
                q = G.hand_quat(a, [0, 0, 1.0])
                start, end = p - a * 0.07, p + a * 0.02
            hs_, he_ = self.A.hand_from_tcp(start, q), self.A.hand_from_tcp(end, q)
            if not self.feasible(arm, [hs_]):
                tried.append(f"{dname}: unreachable"); continue
            codes = self.run_seq({arm: [{"type": "grip", "val": 0.0}, {"type": "move", "pose": hs_.tolist()}]})
            if codes[arm] == "PLAN_FAIL":
                tried.append(f"{dname}: plan failed"); continue
            self.A.run({arm: [{"type": "move", "pose": he_.tolist()}]})
            depth = float((self.tcp(arm) - p) @ a)            # how far past the surface point the fingertip got (m)
            self.A.run({arm: [{"type": "move", "pose": hs_.tolist()}]})
            res = (dname, p, depth)
            break
        if not self.holding.get(arm):                    # the gripper opens again (a closed gripper left on the object fails "released" goals)
            self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        if res is None:
            return {arm: {"code": "UNREACHABLE", "why": f"no reachable press direction for {obj}: " + "; ".join(tried)}}
        dname, p, depth = res
        reached = depth >= -0.05          # the TCP point sits a few cm behind the fingertips (same allowance as the old top press)
        if reached:
            self.pressed.add(obj)
        if -0.05 <= depth < 0:
            short_by_cm = round(-depth * 100, 1)
            return {arm: {"code": "PARTIAL", "direction": dname, "pressed_point_cm": self.cm(p),
                          "reach_past_surface_cm": round(depth * 100, 1), "short_by_cm": short_by_cm,
                          "note": f"指尖停在表面前 {short_by_cm:g} cm，可沿同方向再按；按钮/开关是否改变未测量，请先看图确认，避免重复按压撤销已完成的动作"}}
        return {arm: {"code": "OK" if reached else "NOT_REACHED", "direction": dname, "pressed_point_cm": self.cm(p),
                      "reach_past_surface_cm": round(depth * 100, 1),
                      "note": "whether the button / switch changed is not measured; judge it from the images (if it changed, do not press it again: a second press can undo it)"}}

    def push(self, arm, obj, direction, dist_cm):
        self._forget_release(obj)
        v = {"left": [-1, 0, 0], "right": [1, 0, 0], "front": [0, -1, 0], "back": [0, 1, 0]}.get(str(direction).lower())
        o = self.see(obj)
        if v is None or o is None:
            return {arm: {"code": "BAD_ARG" if v is None else "NOT_SEEN"}}
        v = np.asarray(v, float)
        c = self.world(o["center"])
        ext = float(np.abs(np.asarray(o["size"]) / 200) @ np.abs(v))
        start = c - v * (ext + 0.04)
        start[2] = max(c[2], self.tz + 0.02)
        codes = self.run_seq({arm: [{"type": "grip", "val": 0.0}, self.move_tcp(arm, start + [0, 0, 0.08], TOP_Q),
                                    self.move_tcp(arm, start, TOP_Q), self.move_tcp(arm, start + v * (float(dist_cm) / 100 + 0.02), TOP_Q)]})
        self.run_seq({arm: [self.move_tcp(arm, self.tcp(arm) + [0, 0, 0.08])]})
        o1 = self.see(obj)
        out = {"code": codes[arm]}
        if o1 is not None:
            out["moved_cm"] = round(float((np.asarray(o1["center"]) - np.asarray(o["center"])) @ v), 1)
        return {arm: out}

    def articulate(self, arm, obj, motion="open", degrees=80):
        self._forget_release(obj)
        """open / close a hinged front door: handle and hinge from the point cloud (geo_v2.door_geometry), grasp the
        handle, then move the gripper along the door arc about the vertical hinge axis in small steps, turning the
        wrist with the door; stops when the arm cannot follow. No joint annotations are used."""
        o = self.see(obj)
        if o is None:
            return {arm: {"code": "NOT_SEEN"}}
        sz = np.asarray(o["size"], float)
        if sz[2] < 12.0 and min(sz[:2]) >= 9.0:          # low flat object -> lid on a horizontal hinge
            return self._articulate_lid(arm, obj, o, motion, degrees)
        d = G.door_geometry(o["pts"], self.tz)
        hx, hy, hz = d["handle"]
        hinge = d["hinge"]
        good = "left" if hinge[0] < hx else "right"       # the door swings toward its hinge side
        if arm not in ARMS:
            arm = good                                     # arm="auto": the hinge-side arm follows the arc furthest
        other = "left" if arm == "right" else "right"
        if self.holding.get(other) is None:
            self.call("home", {"arm": other})
        r0 = np.r_[hx, hy] - hinge
        rad = float(np.linalg.norm(r0))
        sign = -1.0 if (hinge[0] < hx) else 1.0           # rotate so the free edge swings toward the robot (-y)
        if str(motion) == "close":
            sign = -sign
        b = np.radians(60)
        a = np.array([0.0, np.cos(b), -np.sin(b)])        # approach from the front, pitched down
        best = None
        for dz in (0.0, -0.02, -0.04):
            p = np.array([hx, hy - 0.005, hz + dz])
            for c in ([1.0, 0, 0], [-1.0, 0, 0]):
                q = G.hand_quat(a, c)
                pre = self.A.hand_from_tcp(p - a * 0.06, q)
                hg = self.A.hand_from_tcp(p, q)
                if self.feasible(arm, [pre]) and self.feasible(arm, [hg]):
                    best = (p, q, pre, hg)
                    break
            if best:
                break
        if best is None:
            return {arm: {"code": "UNREACHABLE", "why": f"cannot reach the door handle of {obj} with the {arm} arm; try the other arm"}}
        p, q, pre, hg = best
        self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}, {"type": "move", "pose": hg.tolist()},
                            {"type": "grip", "val": 0.0}]})
        steps, done = int(degrees // 6), 0
        for k in range(1, steps + 1):
            th = sign * np.radians(6 * k)
            Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            xy = hinge + Rz @ r0
            qk = G.quat_mul(G.yaw_quat(th), q)
            c = self.run_seq({arm: [self.move_tcp(arm, np.r_[xy, p[2]], qk)]})
            if c[arm] == "PLAN_FAIL":
                break
            done = 6 * k
        self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        t = self.tcp(arm)
        self.run_seq({arm: [self.move_tcp(arm, t + np.r_[0, -0.06, 0.04])]})
        out = {"code": "PARTIAL" if float(degrees) - done > 6.0 else ("EXECUTED" if done >= 50 else "SHORT"), "swing_deg": done, "requested_deg": float(degrees), "handle_found": d["found"],
               "note": "door angle is not measured; judge it from the images"}
        if arm != good:
            out["hint"] = f"the door swings toward the {good} side; the {good} arm can follow it further"
        return {arm: out}

    def _articulate_lid(self, arm, obj, o, motion, degrees):
        """lift (or lower) a lid about a horizontal hinge. Free edge = footprint side where the top points are
        highest (a closed lid: the side facing the robot); hinge = opposite footprint edge at base height. The gripper
        pinches the free edge from outside (fingers closing vertically), follows the arc about the hinge in 6 degree
        steps turning the wrist with the lid, and keeps holding the lid at the end."""
        from transforms3d.axangles import axangle2mat
        from transforms3d.quaternions import mat2quat
        pts = np.asarray(o["pts"], float)
        c, u, m = G.pca2(pts[:, :2])
        z = pts[:, 2]
        top = pts[z >= np.percentile(z, 90)]
        v = top[:, :2].mean(0) - c
        axes = [u, -u, m, -m]
        d = max(axes, key=lambda a: float(a @ v)) if np.linalg.norm(v) > 0.01 else max(axes, key=lambda a: -float(a[1]))
        pr = (pts[:, :2] - c) @ d
        e_front, e_back = float(np.percentile(pr, 98)), float(np.percentile(pr, 2))
        front = pts[pr > e_front - 0.02]
        back = pts[pr < e_back + 0.03]
        ez = float(np.percentile(front[:, 2], 90))
        hz = float(np.percentile(back[:, 2], 50))
        hinge = np.r_[c + d * e_back, hz]
        edge = np.r_[c + d * (e_front - 0.012), ez - 0.006]
        d3 = np.r_[d, 0.0]
        k = G.unit(np.cross(d3, [0, 0, 1.0]))            # rotating about k lifts the free edge
        if str(motion) == "close":
            k = -k
        order = [arm] if arm in ARMS else sorted(ARMS, key=lambda a: float(np.linalg.norm(edge[:2] - G.ARM_BASE[a])))
        best = None
        for a_ in order:
            for b in (20.0, 35.0, 5.0, 50.0):
                br = np.radians(b)
                ap = G.unit(np.r_[-d * np.cos(br), -np.sin(br)])       # fingers point into the object, pitched down
                for cz in ([0, 0, 1.0], [0, 0, -1.0]):
                    cc = np.asarray(cz) - ap * float(np.asarray(cz) @ ap)
                    q = G.hand_quat(ap, cc)
                    pre, hg = self.A.hand_from_tcp(edge - ap * 0.06, q), self.A.hand_from_tcp(edge, q)
                    if self.feasible(a_, [pre]) and self.feasible(a_, [hg]):
                        best = (a_, q, pre, hg)
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            return {(arm if arm in ARMS else "left"): {"code": "UNREACHABLE", "why": f"cannot reach the lid edge of {obj}" +
                    (f" with the {arm} arm; try the other arm or arm=\"auto\"" if arm in ARMS else "")}}
        arm, q, pre, hg = best
        other = "left" if arm == "right" else "right"
        if self.holding.get(other) is None:
            self.call("home", {"arm": other})
        self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}, {"type": "move", "pose": hg.tolist()},
                            {"type": "grip", "val": 0.0}]})
        if self.A.opening(arm) < GAP_MIN:
            self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}]})
            return {arm: {"code": "EMPTY_GRASP", "why": "the fingers closed on nothing at the lid edge", "handle_found": False}}
        r0 = edge - hinge
        steps, done = int(float(degrees) // 6), 0
        for i in range(1, steps + 1):
            Rk = axangle2mat(k, np.radians(6 * i))
            pk = hinge + Rk @ r0
            qk = G.quat_mul(np.asarray(mat2quat(Rk)), q)
            cc_ = self.run_seq({arm: [self.move_tcp(arm, pk, qk)]})
            if cc_[arm] == "PLAN_FAIL":
                break
            done = 6 * i
        self.holding[arm] = obj                          # keeps holding the lid edge (no release, no retreat)
        out = {"code": "PARTIAL" if float(degrees) - done > 6.0 else ("EXECUTED" if done >= 50 else "SHORT"), "swing_deg": done, "requested_deg": float(degrees), "handle_found": True,
               "note": "lid moved about a horizontal hinge; the gripper keeps holding the lid edge. The angle is not measured; judge it from the images"}
        return {arm: out}

    # ------------------------------------------------------------ dispatcher
    def call(self, name, a):
        arm = a.get("arm")
        if name == "locate":
            objs = a.get("objects") or a.get("object") or self.names
            objs = [objs] if isinstance(objs, str) else list(objs)
            desc = a.get("descriptions") or {}
            if isinstance(desc, dict):
                for n, t in desc.items():
                    if n in self.names and isinstance(t, str) and t.strip() and self.descriptions.get(n) != t.strip():
                        self.descriptions[n] = t.strip()
                        self.perc.pop(n, None)          # re-detect with the new description
                        self.first.pop(n, None)
            return {"objects": self.describe(objs)}
        if name in ("home", "release") and arm == "both":
            out = {}
            for x in ARMS:
                out.update(self.call(name, {**a, "arm": x}))
            return out
        if name in ("grasp", "place", "carry", "move", "release", "home", "press", "push", "pour") and arm not in ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        if name == "grasp":
            return self.grasp(arm, a["object"], a.get("part"))
        if name in ("place", "carry"):
            td = a.get("turn_deg")
            return self.place(arm, a["object"], a.get("ref", "table"), a.get("relation", "on"), a.get("xy"),
                              release=name == "place", height_cm=float(a.get("height_cm", 10)), align=bool(a.get("align", False)),
                              turn_deg=None if td in (None, "", 0) else float(td), held_part=a.get("held_part"), face=a.get("face"), face_now=a.get("face_now"))
        if name == "pour":
            return self.pour(arm, a.get("direction", "left"), float(a.get("degrees", 100)), bool(a.get("shake", True)), bool(a.get("return_upright", False)))
        if name == "handover":
            to = a.get("to_arm") or a.get("arm")
            if to not in ARMS:
                raise ValueError("to_arm must be 'left' or 'right'")
            return self.handover(a["object"], to)
        if name == "lift":
            if arm not in (*ARMS, "both"):
                raise ValueError("arm must be left, right or both")
            return self.lift(arm, a.get("height_cm", 15))
        if name in ("move", "lift", "pour", "home") and arm in ARMS:
            self._forget_held_contents(arm)
        if name == "move":
            t = self.tcp(arm)
            tgt = t + np.array([a.get("dx", 0), a.get("dy", 0), a.get("dz", 0)], float) / 100
            c = self.run_seq({arm: [self.move_tcp(arm, tgt)]})
            if c[arm] == "PLAN_FAIL":                    # the same move in smaller steps
                c[arm] = self._move_steps(arm, tgt)
                if c[arm] == "PLAN_FAIL":
                    return {arm: {"code": "PLAN_FAIL", "moved_cm": r1((self.tcp(arm) - t) * 100)}}
            return {arm: {"code": c[arm]}}
        if name == "release":
            self.open_gently(arm)
            self.retreat(arm, 0.08)
            self.holding.pop(arm, None)
            self.hold.pop(arm, None)
            return {arm: {"code": "OK", "opening": round(self.A.opening(arm), 2)}}
        if name == "home":
            t = self.tcp(arm)
            c = self.run_seq({arm: [self.move_tcp(arm, np.r_[t[:2], max(t[2], self.tz + 0.20)]), {"type": "move", "pose": self.observe_pose(arm).tolist()}]})
            if c[arm] != "OK":
                c = self.run_seq({arm: [{"type": "move", "pose": list(self.A.home_pose(arm))}]})
            return {arm: {"code": c[arm]}}
        if name == "press":
            return self.press(arm, a["object"], a.get("direction"))
        if name == "articulate":
            r = self.articulate(arm, a["object"], a.get("motion", "open"), float(a.get("degrees", 80)))
            return r
        if name == "push":
            return self.push(arm, a["object"], a.get("direction"), a.get("distance_cm", 10))
        if name == "bimanual":
            return self.bimanual(a.get("calls") or [])
        if name in ("verify", "done"):
            checks = a.get("checks", a.get("check", []))
            if not isinstance(checks, list):
                raise ValueError("checks must be a list of [predicate, args...]")
            ok, res = self.evaluate(checks)
            return {"all_pass": ok, "checks": res}
        if name == "give_up":
            return {"code": "GAVE_UP"}
        return {"code": "UNKNOWN_TOOL", "why": f"no tool named {name}"}

    def bimanual(self, calls):
        if len(calls) != 2:
            raise ValueError("bimanual needs exactly two calls")
        cs = [(c["name"], c.get("arguments") or {}) for c in calls]
        arms = [x[1].get("arm") for x in cs]
        if set(arms) != set(ARMS):
            raise ValueError("bimanual needs one call per arm (left and right)")
        names = {x[0] for x in cs}
        if names == {"grasp"} and len({x[1].get("object") for x in cs}) == 2:     # two objects -> two single grasps (each with retries)
            out = {}
            for n, x in cs:
                if self.holding.get(x["arm"]):
                    raise ValueError(f"{x['arm']} already holds {self.holding[x['arm']]}")
            for n, x in cs:
                out.update(self.grasp(x["arm"], x["object"], x.get("part")))
            return out
        if names == {"grasp"}:
            for _, x in cs:
                self._forget_release(x.get("object"))
            plans, out = {}, {}
            for n, x in cs:
                if self.holding.get(x["arm"]):
                    raise ValueError(f"{x['arm']} already holds {self.holding[x['arm']]}")
                shared = len({y.get("object") for _, y in cs}) == 1
                g, err = self.grasp_plan(x["arm"], x["object"], x.get("part"), own_side=shared)
                if g is None:
                    out[x["arm"]] = err
                else:
                    plans[x["arm"]] = g
            if out:
                return {**out, **{a: {"code": "NOT_RUN", "why": "the other arm had no reachable grasp"} for a in plans}}
            codes = self.run_seq({a: [{"type": "grip", "val": g.get("pregrip", 1.0)}, {"type": "move", "pose": g["pre"].tolist()},
                                      {"type": "move", "pose": g["grasp"].tolist()}] for a, g in plans.items()})
            self.run_seq({a: [{"type": "grip", "val": 0.0}] for a in plans})
            self.soften(list(plans))
            for a, g in plans.items():
                g["grasp_actual"] = np.asarray(self.A.hand_pose(a), float)
            hc, ex = self.hold_test({a: g["obj"] for a, g in plans.items()})
            if len({g["obj"] for g in plans.values()}) == 1:     # one object, two hands: the camera rise test is unreliable
                for a in plans:
                    if hc[a] == "SLIPPED" and self.A.opening(a) >= GAP_MIN:
                        hc[a] = "OK"
                        ex[a]["note"] = "held by both grippers (finger opening); visual rise not confirmed"
            for a, g in plans.items():
                if hc[a] == "OK":
                    self.holding[a] = g["obj"]
                    self.set_hold(a, g)
                out[a] = {"code": hc[a], "grasp": g["why"], "grasp_point_cm": g["point_cm"], **ex.get(a, {})}
            return out
        out = {}
        for n, x in cs:                               # other combinations: run one after the other
            out.update(self.call(n, x))
        return out

    # ------------------------------------------------------------ predicates
    def evaluate(self, checks):
        now = self.perceive([c[1] for c in checks if isinstance(c, list) and len(c) > 1 and isinstance(c[1], str) and c[1] not in ARMS])
        out, allok = [], True
        for c in checks:
            if not isinstance(c, (list, tuple)) or not c:
                out.append({"check": c, "ok": False, "why": "malformed"}); allok = False
                continue
            p, args = str(c[0]), list(c[1:])
            try:
                ok, why = self._pred(p, args, now)
            except KeyError as e:
                ok, why = None, f"cannot see {e}"
            except _Unsure as e:
                ok, why = None, str(e)
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"bad check: {type(e).__name__}: {e}"
            out.append({"check": [p, *args], "ok": ok, "why": why})
            allok = allok and ok is True
        return allok, out

    def _pred(self, p, args, now):
        def g(n):
            o = now[n] if n in now else self.see(n)
            if o is None:
                raise KeyError(n)
            ra = getattr(self, "_rel_at", {}).get(n)
            if ra is not None and n not in self.holding.values() and o is not None and not o.get("tracked"):
                d = float(np.linalg.norm(self.world(o["center"])[:2] - ra["xy"]))
                if d > 0.10 and _implausible(o, ra.get("size")):   # no gripper touched it since it was let go
                    raise _Unsure(f"{n} is measured {d * 100:.0f} cm from where it was let go and no gripper touched it since: probably a "
                                  f"mis-detection; judge it from the images")
            return o
        if p in ("in", "on"):
            a, b = g(args[0]), (self._frozen(args[1]) if isinstance(args[1], str) else None) or g(args[1])
            over = bool(np.all(a["center"][:2] > b["lo"][:2] - 2.0) and np.all(a["center"][:2] < b["hi"][:2] + 2.0))
            above = bool(a["lo"][2] >= (b["lo"][2] if p == "in" else b["hi"][2] - 3.0) - 1.0)
            return over and above, f"{args[0]} centre {r1(a['center'][:2])} over {args[1]} box {r1(b['lo'][:2])}-{r1(b['hi'][:2])}: {over}; bottom {a['lo'][2]:.1f} cm, {args[1]} top {b['hi'][2]:.1f} cm"
        if p == "on_table":
            a = g(args[0])
            return float(a["lo"][2]) < 2.5, f"bottom {a['lo'][2]:.1f} cm above the table"
        if p in ("left_of", "right_of", "front_of", "behind"):
            a, b = g(args[0]), g(args[1])
            ax, sgn = {"left_of": (0, -1), "right_of": (0, 1), "front_of": (1, -1), "behind": (1, 1)}[p]
            d = sgn * float(a["center"][ax] - b["center"][ax])
            return d > 2.0, f"{args[0]} is {d:.1f} cm {p.replace('_', ' ')} {args[1]}"
        if p in ("near", "next_to"):
            a, b = g(args[0]), g(args[1])
            d = float(np.linalg.norm(a["center"][:2] - b["center"][:2]))
            return d <= 25, f"centres {d:.0f} cm apart"
        if p in ("lifted", "height"):
            held_by = [x for x, o in self.holding.items() if o == args[0] and x in self.hold]
            need = float(args[1]) if len(args) > 1 else 10.0
            if held_by:                                   # gripper pose + in-hand offset (cameras are occluded by the hand)
                bot = max((self.held_est(x)[1] - self.tz) * 100 for x in held_by)
                return bot >= need, f"held by {held_by}; bottom {bot:.1f} cm above the table (from the gripper pose)"
            a = g(args[0])
            held = args[0] in self.holding.values()
            if p == "height":
                return float(a["lo"][2]) >= need, f"{args[0]} bottom {a['lo'][2]:.1f} cm above the table"
            return held and float(a["lo"][2]) >= need, f"held: {held}; bottom {a['lo'][2]:.1f} cm above the table"
        if p == "holding":
            arm, obj = args[0], args[1]
            return self.holding.get(arm) == obj and self.A.opening(arm) >= GAP_MIN, f"{arm} holds {self.holding.get(arm)}; opening {self.A.opening(arm) * 100:.0f}%"
        if p == "released":
            o = self.A.opening(args[0])
            return o > 0.8, f"{args[0]} gripper opening {o * 100:.0f}%"
        if p == "pressed":
            return args[0] in self.pressed, f"a press reached the top of {args[0]}: {args[0] in self.pressed}"
        if p == "upright":                               # standing (vertical extent at least its footprint's long side, or as first seen)
            a = g(args[0])
            L, W, _ = footprint(a["pts"])
            f0 = self.first.get(args[0])
            hz, h0 = float(a["size"][2]), float(f0["size"][2]) if f0 is not None else 0.0
            ok = hz >= 0.9 * L or (h0 >= 0.9 * max(footprint(f0["pts"])[0], 1.0) and hz >= 0.8 * h0)
            return ok, f"{args[0]} height {hz:.1f} cm, footprint {L:.1f} x {W:.1f} cm"
        if p in ("at", "at_xy"):                         # object centre near a table point: at(a, x, y, radius_cm=3)
            a = g(args[0])
            x, y = float(args[1]), float(args[2])
            rr = float(args[3]) if len(args) > 3 else 3.0
            d = float(np.hypot(a["center"][0] - x, a["center"][1] - y))
            return d <= rr, f"{args[0]} centre {r1(a['center'][:2])}, {d:.1f} cm from [{x:.0f}, {y:.0f}]"
        if p == "same_row":                              # same front-back position (|dy| small)
            a, b = g(args[0]), g(args[1])
            dy = float(a["center"][1] - b["center"][1])
            return abs(dy) <= 2.5, f"{args[0]} is {dy:+.1f} cm front-back from {args[1]}"
        return False, f"unknown predicate {p}"


# ====================================================================== planner loop

# Inherited tool bodies come from the layers above. Only explicitly gated calls enter the primitive scope.
import planner_base as _BASE
_GateParent = Harness

class Harness(_GateParent):
    @staticmethod
    def _primitive_gate(name, args):
        if name == "present":
            return "present"
        if name == "articulate" and args.get("motion") in ("pull", "push_in", "slide_out", "slide_in"):
            return "drawer"
        if name == "press" and args.get("direction") == "into_face":
            return "panel"
        if name in ("place", "carry"):
            if args.get("relation") in ("drop", "over"):
                return "opening"
            if args.get("low_end") or args.get("refine_orientation") is True:
                return "orientation"
            if args.get("face"):                         # face estimate + post-release check / one correction only
                return "face"
        if name == "bimanual":
            calls = args.get("calls") or []
            if len(calls) == 2 and all(c.get("name") == "present" for c in calls):
                return "present_both"
        return None

    def call(self, name, a):
        if name == "handover" and "mode" in a:               # mode="air" removed; handover runs the default
            a = {k: v for k, v in a.items() if k != "mode"}
        gate = self._primitive_gate(name, a)
        if gate is None:
            return _GateParent.call(self, name, a)
        self.trace.append({"gate": gate, "tool": name})
        if gate == "face":                               # orientation-only scope: place/carry and _face_turn
            prev_o = getattr(self, "_orient_scope", False)  # lowering, targets, perception and checks stay
            self._orient_scope = True
            try:
                return _BASE.Harness.call(self, name, a)
            finally:
                self._orient_scope = prev_o
        previous = getattr(self, "_primitive_scope", False)
        self._primitive_scope = True
        try:
            clean = dict(a)
            clean.pop("refine_orientation", None)
            return _BASE.Harness.call(self, name, clean)
        finally:
            self._primitive_scope = previous

    def grasp(self, *args, confirm=False, **kwargs):
        # No completion guard, no second-pass grasp and no NOT_SEEN arm motion.
        return _GateParent.grasp(self, *args, **kwargs)

    def _guard(self, obj, action, confirm):
        return None

    def press(self, arm, obj, direction=None, confirm=False):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness._press(self, arm, obj, direction)
        return _GateParent.press(self, arm, obj, direction)

    def _check_placed(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness._check_placed(self, *args, **kwargs)
        return _GateParent._check_placed(self, *args, **kwargs)

    def _face_turn(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False) or getattr(self, "_orient_scope", False):
            return _BASE.Harness._face_turn(self, *args, **kwargs)
        return _GateParent._face_turn(self, *args, **kwargs)

    def _place_once(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness._place_once(self, *args, **kwargs)
        return _GateParent._place_once(self, *args, **kwargs)

    def articulate(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness.articulate(self, *args, **kwargs)
        return _GateParent.articulate(self, *args, **kwargs)

    def bimanual(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness.bimanual(self, *args, **kwargs)
        return _GateParent.bimanual(self, *args, **kwargs)

    def handover(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness.handover(self, *args, **kwargs)
        return _GateParent.handover(self, *args, **kwargs)

    def handover_air(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness.handover_air(self, *args, **kwargs)
        return _GateParent.handover_air(self, *args, **kwargs)

    def perceive(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness.perceive(self, *args, **kwargs)
        return _GateParent.perceive(self, *args, **kwargs)

    def place(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False) or getattr(self, "_orient_scope", False):
            return _BASE.Harness.place(self, *args, **kwargs)
        return _GateParent.place(self, *args, **kwargs)

    def place_target(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness.place_target(self, *args, **kwargs)
        return _GateParent.place_target(self, *args, **kwargs)

    def push(self, *args, **kwargs):
        if getattr(self, "_primitive_scope", False):
            return _BASE.Harness.push(self, *args, **kwargs)
        return _GateParent.push(self, *args, **kwargs)

    _see_clear = _BASE.Harness.__dict__['_see_clear']

    _ledger = _BASE.Harness.__dict__['_ledger']

    _support = _BASE.Harness.__dict__['_support']

    _refute = _BASE.Harness.__dict__['_refute']

    ledger_text = _BASE.Harness.__dict__['ledger_text']

    _see_below = _BASE.Harness.__dict__['_see_below']

    _below_table_blob = _BASE.Harness.__dict__['_below_table_blob']

    _opening_target = _BASE.Harness.__dict__['_opening_target']

    _mar = _BASE.Harness.__dict__['_mar']

    _orient = _BASE.Harness.__dict__['_orient']

    _yaw_error = _BASE.Harness.__dict__['_yaw_error']

    _place_v = _BASE.Harness.__dict__['_place_v']

    _present_plan = _BASE.Harness.__dict__['_present_plan']

    present = _BASE.Harness.__dict__['present']

    _panel = _BASE.Harness.__dict__['_panel']

    _press = _BASE.Harness.__dict__['_press']

    _articulate_slide = _BASE.Harness.__dict__['_articulate_slide']

    DIRS = _BASE.Harness.DIRS
    YAW_TOL = _BASE.Harness.YAW_TOL

    FIX_ASPECT = _BASE.Harness.FIX_ASPECT

SYSTEM += """
Optional explicit primitives (ordinary calls keep their original behavior):
- present(arm, xy=[x,y], height_cm=15, turn_deg=null): keep a held object upright above a table point; two present calls can be bimanual.
- place/carry with relation="drop"/"over" locates a container opening, including below the tabletop beside its edge; drop releases, over keeps holding.
- articulate motion="pull"/"push_in" slides a drawer straight out/in (degrees is distance in cm, default 15), reporting its inside point.
- press direction="into_face" finds the protruding half of a thin upright panel and pushes along its observed normal.
- place/carry low_end="left"/"right"/"front"/"back" uses the height asymmetry of an elongated object to turn its lower end that way.
- place/carry refine_orientation=true explicitly requests height-slice orientation estimation and, for place with face or low_end, a post-release measurement and at most one correction. Use this only when a requested orientation remains visibly wrong after an ordinary placement.
"""


SYSTEM += """- Some results carry extra measured fields: instance tells which detection a grasp used; press may press once more after PARTIAL (second_press); articulate reports measured_swing_deg, continued_deg and swing_check; place with turn_deg reports turn_check; verified_earlier warns that an object already placed and verified is being moved again.
"""


SYSTEM += """- push works on low or lying objects only: a tall standing object is refused (NOT_PUSHABLE, nothing moves); it checks first that the gripper can reach behind the object (UNREACHABLE: nothing moved), stops after a short first contact if the object did not move (NO_CONTACT) or tipped over (TOPPLED), and reports the measured moved_cm / side_cm.
"""


def b64(a):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(a)).convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse(text):
    think = ""
    m = re.search(r"<think>(.*?)</think>", text or "", re.S)
    if m:
        think = m.group(1).strip()
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"<tool_call>(.*?)</tool_call>", t, re.S)
    body = m.group(1) if m else t[t.find("{"): t.rfind("}") + 1]
    v, _ = json.JSONDecoder().raw_decode(body.strip())
    return think, v["name"], v.get("arguments", {}) or {}


def jdump(x):
    def d(v):
        if isinstance(v, np.ndarray):
            return [round(float(t), 3) for t in v.ravel()][:12]
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        return str(v)
    return json.dumps(x, default=d)


def strip_pts(x):
    if isinstance(x, dict):
        return {k: strip_pts(v) for k, v in x.items() if k != "pts"}
    if isinstance(x, list):
        return [strip_pts(v) for v in x]
    return x


PRIVILEGED = ("grasp_plan", "place_plan", "obj_pose", "aabb", "joints", "is_articulated", "table_z", "held", "centers",
              "objects", "save_state", "restore_state", "snapshot")


def _seal_privileged(A):
    """the adapter's simulator-state interfaces must never be used by the tools (raise on use)."""
    for m in PRIVILEGED:
        def boom(*a, _m=m, **k):
            raise RuntimeError(f"privileged simulator interface {_m} is disabled in this harness")
        setattr(A, m, boom)


def table_z_from_depth(sim):
    """table height = the dominant horizontal plane in the head camera's depth (as adapter_libero does)."""
    v = next(x for x in PC.capture(sim.rd) if x["name"] == "head_camera")
    p = v["xyz"][v["valid"]]
    p = p[(np.abs(p[:, 0]) < 0.45) & (p[:, 1] > -0.35) & (p[:, 1] < 0.35) & (p[:, 2] > 0.55) & (p[:, 2] < 0.95)]
    zs = np.round(p[:, 2] / 0.002) * 0.002
    vals, cnt = np.unique(zs, return_counts=True)
    z0 = float(vals[np.argmax(cnt)])
    near = p[np.abs(p[:, 2] - z0) < 0.003, 2]
    return float(np.median(near)) if len(near) else z0


def cam_rename_thin(sim):
    """RoboTwin calls a flat pad a 'block'; rename *_block -> *_pad when the CAMERA measures it under 2 cm tall."""
    blocks = [n for n in sim.names if n.endswith("_block")]
    if not blocks:
        return {}
    tz = table_z_from_depth(sim)
    try:
        res = PC.perceive(sim.rd, blocks, tz)
    except Exception:  # noqa: BLE001
        return {}
    ren = {n: n[:-len("_block")] + "_pad" for n in blocks if n in res and float(res[n]["size"][2]) < 2.0}
    for old, new in ren.items():
        sim.names[new] = sim.names.pop(old)
        sim.attr_of[new] = sim.attr_of.pop(old)
        for k, v in list(sim.entity2name.items()):
            if v == old:
                sim.entity2name[k] = new
    return ren


def load_memory(task):
    """the per-task note of this version's frozen note set, or None (HARNESS_MEMORY=0 or MEMSET=None turns it off)."""
    import os
    if MEMSET is None or os.environ.get("HARNESS_MEMORY", "1") == "0":
        return None
    f = MEMDIR / MEMSET / f"{task}.json"
    if not (MEMDIR / MEMSET).is_dir():
        return None
    err = "missing"
    for _ in range(5):        # every RoboTwin-50 task has a note and it must be used; never drop it silently (on a network FS
        try:                  # an episode once ran without its note right after the checkout was copied; cause not reproduced)
            if not f.exists():
                raise FileNotFoundError(str(f))
            d = json.loads(f.read_text(encoding="utf-8"))
            t, g = str(d.get("text") or "").strip(), str(d.get("goal") or "").strip()
            if t:
                return {"text": t, "goal": g}
            err = "empty text"
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        time.sleep(2.0)
    if err == "missing" or err.startswith("FileNotFoundError"):
        print(f"[harness] WARNING: no task note {f}; running without one", file=sys.stderr, flush=True)
        return None
    raise RuntimeError(f"task note {f} cannot be read ({err}); HARNESS_MEMORY=0 disables notes (ablation only)")


def make_sim(task, episode):
    M.VP.A = argparse.Namespace(guard=False, final_check=True, task_checker="v2", wall=720)
    sim = M.SimEnv(task, seed_dir=str(SEEDS), task_config="demo_randomized", seed_base_index=0)
    info = sim.rd.reset_episode(episode)
    sim.T = sim.rd.env
    sim.T.eval_success = False
    sim.T.plan_success = True
    sim._fingers()
    sim._build_names()
    cam_rename_thin(sim)
    return sim, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--cap", type=int, default=None)        # overrides HARNESS_CAP
    ap.add_argument("--wall", type=int, default=None)       # overrides HARNESS_WALL (s)
    a = ap.parse_args()
    global CAP, WALL
    CAP = a.cap or CAP
    WALL = a.wall or WALL
    root = P / a.lane / a.task / f"shard_{a.episode}"
    od = root / f"episode_{a.episode:03d}"
    (od / "img").mkdir(parents=True, exist_ok=True)
    sim, info = make_sim(a.task, a.episode)
    instr = getattr(info, "instruction", None) or a.task.replace("_", " ")
    import os
    no_list = os.environ.get("HARNESS_NO_OBJECT_LIST") == "1"      # switch: no object list for the planner
    names = [] if no_list else list(sim.names)
    H = Harness(sim, names)
    H.start()
    memo = load_memory(a.task)                                     # per-task note lookup
    msgs = [{"role": "system", "content": SYSTEM}]
    steps, end, rejects, ever, t0, result_text = [], "action_limit", 0, False, time.time(), None
    extra = {"chat_template_kwargs": {"enable_thinking": bool(a.thinking)}}
    response_meta = None
    while len(steps) < CAP and time.time() - t0 < WALL:
        k = len(steps)
        obs = sim.rd.observe()["images"]
        content = []
        olist = (", ".join(names) if names else "(no list is given: refer to objects by short visual names of your own, e.g. \"red block\" "
                 "or \"blue pad\"; the tools find them with the cameras)")
        txt = (f"Task: {instr}\n" + (f"Task note (from earlier practice on this task):\n{memo['text']}\n" if memo else "") + f"Objects in the scene: {olist}\n"
               if k == 0 else f"Result of {steps[-1]['call']['name']}: {result_text}\n" + (f"Task goal (from the note): {memo['goal']}\n" if memo and memo["goal"] else ""))
        txt += H.arm_text() + f"\nCalls used: {k}/{CAP}."
        for key, nm in (("head_camera", "head camera"), ("left_camera", "left wrist camera"), ("right_camera", "right wrist camera")):
            if key in obs:
                content += [{"type": "text", "text": nm + ":"}, {"type": "image_url", "image_url": {"url": b64(obs[key])}}]
                if key == "head_camera":
                    from PIL import Image
                    Image.fromarray(np.asarray(obs[key])).save(od / "img" / f"{k:03d}_head.jpg", quality=85)
        content.append({"type": "text", "text": txt})
        msgs.append({"role": "user", "content": content})
        req, seen = [], 0
        for m in reversed(msgs):
            if isinstance(m["content"], list):
                seen += 1
                if seen > KEEP_IMG_TURNS:
                    m = {"role": m["role"], "content": [c for c in m["content"] if c["type"] == "text"]}
            req.append(m)
        req.reverse()
        text, err = None, None
        for att in range(4):
            try:
                body = {"model": a.model, "messages": req, "temperature": 0.0 if att == 0 else 0.6, "max_tokens": a.max_tokens, **extra}
                record_request(od / "model_requests.jsonl", body, k, att)
                rq = urllib_request(a.url, body)
                completion = json.load(rq)
                choice = completion["choices"][0]
                message = choice["message"]
                text = message.get("content") or ""
                think, name, args = parse(text)
                think = think or message.get("reasoning") or message.get("reasoning_content") or ""
                response_meta = {"finish_reason": choice.get("finish_reason"), "usage": completion.get("usage")}
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
                text = None
                time.sleep(3 + 10 * att)
        if text is None:
            end = "invalid_model_output"
            steps.append({"k": k, "error": err})
            break
        msgs.append({"role": "assistant", "content": text})
        H.trace = []
        try:
            res = H.call(name, args)
        except Exception as e:  # noqa: BLE001
            res = {"code": "ERROR", "why": f"{type(e).__name__}: {e}"[:200]}
            H.A.T.plan_success = True
        res = strip_pts(res)
        ok_now = H.A.success()
        ever = ever or ok_now or H.ever
        steps.append({"k": k, "think": think, "call": {"name": name, "arguments": args}, "result": res, "trace": H.trace,
                      "success_after": ok_now, "response_meta": response_meta})
        result_text = jdump(res)
        if name == "done":
            if res.get("all_pass") or rejects >= MAX_REJECT:
                end = "model_done" if res.get("all_pass") else "done_after_rejections"
                break
            rejects += 1
            cks = res.get("checks") or []
            if cks and all(c.get("ok") is not False for c in cks):          # only unmeasurable (ok=null) checks failed
                result_text += (" -> done REJECTED: some checks could not be measured (ok=null), nothing measured false; judge those from the "
                                "images and call done without them if the images show they hold.")
            else:
                result_text += " -> done REJECTED: the checks did not all pass; the task is not complete yet."
        if name == "give_up":
            end = "give_up"
            break
    if end == "action_limit" and time.time() - t0 >= WALL:
        end = "wall_timeout"
    final_ok = H.A.success()
    done = end in ("model_done", "done_after_rejections")
    ep = {"episode_index": a.episode, "seed": getattr(info, "seed", None), "instruction": instr, "seconds": round(time.time() - t0, 1),
          "model_calls": len(steps), "finished_reason": end, "success": bool(done and final_ok), "oracle_success": bool(ever or final_ok),
          "final_sim_success": final_ok, "agent_terminated": done, "rejects": rejects, "error": steps[-1].get("error") if steps else None,
          "memory": {"set": MEMSET, "used": memo is not None, "sha": None if memo is None else __import__("hashlib").sha256(json.dumps(memo).encode()).hexdigest()[:16]}}
    (od / "episode.json").write_text(jdump({**ep, "steps": steps}))
    (root / "results.json").write_text(json.dumps({"framework": a.lane.split("__")[0], "model": a.model, "task": a.task, "episodes": [ep]}, default=str))
    print(json.dumps(ep), flush=True)
    try:
        sim.rd.close()
    except Exception:  # noqa: BLE001
        pass


def urllib_request(url, body):
    import urllib.request
    rq = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    return urllib.request.urlopen(rq, timeout=600)


# grasp-failure loop + honest completion and success protection.
# Part "grasp": a grasp locks onto the instance the planner last saw located
#   when a fresh detection jumps away from it and the parent's grasp point is far from that located instance, the tool
#   re-detects once around the located position before planning (no new motion, no simulator truth).
# Part "honest": measured results instead of commanded ones, at most one in-tool completion step:
#   press PARTIAL -> one deeper press along the same direction; articulate PARTIAL -> one continuation in 3 deg steps and the
#   swing measured from the hand's actual path; place with turn_deg -> the object's long axis measured after the release
#   (from the re-measurement the parent already makes) and its deviation from the intended turn.
#   A place measured OK whose object then passed a verify is remembered; a later call that moves that object gets a note
#   (never blocked).
# Default paths are the parent's; every addition runs only after the parent's own result says it applies.
_LockParent = Harness
LOCK_PARTS = {"grasp", "honest"}
LOCK_RADIUS_CM = 5.0


class _LockMoved(Exception):
    def __init__(self, err):
        super().__init__(err.get("code"))
        self.err = err


def _lk_ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


class Harness(_LockParent):
    # ------------------------------------------------------------ bookkeeping (both parts)
    def call(self, name, a):
        top = not getattr(self, "_lk_in", False)          # tools that call other tools (e.g. home) are not counted
        if not top:
            return _LockParent.call(self, name, a)
        self._lk_k = getattr(self, "_lk_k", 0) + 1
        self._lk_in = True
        try:
            r = _LockParent.call(self, name, a)
        finally:
            self._lk_in = False
        if "honest" in LOCK_PARTS:
            try:
                self._lk_after_call(name, a, r)
            except Exception as e:  # noqa: BLE001
                self.trace.append({"lock_error": repr(e)[:200]})
        if "grasp" in LOCK_PARTS and name in ("push", "place", "carry", "handover", "articulate", "press", "pour", "bimanual"):
            for c in ([a] + list(a.get("calls") or [])):
                if isinstance(c, dict):
                    getattr(self, "_lk_loc", {}).pop((c.get("arguments") or c).get("object"), None)
        if "grasp" in LOCK_PARTS and name == "locate" and isinstance(r, dict):
            for n, d in (r.get("objects") or {}).items():
                if isinstance(d, dict) and d.get("visible") and n in self.perc:
                    self.__dict__.setdefault("_lk_loc", {})[n] = np.asarray(self.perc[n]["center"], float).copy()
        return r

    # ------------------------------------------------------------ part grasp: instance lock
    # Reference = where locate last saw the object, only while nothing has touched it since (a motion call naming it or any grasp
    # of it clears the reference). Only the first look of a grasp call is checked: a fresh detection more than LOCK_RADIUS_CM away
    # -> the tool looks once more with the located position as prior; if the object is still not found there, nothing is
    # executed and the result says so. In-tool retries after a failed attempt are the parent's and are never checked.
    def grasp_plan(self, arm, obj, part=None, exclude=(), own_side=False):
        ref = getattr(self, "_lk_ref", None) if getattr(self, "_lk_grasp", None) == obj else None
        self._lk_lock = (obj, ref) if ref is not None else None
        try:
            g, err = _LockParent.grasp_plan(self, arm, obj, part, exclude, own_side)
        except _LockMoved as m:
            log = getattr(self, "_lk_obs", None)                # A reads the latest observation of this grasp call
            if log is not None:
                log.append({"seen": True, "o": self.perc.get(obj)})
            return None, m.err
        finally:
            self._lk_lock = None
        if getattr(self, "_lk_grasp", None) == obj:        # fix 22:00: only the first look of a grasp call is checked against
            self._lk_ref = None                            # the located instance; in-tool retries follow the object (never blocked)
        return g, err

    def see(self, n):
        o = _LockParent.see(self, n)
        lk = getattr(self, "_lk_lock", None)
        if lk is None or lk[0] != n:
            return o
        self._lk_lock = None                               # only the grasp planner's own first look
        ref, kind = lk[1]
        if o is None or float(np.linalg.norm(np.asarray(o["center"], float)[:2] - ref[:2])) <= LOCK_RADIUS_CM:
            return o
        rec = {"ref_cm": r1(ref), "ref": kind, "seen_cm": r1(o["center"]), "relocked": False}
        if kind == "located":
            self.perc[n] = dict(o, center=np.asarray(ref, float).copy())   # position prior = the located instance
            o2 = _LockParent.perceive(self, [n]).get(n)
            if o2 is not None and float(np.linalg.norm(np.asarray(o2["center"], float)[:2] - ref[:2])) <= LOCK_RADIUS_CM:
                rec["relocked"] = True
                self.trace.append({"lock_lock": rec})
                self._lk_lock_rec = rec
                return o2
            self.perc[n] = o
        self.trace.append({"lock_lock": rec})
        self._lk_lock_rec = rec
        d = float(np.linalg.norm(np.asarray(o["center"], float)[:2] - ref[:2]))
        raise _LockMoved({"code": "TARGET_MOVED", "seen_cm": r1(o["center"]),
                          "why": (f"{n} is no longer where it was {kind} ({r1(ref)[:2]}): the cameras now find it {d:.0f} cm away at "
                                  f"{r1(o['center'])[:2]}; it was pushed, or this detection is another object. Nothing was executed for this "
                                  f"attempt: check the images, locate {n} (with a visual description if needed), then grasp again")})

    def grasp(self, arm, obj, part=None, *args, **kwargs):
        self._lk_lock_rec = None
        loc = getattr(self, "_lk_loc", {}).get(obj) if "grasp" in LOCK_PARTS else None
        outer = getattr(self, "_lk_grasp", None)
        if outer is None and "grasp" in LOCK_PARTS and obj not in self.holding.values():
            self._lk_grasp, self._lk_ref = obj, ((np.asarray(loc, float).copy(), "located") if loc is not None else None)
        try:
            r = _LockParent.grasp(self, arm, obj, part, *args, **kwargs)
        finally:
            if outer is None:
                self._lk_grasp, self._lk_ref = None, None
        res = r.get(arm) if isinstance(r, dict) else None
        rec = self._lk_lock_rec
        if isinstance(res, dict) and rec is not None and rec["relocked"]:
            res["instance"] = f"grasped the {obj} found near the located position {rec['ref_cm'][:2]} (a detection elsewhere was ignored)"
        if "grasp" in LOCK_PARTS:                            # fix 22:00: any grasp attempt may move the object -> wait for a new locate
            getattr(self, "_lk_loc", {}).pop(obj, None)
        return r

    # ------------------------------------------------------------ part honest: success protection (note only)
    MOVES_OBJ = ("grasp", "push", "place", "carry", "articulate", "press", "pour")

    def _lk_after_call(self, name, a, r):
        k = self._lk_k
        pend = self.__dict__.setdefault("_lk_placed", {})
        ver = self.__dict__.setdefault("_lk_verified", {})
        if name == "place" and isinstance(r, dict):
            out = r.get(a.get("arm")) or {}
            if isinstance(out, dict) and out.get("code") == "OK" and a.get("object"):
                pend[a["object"]] = k
        if name in ("verify", "done") and isinstance(r, dict) and r.get("all_pass") is True:
            named = {x for c in (r.get("checks") or []) for x in (c.get("check") or [])[1:] if isinstance(x, str)}
            for n in list(pend):
                if n in named:
                    ver[n] = k
                    pend.pop(n)
        obj = a.get("object")
        if name in self.MOVES_OBJ and obj in ver and isinstance(r, dict) and ver[obj] < k:
            arm = a.get("arm") if a.get("arm") in ARMS else next((x for x in ARMS if x in r), None)
            out = r.get(arm) if arm else None
            if isinstance(out, dict):
                out["verified_earlier"] = (f"{obj} was placed and verified in place {k - ver[obj]} calls ago; if a check on it fails now, it may "
                                           f"be hidden or mis-measured: look again from another view (move the arms home, locate) before re-grasping "
                                           f"or moving it")
                self.trace.append({"lock_verified_note": obj, "calls_ago": k - ver[obj], "tool": name})
            if name == "grasp" and isinstance(out, dict) and out.get("code") == "OK":
                ver.pop(obj, None)

    # ------------------------------------------------------------ part honest: place turn measured
    def place(self, arm, obj, ref, relation="on", xy=None, release=True, height_cm=10.0, align=False, turn_deg=None, held_part=None,
              face=None, face_now=None, *args, **kwargs):
        want = None
        if ("honest" in LOCK_PARTS and release and not getattr(self, "_primitive_scope", False) and turn_deg is not None and not face and not align
                and self.holding.get(arm) == obj and arm in self.hold and "u" in self.hold[arm]):
            u0 = (Gm.mat(self.A.hand_pose(arm))[:3, :3] @ self.hold[arm]["u"])[:2]
            want = _lk_ang(u0) + float(turn_deg)
        r = _LockParent.place(self, arm, obj, ref, relation, xy, release, height_cm, align, turn_deg, held_part, face, face_now, *args, **kwargs)
        out = r.get(arm) if isinstance(r, dict) else None
        if want is not None and isinstance(out, dict) and out.get("code") in ("OK", "OFF_TARGET") and not self.holding.get(arm) \
                and "orientation_check" not in out and not out.get("regrasped"):
            o3 = self.perc.get(obj)
            if o3 is not None and "pts" in o3 and len(o3["pts"]) >= 60:
                ori = self._orient(o3["pts"])
                if ori is not None and ori["aspect"] >= 1.3:
                    now = _lk_ang(ori["u"])
                    err = (now - want + 90) % 180 - 90
                    out["turn_check"] = {"long_side_now_deg": round((now + 90) % 180 - 90, 1), "off_from_intended_turn_deg": round(err, 1)}
                    if abs(err) > 15:
                        out["turn_check"]["note"] = (f"measured after release: the long side is {abs(err):.0f} deg away from where the requested "
                                                     f"turn should have put it (it turned in the gripper or on release); if the orientation "
                                                     f"matters, check the images and correct it")
                    self.trace.append({"lock_turn_check": out["turn_check"]})
        return r

    # ------------------------------------------------------------ part honest: press, one deeper press after PARTIAL
    def press(self, arm, obj, direction=None, confirm=False):
        r = _LockParent.press(self, arm, obj, direction, confirm)
        out = r.get(arm) if isinstance(r, dict) else None
        if "honest" not in LOCK_PARTS or getattr(self, "_primitive_scope", False) or not isinstance(out, dict) or out.get("code") != "PARTIAL":
            return r
        dname = out.get("direction")
        p = self.world(out["pressed_point_cm"])
        if dname in ("down", "tilted"):
            if dname == "down":
                a, q = np.array([0, 0, -1.0]), TOP_Q
            else:
                b = np.radians(50)
                a = G.unit(np.array([0.0, np.cos(b), -np.sin(b)]))
                q = G.hand_quat(a, [-1.0, 0, 0])
            back = 0.08
        else:
            a = np.asarray(self.PRESS_DIRS[dname], float)
            q = G.hand_quat(a, [0, 0, 1.0])
            back = 0.07
        extra = float(out.get("short_by_cm") or 0) / 100 + 0.01
        hs_, he_ = self.A.hand_from_tcp(p - a * back, q), self.A.hand_from_tcp(p + a * (0.02 + extra), q)
        codes = self.run_seq({arm: [{"type": "grip", "val": 0.0}, {"type": "move", "pose": hs_.tolist()}]})
        if codes[arm] == "PLAN_FAIL":
            out["second_press"] = "not possible (plan failed)"
        else:
            self.A.run({arm: [{"type": "move", "pose": he_.tolist()}]})
            depth = float((self.tcp(arm) - p) @ a)
            self.A.run({arm: [{"type": "move", "pose": hs_.tolist()}]})
            out["second_press"] = {"reach_past_surface_cm": round(depth * 100, 1)}
            out["reach_past_surface_cm"] = round(max(depth, float(out["reach_past_surface_cm"]) / 100) * 100, 1)
            if depth >= 0:
                out["code"] = "OK"
                out.pop("short_by_cm", None)
                out["note"] = ("the first press stopped short; the tool pressed once more, deeper, and reached past the surface. Whether the "
                               "button / switch changed is not measured; judge it from the images (do not press it again if it changed)")
            else:
                out["short_by_cm"] = round(-depth * 100, 1)
                out["note"] = (f"pressed twice (the second time deeper); the fingertip still stopped {-depth * 100:.1f} cm before the surface. "
                               f"Whether it changed is not measured: check the images before pressing again")
        if not self.holding.get(arm):
            self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        self.trace.append({"lock_press2": out.get("second_press")})
        return r

    # ------------------------------------------------------------ part honest: articulate (hinged door / lid), measured swing + one continuation
    def articulate(self, *args, **kwargs):
        if "honest" in LOCK_PARTS and not getattr(self, "_primitive_scope", False):
            return self._lk_articulate(*args, **kwargs)
        return _LockParent.articulate(self, *args, **kwargs)

    def _lk_articulate(self, arm, obj, motion="open", degrees=80):
        self._forget_release(obj)
        """open / close a hinged front door: handle and hinge from the point cloud (geo_v2.door_geometry), grasp the
        handle, then move the gripper along the door arc about the vertical hinge axis in small steps, turning the
        wrist with the door; stops when the arm cannot follow. No joint annotations are used."""
        o = self.see(obj)
        if o is None:
            return {arm: {"code": "NOT_SEEN"}}
        sz = np.asarray(o["size"], float)
        if sz[2] < 12.0 and min(sz[:2]) >= 9.0:          # low flat object -> lid on a horizontal hinge
            return self._lk_articulate_lid(arm, obj, o, motion, degrees)
        d = G.door_geometry(o["pts"], self.tz)
        hx, hy, hz = d["handle"]
        hinge = d["hinge"]
        good = "left" if hinge[0] < hx else "right"       # the door swings toward its hinge side
        if arm not in ARMS:
            arm = good                                     # arm="auto": the hinge-side arm follows the arc furthest
        other = "left" if arm == "right" else "right"
        if self.holding.get(other) is None:
            self.call("home", {"arm": other})
        r0 = np.r_[hx, hy] - hinge
        rad = float(np.linalg.norm(r0))
        sign = -1.0 if (hinge[0] < hx) else 1.0           # rotate so the free edge swings toward the robot (-y)
        if str(motion) == "close":
            sign = -sign
        b = np.radians(60)
        a = np.array([0.0, np.cos(b), -np.sin(b)])        # approach from the front, pitched down
        best = None
        for dz in (0.0, -0.02, -0.04):
            p = np.array([hx, hy - 0.005, hz + dz])
            for c in ([1.0, 0, 0], [-1.0, 0, 0]):
                q = G.hand_quat(a, c)
                pre = self.A.hand_from_tcp(p - a * 0.06, q)
                hg = self.A.hand_from_tcp(p, q)
                if self.feasible(arm, [pre]) and self.feasible(arm, [hg]):
                    best = (p, q, pre, hg)
                    break
            if best:
                break
        if best is None:
            return {arm: {"code": "UNREACHABLE", "why": f"cannot reach the door handle of {obj} with the {arm} arm; try the other arm"}}
        p, q, pre, hg = best
        self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}, {"type": "move", "pose": hg.tolist()},
                            {"type": "grip", "val": 0.0}]})
        t0 = self.tcp(arm)
        steps, done = int(degrees // 6), 0
        for k in range(1, steps + 1):
            th = sign * np.radians(6 * k)
            Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            xy = hinge + Rz @ r0
            qk = G.quat_mul(G.yaw_quat(th), q)
            c = self.run_seq({arm: [self.move_tcp(arm, np.r_[xy, p[2]], qk)]})
            if c[arm] == "PLAN_FAIL":
                break
            done = 6 * k
        cont = None
        if float(degrees) - done > 6.0:                   # PARTIAL -> one continuation in 3 deg steps from where it stopped
            d0, cont = done, 0
            for dd in np.arange(done + 3, float(degrees) + 0.1, 3.0):
                th = sign * np.radians(dd)
                Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                c = self.run_seq({arm: [self.move_tcp(arm, np.r_[hinge + Rz @ r0, p[2]], G.quat_mul(G.yaw_quat(th), q))]})
                if c[arm] == "PLAN_FAIL":
                    break
                done = float(dd)
            cont = round(done - d0)
        v0, v1 = t0[:2] - hinge, self.tcp(arm)[:2] - hinge      # swing measured from the hand's actual position
        meas = abs(np.degrees(np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], float(v0 @ v1))))
        self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        t = self.tcp(arm)
        self.run_seq({arm: [self.move_tcp(arm, t + np.r_[0, -0.06, 0.04])]})
        out = {"code": "PARTIAL" if float(degrees) - done > 6.0 else ("EXECUTED" if done >= 50 else "SHORT"), "swing_deg": round(done),   # fix 22:50: code as parent
               "measured_swing_deg": round(meas), "requested_deg": float(degrees), "handle_found": d["found"],
               "note": "measured_swing_deg is the arc the hand actually followed while holding the handle; if the handle slipped the door moved less: judge it from the images"}
        if cont is not None:
            out["continued_deg"] = cont
        if done - meas > 15.0:                           # fix 22:50: measured arc clearly shorter than commanded -> say so (code unchanged)
            out["swing_check"] = (f"the hand followed only {meas:.0f} of the {done:.0f} deg commanded; the handle may have slipped and "
                                  f"the {'door' if 'door' in out['note'] else 'lid'} moved less: check the images before continuing")
        if arm != good:
            out["hint"] = f"the door swings toward the {good} side; the {good} arm can follow it further"
        return {arm: out}

    def _lk_articulate_lid(self, arm, obj, o, motion, degrees):
        """lift (or lower) a lid about a horizontal hinge. Free edge = footprint side where the top points are
        highest (a closed lid: the side facing the robot); hinge = opposite footprint edge at base height. The gripper
        pinches the free edge from outside (fingers closing vertically), follows the arc about the hinge in 6 degree
        steps turning the wrist with the lid, and keeps holding the lid at the end."""
        from transforms3d.axangles import axangle2mat
        from transforms3d.quaternions import mat2quat
        pts = np.asarray(o["pts"], float)
        c, u, m = G.pca2(pts[:, :2])
        z = pts[:, 2]
        top = pts[z >= np.percentile(z, 90)]
        v = top[:, :2].mean(0) - c
        axes = [u, -u, m, -m]
        d = max(axes, key=lambda a: float(a @ v)) if np.linalg.norm(v) > 0.01 else max(axes, key=lambda a: -float(a[1]))
        pr = (pts[:, :2] - c) @ d
        e_front, e_back = float(np.percentile(pr, 98)), float(np.percentile(pr, 2))
        front = pts[pr > e_front - 0.02]
        back = pts[pr < e_back + 0.03]
        ez = float(np.percentile(front[:, 2], 90))
        hz = float(np.percentile(back[:, 2], 50))
        hinge = np.r_[c + d * e_back, hz]
        edge = np.r_[c + d * (e_front - 0.012), ez - 0.006]
        d3 = np.r_[d, 0.0]
        k = G.unit(np.cross(d3, [0, 0, 1.0]))            # rotating about k lifts the free edge
        if str(motion) == "close":
            k = -k
        order = [arm] if arm in ARMS else sorted(ARMS, key=lambda a: float(np.linalg.norm(edge[:2] - G.ARM_BASE[a])))
        best = None
        for a_ in order:
            for b in (20.0, 35.0, 5.0, 50.0):
                br = np.radians(b)
                ap = G.unit(np.r_[-d * np.cos(br), -np.sin(br)])       # fingers point into the object, pitched down
                for cz in ([0, 0, 1.0], [0, 0, -1.0]):
                    cc = np.asarray(cz) - ap * float(np.asarray(cz) @ ap)
                    q = G.hand_quat(ap, cc)
                    pre, hg = self.A.hand_from_tcp(edge - ap * 0.06, q), self.A.hand_from_tcp(edge, q)
                    if self.feasible(a_, [pre]) and self.feasible(a_, [hg]):
                        best = (a_, q, pre, hg)
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            return {(arm if arm in ARMS else "left"): {"code": "UNREACHABLE", "why": f"cannot reach the lid edge of {obj}" +
                    (f" with the {arm} arm; try the other arm or arm=\"auto\"" if arm in ARMS else "")}}
        arm, q, pre, hg = best
        other = "left" if arm == "right" else "right"
        if self.holding.get(other) is None:
            self.call("home", {"arm": other})
        self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}, {"type": "move", "pose": hg.tolist()},
                            {"type": "grip", "val": 0.0}]})
        if self.A.opening(arm) < GAP_MIN:
            self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}]})
            return {arm: {"code": "EMPTY_GRASP", "why": "the fingers closed on nothing at the lid edge", "handle_found": False}}
        r0 = edge - hinge
        t0 = self.tcp(arm)
        steps, done = int(float(degrees) // 6), 0
        for i in range(1, steps + 1):
            Rk = axangle2mat(k, np.radians(6 * i))
            pk = hinge + Rk @ r0
            qk = G.quat_mul(np.asarray(mat2quat(Rk)), q)
            cc_ = self.run_seq({arm: [self.move_tcp(arm, pk, qk)]})
            if cc_[arm] == "PLAN_FAIL":
                break
            done = 6 * i
        cont = None
        if float(degrees) - done > 6.0:                   # PARTIAL -> one continuation in 3 deg steps from where it stopped
            d0, cont = done, 0
            for dd in np.arange(done + 3, float(degrees) + 0.1, 3.0):
                Rk = axangle2mat(k, np.radians(dd))
                cc_ = self.run_seq({arm: [self.move_tcp(arm, hinge + Rk @ r0, G.quat_mul(np.asarray(mat2quat(Rk)), q))]})
                if cc_[arm] == "PLAN_FAIL":
                    break
                done = float(dd)
            cont = round(done - d0)
        v0, v1 = t0 - hinge, self.tcp(arm) - hinge               # swing measured from the hand's actual position, about the hinge axis
        v0, v1 = v0 - k * float(v0 @ k), v1 - k * float(v1 @ k)
        meas = abs(np.degrees(np.arctan2(float(np.cross(v0, v1) @ k), float(v0 @ v1))))
        self.holding[arm] = obj                          # keeps holding the lid edge (no release, no retreat)
        out = {"code": "PARTIAL" if float(degrees) - done > 6.0 else ("EXECUTED" if done >= 50 else "SHORT"), "swing_deg": round(done),   # fix 22:50: code as parent
               "measured_swing_deg": round(meas), "requested_deg": float(degrees), "handle_found": True,
               "note": "lid moved about a horizontal hinge; the gripper keeps holding the lid edge. measured_swing_deg is the arc the hand actually followed; if the edge slipped the lid moved less: judge it from the images"}
        if cont is not None:
            out["continued_deg"] = cont
        if done - meas > 15.0:                           # fix 22:50: measured arc clearly shorter than commanded -> say so (code unchanged)
            out["swing_check"] = (f"the hand followed only {meas:.0f} of the {done:.0f} deg commanded; the handle may have slipped and "
                                  f"the {'door' if 'door' in out['note'] else 'lid'} moved less: check the images before continuing")
        return {arm: out}

# end polarity of a long object from its height profile.
# Only for place/carry with the explicit low_end= parameter; nothing else changes.
# (1) the lower end is found from the object's height profile along its long axis (8 slices, 90th-percentile height,
#     outer 3 slices of each end compared) instead of the centroid of its highest points; below END_POL_MIN_CM the old
#     estimate is used unchanged.
# (2) after a low_end release the object is measured again at rest: if its lower end clearly points away from the requested
#     direction (more than END_FLIP_DEG), it is re-grasped where it lies and put down once more with the measured turn; the
#     result reports lower_end_check (where the lower end points now). No simulator truth, no task names, no checker values.
_EndPolarityParent = Harness
END_POL_MIN_CM = 0.4
END_FLIP_DEG = 120.0
END_FIX_MIN_CM = 1.0       # the re-grasp fix needs a clear height difference at rest, larger than the one the turn used


def _end_word(v):
    v = np.asarray(v, float)
    return ("right" if v[0] > 0 else "left") if abs(v[0]) >= abs(v[1]) else ("back" if v[1] > 0 else "front")


class Harness(_EndPolarityParent):
    def _end_polarity(self, P_):
        """lower end of an elongated object from its points (world, m): returns (unit xy vector toward the lower end,
        height difference of the two end thirds in cm, aspect) or None."""
        P_ = np.asarray(P_, float)
        if len(P_) < 60:
            return None
        z0, z1 = float(np.percentile(P_[:, 2], 2)), float(np.percentile(P_[:, 2], 98))
        base = P_[P_[:, 2] <= z0 + 0.3 * (z1 - z0)]
        u, m, L, W = self._mar((base if len(base) >= 40 else P_)[:, :2])
        if L / max(W, 1e-3) < 1.6:
            return None
        a = (P_[:, :2] - np.median(P_[:, :2], 0)) @ u
        lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
        edges = np.linspace(lo, hi, 9)
        zs = []
        for i in range(8):
            s = (a >= edges[i]) & (a < edges[i + 1])
            zs.append(float(np.percentile(P_[s, 2], 90)) if s.sum() >= 5 else np.nan)
        zs = np.asarray(zs)
        ok = ~np.isnan(zs)
        if ok.sum() < 6:
            return None
        zs = np.interp(np.arange(8), np.arange(8)[ok], zs[ok])
        f = float(np.mean(zs[-3:]) - np.mean(zs[:3]))
        low = -np.asarray(u, float) if f > 0 else np.asarray(u, float)
        return low, round(abs(f) * 100, 2), round(float(L / max(W, 1e-3)), 2)

    def _face_turn(self, arm, obj, face, face_now=None, low_end=None):
        if low_end and (getattr(self, "_primitive_scope", False) or getattr(self, "_orient_scope", False)) and arm in self.hold \
                and "pts_h" in self.hold[arm]:
            d = self.DIRS.get(str(low_end).lower())
            pol = self._end_polarity(self._tracked(arm, obj)["pts"]) if d is not None else None
            if pol is not None and pol[1] >= END_POL_MIN_CM:
                n = pol[0]
                y = (np.degrees(np.arctan2(d[1], d[0])) - np.degrees(np.arctan2(n[1], n[0])) + 180) % 360 - 180
                self.trace.append({"end_low_end_turn": round(float(y), 1), "end_height_diff_cm": pol[1], "aspect": pol[2]})
                self._end_turn_conf = pol[1]
                return float(y)
            self.trace.append({"end_low_end_turn": None, "end_height_diff_cm": None if pol is None else pol[1]})
        return _EndPolarityParent._face_turn(self, arm, obj, face, face_now, low_end) if low_end is not None else \
            _EndPolarityParent._face_turn(self, arm, obj, face, face_now)

    def _end_check(self, obj, low_end):
        d = self.DIRS.get(str(low_end).lower())
        o = self.see(obj)
        if d is None or o is None:
            return None, None
        pol = self._end_polarity(o["pts"])
        if pol is None:
            return None, None
        n = pol[0]
        off = (np.degrees(np.arctan2(d[1], d[0])) - np.degrees(np.arctan2(n[1], n[0])) + 180) % 360 - 180
        rec = {"lower_end_points": _end_word(n) if pol[1] >= END_POL_MIN_CM else "unclear", "requested": str(low_end),
               "off_deg": round(float(off), 1), "end_height_diff_cm": pol[1]}
        return rec, (float(off) if pol[1] >= END_POL_MIN_CM else None)

    def place(self, arm, obj, ref, relation="on", xy=None, release=True, height_cm=10.0, align=False, turn_deg=None, held_part=None,
              face=None, face_now=None, low_end=None, *args, **kwargs):
        if low_end is None:
            return _EndPolarityParent.place(self, arm, obj, ref, relation, xy, release, height_cm, align, turn_deg, held_part, face, face_now, *args, **kwargs)
        self._end_turn_conf = None
        r = _EndPolarityParent.place(self, arm, obj, ref, relation, xy, release, height_cm, align, turn_deg, held_part, face, face_now, low_end, *args, **kwargs)
        out = r.get(arm) if isinstance(r, dict) else None
        if not (release and isinstance(out, dict) and out.get("code") in ("OK", "OFF_TARGET") and not self.holding.get(arm)):
            return r
        try:
            rec, off = self._end_check(obj, low_end)
        except Exception as e:  # noqa: BLE001
            self.trace.append({"end_check_error": repr(e)[:160]})
            return r
        if rec is None:
            return r
        tc = getattr(self, "_end_turn_conf", None)
        if off is not None and abs(off) > END_FLIP_DEG and (rec["end_height_diff_cm"] < END_FIX_MIN_CM or (tc is not None and rec["end_height_diff_cm"] <= tc)):
            rec["note"] = "the two ends are not clearly different in height here; not corrected, check the images"
            off = None
        if off is not None and abs(off) > END_FLIP_DEG and not getattr(self, "_end_fixing", False):
            self._end_fixing = True
            try:
                g = self.grasp(arm, obj)
                if (g.get(arm) or {}).get("code") == "OK":
                    r2 = _EndPolarityParent.place(self, arm, obj, ref, relation, xy, True, height_cm, False, float(off), held_part, None, None)
                    o2 = r2.get(arm) if isinstance(r2, dict) else None
                    if isinstance(o2, dict):
                        rec2, _ = self._end_check(obj, low_end) if not self.holding.get(arm) else (None, None)
                        rec["fix"] = f"re-grasped and turned by {off:.0f} deg"
                        rec["after_fix"] = rec2
                        o2["lower_end_check"] = rec
                        self.trace.append({"end_flip_fix": round(off, 1), "after": rec2})
                        return r2
                    rec["fix"] = "re-place failed"
                else:
                    rec["fix"] = f"re-grasp for the turn failed: {(g.get(arm) or {}).get('code')}"
            finally:
                self._end_fixing = False
        out["lower_end_check"] = rec
        self.trace.append({"end_lower_end_check": rec})
        return r


SYSTEM += """- place with low_end also reports lower_end_check: where the object's lower end points after the release, measured from its height profile ("unclear" when its two ends are about equally high); when it pointed the opposite way the tool re-grasped it and turned it once (fix).
"""


# measured end of a held object after carry with held_part.
# carry(held_part=...) positions the named end from the in-hand points recorded at the grasp; if the object turned or slid in
# the fingers, that end is elsewhere. After such a carry the held object is segmented again by the cameras (fresh points, not
# the tracked ones), the named end is measured, and the result reports held_end_check. Only when the measured end is clearly
# off the target (more than END_END_TOL_CM, less than END_END_MAX_CM) the gripper is moved by the measured difference at the
# same height, at most twice. No simulator truth, no task names, no checker values.
_HeldEndParent = Harness
END_END_TOL_CM = 1.5
END_END_MAX_CM = 10.0


class Harness(_HeldEndParent):
    def _end_fresh_end(self, arm, obj, part):
        """(end xy (m) or None, reason) from a fresh camera segmentation of the held object (not the tracked points)."""
        want = {"left": [-1.0, 0], "right": [1.0, 0], "front": [0, -1.0], "back": [0, 1.0]}.get(str(part).lower().split()[0])
        if want is None:
            return None, "part is not a direction end"
        tr = self.perc.get(obj)
        try:                                              # fresh segmentation (not the tracked points)
            o = PC.perceive(self.sim.rd, [obj], self.tz, priors=self.first, pos_prior=self.pos_prior(),
                            prompt_overrides=dict(self.descriptions), drop_robot=False).get(obj)
        except Exception as e:  # noqa: BLE001
            return None, f"perception error {type(e).__name__}"
        if o is None:
            return None, "not detected"
        P_ = np.asarray(o["pts"], float)
        tcp = self.tcp(arm)
        P_ = P_[(np.linalg.norm(P_[:, :2] - tcp[:2], axis=1) < 0.25) & (P_[:, 2] < tcp[2] + 0.02)]   # at the gripper, not the arm above it
        if len(P_) < 60:
            return None, "too few points near the gripper"
        c, u, _ = G.pca2(P_[:, :2])
        u = np.asarray(u, float)
        if float(u @ np.asarray(want)) < 0:
            u = -u
        t = (P_[:, :2] - c) @ u
        L = G.extent(t)
        reg = P_[t >= np.percentile(t, 97) - min(0.05, 0.22 * L)]
        if len(reg) < 5:
            return None, "end not resolved"
        return np.median(reg[:, :2], 0), None

    def place(self, arm, obj, ref, relation="on", xy=None, release=True, height_cm=10.0, align=False, turn_deg=None, held_part=None,
              face=None, face_now=None, *args, **kwargs):
        r = _HeldEndParent.place(self, arm, obj, ref, relation, xy, release, height_cm, align, turn_deg, held_part, face, face_now, *args, **kwargs)
        out = r.get(arm) if isinstance(r, dict) else None
        if not held_part or release or not isinstance(out, dict) or out.get("code") != "OK" or self.holding.get(arm) != obj \
                or not out.get("target_cm"):
            return r
        try:
            tgt = np.asarray(out["target_cm"], float)[:2] / 100
            rec = {"part": held_part, "moves": []}
            prev = None
            for it in range(3):
                e, why = self._end_fresh_end(arm, obj, held_part)
                if e is None:
                    rec["end_offset_cm"] = None
                    rec["note"] = f"the held object could not be measured again ({why}); check the images"
                    break
                d = tgt - e
                rec["end_offset_cm"] = [round(float(-d[0]) * 100, 1), round(float(-d[1]) * 100, 1)]
                n = float(np.linalg.norm(d))
                if prev is not None and n > 0.5 * prev[0]:      # the measured end did not follow the gripper: not the held object
                    t = self.tcp(arm)
                    self.run_seq({arm: [self.move_tcp(arm, np.r_[t[:2] - prev[1], t[2]])]})
                    rec["moves"].append({"undone": True})
                    rec["note"] = "the measured end did not move with the gripper, so the measurement was not trusted and the move was undone; check the images"
                    rec["end_offset_cm"] = prev[2]
                    break
                if n * 100 <= END_END_TOL_CM or n * 100 > END_END_MAX_CM or it == 2:
                    break
                t = self.tcp(arm)
                c_ = self.run_seq({arm: [self.move_tcp(arm, np.r_[t[:2] + d, t[2]])]})
                rec["moves"].append({"by_cm": [round(float(d[0]) * 100, 1), round(float(d[1]) * 100, 1)], "code": c_[arm]})
                if c_[arm] == "PLAN_FAIL":
                    break
                prev = (n, d, rec["end_offset_cm"])
            if not rec["moves"]:
                rec.pop("moves")
            out["held_end_check"] = rec
            self.trace.append({"end_held_end_check": rec})
        except Exception as ex:  # noqa: BLE001
            self.trace.append({"end_held_end_error": repr(ex)[:160]})
        return r


SYSTEM += """- carry with held_part also reports held_end_check: where the cameras see that end of the held object now relative to the target (end_offset_cm); if it was clearly off, the gripper was moved by the measured difference (moves).
"""


# release without sweeping the fingers; honest blind grasps.
# 1. Release order (place's release and the release tool): the fingers open just past the contact opening (the parent's
#    first two steps: contact opening +0.05, +0.25), the hand backs straight up exactly as the parent's retreat, and only
#    then do the fingers open fully.  The parent opened fully while the fingertips were still down at the object, which
#    swings the fingers sideways into whatever surrounds the released object (rims of nested or neighbouring objects, the
#    container wall) and can throw it. Same waypoints, same final
#    gripper state, same return values; for an object grasped wide open (contact opening >= 0.75) the order is unchanged.
# 2. Grasp honesty: when the lift test could not see the object (code OK judged only by the finger opening), the result
#    says so (held_check) and the in-tool state is unchanged (never blocked).  A later place of that object that cannot
#    find it after release says that the hold had never been confirmed by the cameras.
# Camera / proprioception only; no task names, no simulator truth, no checker thresholds.
_ReleaseParent = Harness
EXR_PART_OPEN = 0.25          # = the parent's second opening step (contact opening + 0.25)


class Harness(_ReleaseParent):
    def open_gently(self, arm):
        if getattr(self, "_rl_rel", None) != arm:
            return _ReleaseParent.open_gently(self, arm)
        o = float(self.A.opening(arm))
        if o + EXR_PART_OPEN >= 1.0:                      # already (almost) fully open after the partial step: parent order
            self._rl_rel = None
            return _ReleaseParent.open_gently(self, arm)
        for v in (min(1.0, o + 0.05), min(1.0, o + EXR_PART_OPEN)):
            self.run_seq({arm: [{"type": "grip", "val": v}]})
        self._rl_pending = arm                           # full opening after the straight back-off (retreat)
        self.trace.append({"release_release": "partial open, back off up, then open fully"})

    def retreat(self, arm, total=0.10):
        r = _ReleaseParent.retreat(self, arm, total)
        if getattr(self, "_rl_pending", None) == arm:
            self._rl_pending = None
            self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        return r

    def _rl_scope(self, arm, fn, *a, **k):
        prev, prevp = getattr(self, "_rl_rel", None), getattr(self, "_rl_pending", None)
        self._rl_rel, self._rl_pending = arm, None
        try:
            return fn(*a, **k)
        finally:
            if getattr(self, "_rl_pending", None) == arm:    # release without the retreat (should not happen): open fully here
                self._rl_pending = None
                self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
            self._rl_rel, self._rl_pending = prev, prevp

    def _place_once(self, arm, obj, ref, relation="on", xy=None, release=True, *args, **kwargs):
        if not release:
            return _ReleaseParent._place_once(self, arm, obj, ref, relation, xy, release, *args, **kwargs)
        r = self._rl_scope(arm, _ReleaseParent._place_once, self, arm, obj, ref, relation, xy, release, *args, **kwargs)
        out = r.get(arm) if isinstance(r, dict) else None
        if isinstance(out, dict) and out.get("code") == "RELEASED_NOT_SEEN" and obj in getattr(self, "_rl_blind", set()):
            out["why"] = (str(out.get("why", "")) + f"; note: during the grasp's lift test the cameras did not see {obj} either (the hold was "
                          f"judged from the finger opening only)")
        if not self.holding.get(arm):
            getattr(self, "_rl_blind", set()).discard(obj)
        return r

    def call(self, name, a):
        if name == "release" and a.get("arm") in ARMS and self.holding.get(a.get("arm")) is not None:
            arm = a["arm"]
            return self._rl_scope(arm, _ReleaseParent.call, self, name, a)
        return _ReleaseParent.call(self, name, a)

    def grasp(self, arm, obj, part=None, *args, **kwargs):
        r = _ReleaseParent.grasp(self, arm, obj, part, *args, **kwargs)
        out = r.get(arm) if isinstance(r, dict) else None
        if isinstance(out, dict) and out.get("code") == "OK" and "not visible" in str(out.get("lift_test", "")):
            self.__dict__.setdefault("_rl_blind", set()).add(obj)
            out["held_check"] = (f"finger opening only: the cameras could not see {obj} during the lift test (a small object is often hidden by "
                                 f"the gripper), so that the held thing is {obj} is not confirmed by the cameras")
            self.trace.append({"release_held_check": obj})
        elif isinstance(out, dict) and out.get("code") == "OK":
            getattr(self, "_rl_blind", set()).discard(obj)
        return r


SYSTEM += """- grasp may report held_check when the cameras could not confirm the grasped object in the hand (code OK only from the finger opening).
"""

# first grasp of tall slender upright objects + push rebuilt.
# Part "axis": an object the cameras see standing tall and slender (the parent's own condition: height >= 10 cm and >= 2x its
#   footprint, no handle) whose footprint fits the open gripper both ways (long side <= 8.5 cm): every side / tilted approach
#   closes the fingers HORIZONTALLY across the object (perpendicular to the approach). The parent derives the closing axis
#   from the footprint's narrow direction, which on a round footprint is arbitrary: with the 75 / 60 deg tilted approaches it
#   then closes partly along the object's height (one finger above, one below) and the closing tips it over; and the level
#   approach is skipped whenever that arbitrary axis happens to point along the reach. Level (90 deg) candidates the parent
#   already had are kept unchanged and in the same order. Sub-part "no60": the steepest (60 deg) approach is not used for such
#   objects (bench: it tipped the object over at closing in 4 of 4 cases, parent and axis alike).
# Part "sweep": before the joint-space move to the pre-grasp pose of a grasp, the planned path is checked (plan only) with
#   forward kinematics of the arm (proprioception) against the camera point cloud: if the arm or hand would pass through
#   the column of a standing object (target or another object) the transit goes via a raised waypoint above the pre-grasp point
#   and then straight down; if that does not plan, the parent's move is used unchanged.
# Part "push": push only on low or lying objects (a tall slender upright object is refused: NOT_PUSHABLE, nothing moves); the
#   gripper pose is chosen among straight-down and tilted closed-gripper poses that all plan before anything moves
#   (UNREACHABLE with the number checked otherwise); low contact height; it pushes a first short step, lifts the gripper off
#   and looks: stops if the object did not move (NO_CONTACT), tipped over (TOPPLED) or is no longer seen (NOT_SEEN); otherwise
#   it goes back to the measured object face and pushes the rest; moved_cm / side_cm are measured by the cameras.
_SlenderParent = Harness
SLENDER_PARTS = {"axis", "no60", "push", "sweep"}
SLENDER_SLIM_CM = 8.5
_SLENDER_DV = {"left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0), "front": (0.0, -1.0, 0.0), "back": (0.0, 1.0, 0.0)}


def _sl_tall(o):
    sz = np.asarray(o["size"], float)
    return bool(sz[2] >= 10.0 and sz[2] >= 2.0 * max(sz[0], sz[1]))


class Harness(_SlenderParent):
    # ------------------------------------------------------------ part axis
    def _sl_slim(self, s):
        return s.get("out") is None and float((s.get("shape") or {}).get("long_cm", 99.0)) <= SLENDER_SLIM_CM

    def _sl_tall_cands(self, s, arm):
        """candidates of a tall object (parent: G.candidates(s, arm, tilts=(90, 75, 60)))."""
        base = G.candidates(s, arm, tilts=(90.0, 75.0, 60.0))
        if "axis" not in SLENDER_PARTS or not self._sl_slim(s):
            return base
        tilts = (90.0, 75.0) if "no60" in SLENDER_PARTS else (90.0, 75.0, 60.0)
        r = G.unit(np.r_[np.asarray(s["p"], float)[:2] - G.ARM_BASE[arm], 0.0])
        out, changed = [], set()
        for al in tilts:
            a_ = np.radians(al)
            a = G.unit(np.r_[r[:2] * np.sin(a_), -np.cos(a_)])
            hz = G.unit(np.cross([0.0, 0.0, 1.0], a))
            mine = [(b_al, b_a, b_c) for b_al, b_a, b_c in base if b_al == al]
            if mine and float(np.asarray(mine[0][2]) @ hz) < 0:
                hz = -hz
            new = [(al, a, hz), (al, a, -hz)]
            if len(mine) == 2 and all(np.allclose(m[1], n[1], atol=1e-6) and np.allclose(m[2], n[2], atol=1e-6) for m, n in zip(mine, new)):
                out += mine                                    # the parent's own candidates, unchanged
            else:
                out += new
                changed.add(al)
        if "no60" in SLENDER_PARTS and any(b[0] == 60.0 for b in base):
            changed.add(60.0)
        self._sl_axis_changed = changed
        self._sl_axis_any = getattr(self, "_sl_axis_any", False) or bool(changed)
        return out

    def _sl_turn_c(self, s, a2, m, tall):
        """closing axis of a turned approach widening (parent expression unless part axis applies)."""
        if tall and "axis" in SLENDER_PARTS and self._sl_slim(s):
            return np.cross([0.0, 0.0, 1.0], a2)
        return m - a2 * float(m @ a2) if s.get("out") is None else np.array([0, 0, 1.0]) - a2 * float(a2[2])

    def grasp_plan(self, arm, obj, part=None, exclude=(), own_side=False):
        self._sl_axis_changed, self._sl_axis_any = set(), False
        g, err = _SlenderParent.grasp_plan(self, arm, obj, part, exclude, own_side)
        if g is not None and self._sl_axis_any:          # the candidate list differed from the parent one (the choice may differ)
            g["exg_axis"] = True
            self.trace.append({"slender_axis": {"tilt": round(float(g["tilt"])), "ext": g.get("ext"), "point_cm": g["point_cm"],
                                            "changed_tilt": float(g.get("tilt", -1)) in self._sl_axis_changed}})
        self._sl_g = g
        return g, err

    # ------------------------------------------------------------ part sweep
    def _sl_fk_points(self, arm, traj):
        """world positions of the arm's links (and midpoints) along a planned joint trajectory (proprioceptive kinematics)."""
        rb = self.A.T.robot
        ent = rb.left_entity if arm == "left" else rb.right_entity
        pl = rb.left_planner if arm == "left" else rb.right_planner
        pm = self.__dict__.setdefault("_sl_pm", {}).get(id(ent))
        if pm is None:
            pm = ent.create_pinocchio_model()
            self._sl_pm[id(ent)] = pm
        idx = [pl.all_joints.index(n) for n in pl.active_joints_name if n in pl.all_joints]
        cur = np.asarray(ent.get_qpos(), float)
        root = ent.get_root_pose() if hasattr(ent, "get_root_pose") else ent.get_pose()
        names = [l.get_name() for l in ent.get_links()]
        pre = "fl_link" if arm == "left" else "fr_link"
        mine = [i for i, n in enumerate(names) if n.startswith(pre)]
        P = []
        for q in traj:
            full = cur.copy()
            full[idx] = np.asarray(q, float)[: len(idx)]
            pm.compute_forward_kinematics(full)
            prev = None
            for i in (mine or range(len(names))):
                p = np.asarray((root * pm.get_link_pose(i)).p, float)
                P.append(p)
                if prev is not None and np.linalg.norm(p - prev) < 0.4:
                    for f in (0.33, 0.67):
                        P.append(prev + f * (p - prev))
                prev = p
        return np.asarray(P)

    def _sl_columns(self, arm):
        """standing columns in the camera point cloud: points above the table (robot points removed), on a 2 cm grid -> top z."""
        views = PC.capture(self.sim.rd)
        views = PC._drop_robot(views, PC._robot_points(self.sim.rd))
        P_ = np.concatenate([v["xyz"][v["valid"]] for v in views])
        P_ = P_[(P_[:, 2] > self.tz + 0.03) & (P_[:, 2] < self.tz + 0.45) & (np.abs(P_[:, 0]) < 0.75) & (P_[:, 1] > -0.30) & (P_[:, 1] < 0.45)]
        return P_

    def _sl_sweep_hits(self, arm, pose):
        """plan (only) the move to `pose`; return (#trajectory points within 2.5 cm of a camera point above the table, trajectory)."""
        try:
            res = self.planner(arm)(list(map(float, pose)))
        except Exception:  # noqa: BLE001
            return None, None
        if not isinstance(res, dict) or res.get("status") != "Success" or res.get("position") is None:
            return None, None
        traj = np.asarray(res["position"], float)
        traj = traj[:: max(1, len(traj) // 25)]
        A = self._sl_fk_points(arm, traj)
        cols = self._sl_columns(arm)
        if not len(cols) or not len(A):
            return 0, traj
        from scipy.spatial import cKDTree
        tree = cKDTree(cols)
        start = self._sl_fk_points(arm, traj[:1])
        near0 = cKDTree(start)
        d0, _ = near0.query(cols, k=1)
        cols = cols[d0 > 0.08]                        # points next to the arm at its start are the arm / what it holds
        if not len(cols):
            return 0, traj
        tree = cKDTree(cols)
        d, _ = tree.query(A, k=1)
        return int((d < 0.025).sum()), traj

    def run_seq(self, seqs):
        g = getattr(self, "_sl_g", None)
        if ("sweep" in SLENDER_PARTS and g is not None and len(seqs) == 1 and g["arm"] in seqs and len(seqs[g["arm"]]) == 4
                and seqs[g["arm"]][1].get("type") == "move" and seqs[g["arm"]][2].get("type") == "move"
                and np.allclose(seqs[g["arm"]][1]["pose"], np.asarray(g["pre"], float)) and np.allclose(seqs[g["arm"]][2]["pose"], np.asarray(g["grasp"], float))):
            self._sl_g = None
            arm = g["arm"]
            if _sl_tall(g["obs"]) or self._sl_tall_near(arm, g):
                try:
                    via = self._sl_via(arm, g)
                except Exception as e:  # noqa: BLE001
                    self.trace.append({"slender_sweep_error": repr(e)[:160]})
                    via = None
                if via is not None:
                    L = list(seqs[arm])
                    return _SlenderParent.run_seq(self, {arm: [L[0]] + [{"type": "move", "pose": v.tolist()} for v in via] + L[1:]})
        return _SlenderParent.run_seq(self, seqs)

    def _sl_tall_near(self, arm, g):
        """another camera-seen standing tall object lies between the hand and the grasp point (xy corridor 12 cm wide)."""
        t = self.tcp(arm)[:2]
        p = np.asarray(g["spec_p"], float)[:2]
        for n, o in self.perc.items():
            if n == g["obj"] or o is None or o.get("tracked") or not _sl_tall(o):
                continue
            c = self.world(o["center"])[:2]
            d = p - t
            u = float(np.clip((c - t) @ d / max(float(d @ d), 1e-9), 0, 1))
            if float(np.linalg.norm(t + u * d - c)) < 0.06 + float(max(o["size"][:2])) / 200:
                return True
        return False

    def _sl_via(self, arm, g):
        """gated: the planned transit to the pre-grasp pose passes through standing points -> raised waypoint(s) above the pre-grasp."""
        hits, _ = self._sl_sweep_hits(arm, g["pre"])
        rec = {"obj": g["obj"], "hits": hits}
        if not hits:
            self.trace.append({"slender_sweep": rec})
            return None
        o = g["obs"]
        top = self.world(o["hi"])[2]
        pre_t = self.A.tcp_point(np.asarray(g["pre"], float))
        q = np.asarray(g["pre"], float)[3:]
        for dz in (0.10, 0.16):
            z = max(top, pre_t[2]) + dz
            up = self.A.hand_from_tcp(np.r_[pre_t[:2], z], q)
            h2, _ = self._sl_sweep_hits(arm, up)
            if h2 is None or h2 > 0:
                continue
            if self.feasible(arm, [up]):
                rec.update(via_cm=self.cm(np.r_[pre_t[:2], z]), via_hits=h2)
                self.trace.append({"slender_sweep": rec})
                mid = self.A.hand_from_tcp(np.r_[pre_t[:2], 0.5 * (z + pre_t[2])], q)
                return [np.asarray(up, float), np.asarray(mid, float)]
        rec["via"] = "none planned clear; parent move"
        self.trace.append({"slender_sweep": rec})
        return None

    # ------------------------------------------------------------ part push
    SLENDER_PUSH_TILTS = (0.0, 30.0, 50.0)

    def push(self, arm, obj, direction, dist_cm):
        if "push" not in SLENDER_PARTS:
            return _SlenderParent.push(self, arm, obj, direction, dist_cm)
        self._forget_release(obj)
        v = _SLENDER_DV.get(str(direction).lower())
        if v is None:
            return {arm: {"code": "BAD_ARG", "why": "direction must be left / right / front / back"}}
        if self.holding.get(arm):
            return {arm: {"code": "BAD_ARG", "why": f"the {arm} arm holds {self.holding[arm]}; place or release it first (push uses the closed empty gripper)"}}
        o = self.see(obj)
        if o is None:
            return {arm: {"code": "NOT_SEEN"}}
        v = np.asarray(v, float)
        dist = float(np.clip(float(dist_cm), 1.0, 25.0)) / 100
        sz = np.asarray(o["size"], float)
        if _sl_tall(o):
            self.trace.append({"slender_push": "not_pushable", "size_cm": r1(sz)})
            return {arm: {"code": "NOT_PUSHABLE", "size_cm": r1(sz), "why": f"{obj} stands {sz[2]:.0f} cm tall on a {max(sz[0], sz[1]):.0f} cm footprint: "
                          "a push would knock it over. Nothing moved. Grasp it instead (the arm on its side, or the arm that reaches and then a hand-over)"}}
        c = self.world(o["center"])
        h = max(float(sz[2]) / 100, 0.01)
        ext = float(np.abs(sz / 200) @ np.abs(v))
        zc = self.tz + float(np.clip(0.4 * h, 0.012, 0.035))
        start = np.r_[c[:2] - v[:2] * (ext + 0.035), zc]
        end = start + v * (0.035 + dist)
        above = start + [0, 0, max(0.08, h + 0.04 - (zc - self.tz))]
        cand, tried = None, 0
        for t in self.SLENDER_PUSH_TILTS:
            heads = [v] if t == 0 else [v, -v]
            for hd in heads:
                a = G.unit(np.r_[hd[:2] * np.sin(np.radians(t)), -np.cos(np.radians(t))])
                for cl in ([np.cross([0, 0, 1.0], v), v] if t == 0 else [np.cross([0, 0, 1.0], hd)]):
                    cl = G.unit(np.asarray(cl, float) - a * float(np.asarray(cl, float) @ a))
                    q = G.hand_quat(a, cl)
                    poses = [self.A.hand_from_tcp(p_, q) for p_ in (above, start, end)]
                    tried += 1
                    if all(self.feasible(arm, [p_]) for p_ in poses):
                        cand = (t, poses)
                        break
                if cand:
                    break
            if cand:
                break
        rec = {"tried": tried, "tilt": None if cand is None else cand[0], "contact_cm": self.cm(start)}
        if cand is None:
            self.trace.append({"slender_push": rec})
            return {arm: {"code": "UNREACHABLE", "poses_checked": tried,
                          "why": f"no closed-gripper pose of the {arm} arm reaches behind {obj} to push it {direction} ({tried} poses checked with the "
                                 "motion planner, nothing moved); use the other arm, another direction, or grasp it"}}
        t, (hp_above, hp_start, hp_end) = cand
        codes = _SlenderParent.run_seq(self, {arm: [{"type": "grip", "val": 0.0}, {"type": "move", "pose": hp_above.tolist()}, {"type": "move", "pose": hp_start.tolist()}]})
        if codes[arm] == "PLAN_FAIL":
            self.run_seq({arm: [self.move_tcp(arm, self.tcp(arm) + [0, 0, 0.08])]})
            self.trace.append({"slender_push": rec})
            return {arm: {"code": "PLAN_FAIL", "why": f"could not bring the closed gripper behind {obj}; nothing was pushed"}}
        first = min(0.03, dist)
        q = self.A.hand_pose(arm)[3:]
        p1 = start + v * (0.035 + first)
        _SlenderParent.run_seq(self, {arm: [self.move_tcp(arm, p1, q)]})
        lift = max(0.06, h + 0.06 - (zc - self.tz))
        _SlenderParent.run_seq(self, {arm: [self.move_tcp(arm, p1 + [0, 0, lift], q)]})     # look with the gripper lifted above the object (it occludes it)
        o1 = self.see(obj)
        m1 = None if o1 is None else float((np.asarray(o1["center"]) - np.asarray(o["center"]))[:2] @ v[:2])
        stop = None
        if o1 is None:
            stop = "NOT_SEEN"
        elif float(sz[2]) >= 8.0 and float(o1["size"][2]) < 0.6 * float(sz[2]):
            stop = "TOPPLED"
        elif m1 < 1.0:
            stop = "NO_CONTACT"
        if stop is None and dist > first:
            back = np.r_[self.world(o1["center"])[:2] - v[:2] * (ext + 0.005), zc]      # fingertips back at the object face as measured now
            if _SlenderParent.run_seq(self, {arm: [self.move_tcp(arm, back + [0, 0, lift], q), self.move_tcp(arm, back, q)]})[arm] == "PLAN_FAIL":
                stop = "PLAN_FAIL"
            else:
                rest = dist - max(m1, 0.0) / 100
                n = max(1, int(np.ceil(rest / 0.04)))
                for k in range(1, n + 1):
                    _SlenderParent.run_seq(self, {arm: [self.move_tcp(arm, back + v * (rest * k / n), q)]})
        self.run_seq({arm: [self.move_tcp(arm, self.tcp(arm) + [0, 0, 0.08])]})
        o2 = self.see(obj)
        out = {"code": stop or "OK", "push_tilt_deg": round(t), "contact_height_cm": round((zc - self.tz) * 100, 1)}
        if o2 is not None:
            d = (np.asarray(o2["center"]) - np.asarray(o["center"]))[:2]
            out["moved_cm"] = round(float(d @ v[:2]), 1)
            out["side_cm"] = round(float(d @ np.array([-v[1], v[0]])), 1)
            if stop is None and out["moved_cm"] < 0.6 * dist * 100:
                out["code"] = "SHORT"
        elif stop is None:
            out["code"] = "NOT_SEEN"
            out["why"] = f"the push ran, but afterwards the cameras do not find {obj}; locate it"
        if stop == "NO_CONTACT":
            out["why"] = f"the gripper went through the push line but {obj} did not move (missed it or it is blocked); stopped after {first * 100:.1f} cm"
        if stop == "TOPPLED":
            out["why"] = f"{obj} tipped over at the first contact; stopped"
        if stop == "NOT_SEEN":
            out["why"] = f"after the first contact the cameras no longer find {obj}; stopped (locate it)"
        if stop == "PLAN_FAIL":
            out["why"] = f"could not bring the gripper back to {obj} after the first contact; stopped"
        rec.update(stop=stop, first_moved_cm=None if m1 is None else round(m1, 1))
        self.trace.append({"slender_push": rec})
        return {arm: out}



# Honest height of a held object (return semantics).
# (i) the parent's result is returned untouched unless the parent said True (the estimate may only
# turn a pass into 'below the height', never a fail / unsure into a pass); (ii) the lower height is reported as an ESTIMATE with
# its basis (height_est_cm, height_basis on the check), not as a measurement, and the why no longer asserts that the object hangs.
# Height estimate: (i) only an object held by ONE arm (two arms hold it up between them: parent result); (ii) the swing estimate
# is the middle between the rigid lowest point and the lowest point when hanging freely below the finger axis (a half-angle
# swing read 2-3 cm too low on the bench).
# The parent's "height" / "lifted" check for a held object moves one grasp-time reference point (footprint centre at the
# measured bottom) rigidly with the gripper.  Two things make a held object reach lower than that point: (1) the hand has
# tilted / turned since the grasp, so another part of the object is now lowest; (2) an object gripped away from its centre
# of mass swings about the finger closing axis under its own weight (the fingers pinch, they do not clamp the rotation).
# The estimate moves the object's own grasp-time cloud with the gripper (1), and when the cloud's centroid sits off the finger
# axis (lever >= HELD_H_LEVER_CM) it also takes into account a swing of half the way towards hanging straight below the
# axis (2) -- a conservative middle between "did not swing" and "hangs freely".  Gate: only when the resulting lowest
# point is more than HELD_H_GAP_CM below the parent's number does the check use it (and say so in its why); otherwise the
# parent's result is returned word for word.  No new motion, no simulator truth, no task names.
_HeldHeightParent = Harness
HELD_H_GAP_CM = 1.5
HELD_H_LEVER_CM = 2.0
HELD_H_SWING = 0.5


def _hh_rot(c, ang):
    K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


class Harness(_HeldHeightParent):
    def _hh_low_cm(self, arm):
        """(lowest point of the tracked cloud, lowest point after the partial swing or None, lever cm) above the table."""
        h = self.hold.get(arm)
        if not h or h.get("pts_h") is None or len(h["pts_h"]) < 30:
            return None
        hp = self.A.hand_pose(arm)
        R = Gm.mat(hp)[:3, :3]
        t = self.A.tcp_point(hp)
        P = (R @ np.asarray(h["pts_h"], float).T).T                    # object points relative to the fingertip centre
        rigid = float((np.percentile(P[:, 2] + t[2], 2) - self.tz) * 100)
        c = np.asarray(R[:, 1], float)
        c = c / (np.linalg.norm(c) + 1e-12)                             # finger closing axis
        d = P.mean(0)
        d_perp = d - c * float(d @ c)
        lever = float(np.linalg.norm(d_perp)) * 100
        down = np.array([0.0, 0.0, -1.0]) + c * float(c[2])
        if abs(c[2]) > 0.8 or lever < HELD_H_LEVER_CM or np.linalg.norm(down) < 1e-6:
            return rigid, None, lever
        u, v = d_perp / np.linalg.norm(d_perp), down / np.linalg.norm(down)
        ang = float(np.arctan2(np.dot(np.cross(u, v), c), np.dot(u, v)))
        z = (_hh_rot(c, ang) @ P.T).T[:, 2] + t[2]
        hang = float((np.percentile(z, 2) - self.tz) * 100)
        return rigid, (1.0 - HELD_H_SWING) * rigid + HELD_H_SWING * hang, lever

    def _pred(self, p, args, now):
        ok, why = _HeldHeightParent._pred(self, p, args, now)
        if p not in ("lifted", "height") or not args or ok is not True:
            return ok, why                                # only a parent pass can be revised (never fail / unsure -> pass)
        held_by = [x for x, o in self.holding.items() if o == args[0] and x in self.hold]
        if len(held_by) != 1:
            return ok, why
        need = float(args[1]) if len(args) > 1 else 10.0
        old = (self.held_est(held_by[0])[1] - self.tz) * 100
        e = self._hh_low_cm(held_by[0])
        if e is None:
            return ok, why
        rigid, swing, lever = e
        low = min(rigid, swing) if swing is not None else rigid
        if old - low <= HELD_H_GAP_CM:
            return ok, why
        used_swing = swing is not None and swing < rigid
        basis = ("estimate, not measured now: the grasp-time point cloud of the object moved with the current gripper pose"
                 + ("; it is gripped off its centre, so it may also have swung round the fingers: taken half-way between not "
                    "swinging and hanging straight down" if used_swing else ""))
        self.__dict__.setdefault("_hh_marks", {})[repr([p, *args])] = {"height_est_cm": round(low, 1), "height_basis": basis}
        self.trace.append({"held_height": {"obj": args[0], "grip_estimate_cm": round(old, 1), "estimate_cm": round(low, 1),
                                           "rigid_cm": round(rigid, 1), "lever_cm": round(lever, 1), "swing": used_swing, "need": need}})
        if low >= need:
            return True, why + f"; estimated lowest point about {low:.1f} cm (estimate, not measured)"
        return False, (f"held by {held_by}; the gripper-pose estimate says bottom {old:.1f} cm, but the estimated lowest point of the held "
                       f"{args[0]} is only about {low:.1f} cm ({basis}); not confirmed to be {need:g} cm up")

    def evaluate(self, checks):
        self.__dict__["_hh_marks"] = {}
        allok, out = _HeldHeightParent.evaluate(self, checks)
        marks = self.__dict__.get("_hh_marks") or {}
        for c in out:
            m = marks.get(repr(c.get("check")))
            if m:
                c.update(m)
        return allok, out


# press into_face with the closed fingertips side by side.
# Only press(direction="into_face") inside the panel scope is changed; every other call is the parent's. No global
# SYSTEM text is added: what the tool did is said only in the result of the press that used it.
# Part "roll": the parent's into_face push (planner_base.Harness._press: along the plate's own normal, at its protruding half) holds the
#   closed gripper with its closing axis vertical, so the two closed fingertips sit one above the other.  Here the same push
#   (same point, same normal, same 7 cm start / 4 cm depth, same reach check and retreat) is done with the hand turned 90 deg
#   about the push direction (closing axis horizontal, fingertips side by side).  The parent's code is not modified or
#   replaced at runtime: _pn_press is a local copy of the parent's into_face branch that builds the turned hand pose itself.
# Part "rollfb": if the turned push stops well short of the plate (NOT_REACHED), the parent's press is called once, unchanged.
# Part "roll2": after a turned push that reached the plate (OK), the plate is looked at again (the parent's perception, one
#   call) and compared with the look before the push.  Only when the plate is seen again with the same protruding half, a
#   normal within PANEL_N_DEG and a protrusion difference that moved by less than PANEL_PROT_CM (no visible change) is a second
#   push tried, with the same turned hand on the same half, 2 cm from the plate's middle height - and only when that point lies
#   inside that half's observed height range with points around it; otherwise nothing is done and the result says why.
#   The result reports a second push as attempted / reached (fingertip depth), never as done or as having changed the plate.
# Inputs read by the new code: the camera observation returned by the parent's self.see (unchanged parent perception), the
# parent's _panel on those points, and the robot's own hand / TCP pose and motion planner (self.A, self.tcp, self.feasible,
# self.run_seq - the same calls the parent press makes).  No simulator object or scene state, no task names, no checker values.
_PanelParent = Harness
PANEL_PARTS = set(_os.environ.get("HARNESS_PANEL_PARTS", "roll,rollfb,roll2").split(","))
PANEL_R2_DZ = 0.02
PANEL_N_DEG = 5.0        # normal change that counts as a visible change of the plate
PANEL_PROT_CM = 0.2      # change of the two halves' protrusion difference that counts as a visible change
PANEL_EDGE = 0.003       # the second point must lie this far inside the half's observed height range


class Harness(_PanelParent):
    @staticmethod
    def _pn_q(a):
        a = np.asarray(a, float)
        return _BASE.G.hand_quat(a, [-float(a[1]), float(a[0]), 0.0])

    def _pn_press(self, arm, obj):
        """copy of the parent's into_face branch (planner_base.Harness._press with direction="into_face") with the turned hand."""
        self._forget_release(obj)
        o = self.see(obj)
        if o is None:
            return {arm: {"code": "NOT_SEEN"}}, None
        pts = np.asarray(o["pts"], float)
        pn = self._panel(pts)
        res, tried = None, []
        if pn is None:
            tried.append("into_face: no thin upright plate found")
        else:
            a = -pn["n"]
            p = pn["p"]
            self.trace.append({"press_panel": {"normal_deg": round(float(np.degrees(np.arctan2(a[1], a[0]))), 1), "raised": pn["raised"],
                                               "prot_diff_cm": pn["prot_diff_cm"], "p_cm": self.cm(p)}, "panel_turned_hand": True})
            q = self._pn_q(a)
            start, end = p - a * 0.07, p + a * 0.04
            hs_, he_ = self.A.hand_from_tcp(start, q), self.A.hand_from_tcp(end, q)
            if not self.feasible(arm, [hs_]):
                tried.append("into_face: unreachable")
            else:
                codes = self.run_seq({arm: [{"type": "grip", "val": 0.0}, {"type": "move", "pose": hs_.tolist()}]})
                if codes[arm] == "PLAN_FAIL":
                    tried.append("into_face: plan failed")
                else:
                    self.A.run({arm: [{"type": "move", "pose": he_.tolist()}]})
                    depth = float((self.tcp(arm) - p) @ a)
                    self.A.run({arm: [{"type": "move", "pose": hs_.tolist()}]})
                    res = (p, depth, a)
        if not self.holding.get(arm):
            self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        if res is None:
            return {arm: {"code": "UNREACHABLE", "why": f"no reachable press direction for {obj}: " + "; ".join(tried)}}, None
        p, depth, a = res
        reached = depth >= -0.05
        if reached:
            self.pressed.add(obj)
        return {arm: {"code": "OK" if reached else "NOT_REACHED", "direction": "into_face", "pressed_point_cm": self.cm(p),
                      "reach_past_surface_cm": round(depth * 100, 1),
                      "note": "whether the button / switch changed is not measured; judge it from the images (if it changed, do not press it again: a second press can undo it)"}}, (o, a, pn)

    def _pn_second(self, arm, obj, o, pn0, a, out):
        """returns the second_stroke record (always: attempted or not, and why)."""
        p = self.world(out["pressed_point_cm"])
        pts0 = np.asarray(o["pts"], float)
        zmid = float(np.median(pts0[:, 2]))
        dz = float(p[2]) - zmid
        if abs(dz) >= PANEL_R2_DZ - 0.002 or abs(dz) < 0.003:
            return {"attempted": False, "why": "the first push point is not between the plate middle and the second point"}
        o1 = self.see(obj)                                # look again: did the first push visibly change the plate?
        pn1 = self._panel(np.asarray(o1["pts"], float)) if o1 is not None else None
        if pn1 is None:
            return {"attempted": False, "why": "after the first push the plate is not seen as a plate again; not pushed again"}
        dn = float(np.degrees(np.arccos(np.clip(abs(float(np.asarray(pn1["n"]) @ np.asarray(pn0["n"]))), -1.0, 1.0))))
        dprot = abs(float(pn1["prot_diff_cm"]) - float(pn0["prot_diff_cm"]))
        look = {"normal_change_deg": round(dn, 1), "protrusion_change_cm": round(dprot, 1), "protruding_half_before": pn0["raised"],
                "protruding_half_after": pn1["raised"]}
        if pn1["raised"] != pn0["raised"] or dn >= PANEL_N_DEG or dprot >= PANEL_PROT_CM:
            return {"attempted": False, "look_after_first": look,
                    "why": "the plate looks different after the first push (it may have changed); not pushed again"}
        p2 = np.asarray(p, float).copy()
        p2[2] = zmid + np.sign(dz) * PANEL_R2_DZ
        half = pts0[(pts0[:, 2] - zmid) * np.sign(dz) > 0]
        n0 = np.asarray(pn0["n"], float)
        lat = np.array([-n0[1], n0[0], 0.0])
        if len(half) < 5:
            return {"attempted": False, "look_after_first": look, "why": "too few points on that half of the plate"}
        z_lo, z_hi = float(np.percentile(half[:, 2], 5)), float(np.percentile(half[:, 2], 95))
        near = half[(np.abs(half[:, 2] - p2[2]) <= 0.006) & (np.abs((half - p2) @ lat) <= 0.015)]
        if not (z_lo + PANEL_EDGE <= p2[2] <= z_hi - PANEL_EDGE) or len(near) < 5:
            return {"attempted": False, "look_after_first": look,
                    "why": "the second point would be outside the observed plate (or without points around it); not pushed"}
        q = self._pn_q(a)
        hs_, he_ = self.A.hand_from_tcp(p2 - a * 0.07, q), self.A.hand_from_tcp(p2 + a * 0.04, q)
        if not self.feasible(arm, [hs_]):
            return {"attempted": False, "look_after_first": look, "why": "not reachable"}
        rec = {"look_after_first": look, "point_cm": self.cm(p2)}
        codes = self.run_seq({arm: [{"type": "grip", "val": 0.0}, {"type": "move", "pose": hs_.tolist()}]})
        if codes[arm] == "PLAN_FAIL":
            rec.update(attempted=False, why="plan failed")
        else:
            self.A.run({arm: [{"type": "move", "pose": he_.tolist()}]})
            depth = float((self.tcp(arm) - p2) @ a)
            self.A.run({arm: [{"type": "move", "pose": hs_.tolist()}]})
            rec.update(attempted=True, reached=bool(depth >= -0.05), reach_past_surface_cm=round(depth * 100, 1),
                       note=("a second push on the same half was tried" + ("" if depth >= -0.05 else "; the fingertip stopped short of the plate")
                             + "; whether the plate changed is not confirmed: judge it from the images"))
        if not self.holding.get(arm):
            self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        return rec

    def press(self, arm, obj, direction=None, confirm=False):
        if "roll" not in PANEL_PARTS or direction != "into_face" or not getattr(self, "_primitive_scope", False):
            return _PanelParent.press(self, arm, obj, direction, confirm)
        r, info = self._pn_press(arm, obj)
        out = r.get(arm)
        if out.get("code") in ("OK", "NOT_REACHED"):
            out["hand"] = "the push was made with the hand turned about the push direction (closed fingertips meant to be side by side)"
        if "rollfb" in PANEL_PARTS and out.get("code") == "NOT_REACHED":
            self.trace.append({"panel_fallback": out.get("reach_past_surface_cm")})
            r2 = _PanelParent.press(self, arm, obj, direction, confirm)
            o2 = r2.get(arm) if isinstance(r2, dict) else None
            if isinstance(o2, dict):
                o2["turned_hand_first"] = {"code": "NOT_REACHED", "reach_past_surface_cm": out.get("reach_past_surface_cm"),
                                           "note": "the push with the turned hand stopped short of the plate, so the push was tried once more with the hand upright"}
            return r2
        if "roll2" in PANEL_PARTS and out.get("code") == "OK" and info is not None:
            try:
                out["second_stroke"] = self._pn_second(arm, obj, info[0], info[2], info[1], out)
            except Exception as ex:  # noqa: BLE001
                out["second_stroke"] = {"attempted": None, "why": f"stopped by an error ({type(ex).__name__}); whether a second push happened is not confirmed"}
                if not self.holding.get(arm):
                    try:
                        self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
                    except Exception:  # noqa: BLE001
                        pass
            self.trace.append({"panel_second_stroke": out["second_stroke"]})
        return r


# a pulled drawer measured by the cameras
# and putting a held object down into it.  Only the explicit drawer scope (articulate motion="pull"/"slide_out", already gated)
# changes, plus a place into the drawer that pull reported (state exists only after such a pull); everything else = parent.
# (1) pull: the drawer front is measured from the cabinet's point cloud before and after the slide (front-most points in the height
#     band of the chosen handle); the result reports drawer_open_cm.  The code is no longer EXECUTED when the cameras see the drawer
#     front move clearly less than the hand: SHORT with why, and one automatic second grasp+slide when it hardly moved.
#     drawer_inside_xy_cm comes from the measured opening (not from the hand path); the open part's floor and rim heights are
#     measured from the cloud when visible.
# (2) place(arm, obj, ref="table"/that cabinet, xy inside the opened drawer): carries the object above the measured rim (the parent
#     carried it at table-place height and struck the drawer walls -> POSE_ERR, then released far from the target), lowers it
#     onto the measured floor, and does not let go when the hand is blocked away from the target (BLOCKED, still holding).  After
#     release the object is re-measured: OK only when the cameras see it inside the opened drawer region, else UNSURE.
# No simulator truth, no task names, no checker values; SYSTEM / tool descriptions unchanged (new text only in these results).
# (a) a place with explicit turn_deg / align / held_part / face / face_now goes to the parent path
# (b) before every release the current drawer front is measured: unknown, pushed back in (>2 cm) or the held object not over the
# current opening -> BLOCKED, still holding; (c) after release OK / in_drawer=True only with the current front measured, the centre
# inside the opening as measured now and the object's bottom below the measured rim (and not below the floor); unknown -> UNSURE,
# in_drawer=None.
_DrawerParent = Harness
DRAWER_BAND_Z = 0.04         # height band around the handle used for the drawer front
DRAWER_BAND_X = 0.12
DRAWER_MIN_OPEN_CM = 3.0     # below this the drawer did not open: one more grasp+slide
DRAWER_RIM_CLEAR = 0.04      # carry the object bottom this far above the measured rim
DRAWER_HANDLE_MIN_PTS = 3
DRAWER_CLEAR_M = 0.02        # arm link points (forward kinematics) closer than this to the cabinet cloud count as a hit
DRAWER_MAX_CHECK = 24        # IK checks for the drawer placement
DRAWER_RAW_MIN = 15          # full-resolution points for a handle group
DRAWER_PUSHED_M = 0.02       # the drawer front moved back in more than this during a placement -> stop / re-judge
DRAWER_TRIES = 3             # grasp+slide attempts while the cameras see no opening


class Harness(_DrawerParent):
    def _dr_front(self, pts, hx, hz):
        pts = np.asarray(pts, float)
        m = (np.abs(pts[:, 2] - hz) < DRAWER_BAND_Z) & (np.abs(pts[:, 0] - hx) < DRAWER_BAND_X)
        if int(m.sum()) < 20:
            return None
        return float(np.percentile(pts[m, 1], 5))

    def _dr_raw_handles(self, pts, face):
        """handles = full-resolution camera points (robot removed) just in front of the cabinet's front face, within its width,
        grouped by height; the lowest group with enough points.  None if nothing clear."""
        try:
            views = PC.capture(self.sim.rd)
            views = PC._drop_robot(views, PC._robot_points(self.sim.rd))
            R_ = np.concatenate([v["xyz"][v["valid"]] for v in views])
        except Exception as e:  # noqa: BLE001
            self.trace.append({"drawer_raw_error": repr(e)[:160]})
            return None
        x0, x1 = np.percentile(pts[:, 0], 2) + 0.01, np.percentile(pts[:, 0], 98) - 0.01
        top = float(np.percentile(pts[:, 2], 98))
        m = ((R_[:, 0] > x0) & (R_[:, 0] < x1) & (R_[:, 1] < face - 0.006) & (R_[:, 1] > face - 0.25)
             & (R_[:, 2] > self.tz + 0.03) & (R_[:, 2] < top))
        H_ = R_[m]
        if len(H_) < DRAWER_RAW_MIN:
            self.trace.append({"drawer_raw_handles": 0, "n": int(len(H_))})
            return None
        zs = np.sort(H_[:, 2])
        groups, cur = [], [zs[0]]
        for z in zs[1:]:
            if z - cur[-1] > 0.02:
                groups.append(cur); cur = [z]
            else:
                cur.append(z)
        groups.append(cur)
        groups = [g for g in groups if len(g) >= DRAWER_RAW_MIN]
        bars = []
        for g in groups:                                   # a handle is a thin horizontal bar: long across, short front-back
            G_ = H_[(H_[:, 2] >= g[0] - 1e-6) & (H_[:, 2] <= g[-1] + 1e-6)]
            ys = G_[:, 1]
            # the bar is the front-most thin layer of the group (an opened drawer's sides lie behind it)
            G_ = G_[ys < np.percentile(ys, 5) + 0.02]
            if len(G_) >= DRAWER_RAW_MIN and np.ptp(np.percentile(G_[:, 0], [5, 95])) >= 0.08:
                bars.append(G_)
        groups = bars
        if not groups:
            self.trace.append({"drawer_raw_handles": 0, "n": int(len(H_))})
            return None
        sel = groups[0]
        h = (float(np.median(sel[:, 0])), float(np.median(sel[:, 1])), float(np.median(sel[:, 2])))
        self.trace.append({"drawer_raw_handles": len(groups), "sizes": [len(g) for g in groups], "chosen_cm": self.cm(h)})
        return h

    def _dr_handle_poses(self, arm, hx, hy, hz):
        """feasible handle grasps, one per approach pitch (first feasible height offset / finger direction), at most 3."""
        best = []
        for pitch in (60.0, 45.0, 75.0, 30.0):
            b = np.radians(pitch)
            a = np.array([0.0, np.cos(b), -np.sin(b)])
            found = None
            for dz in (0.0, -0.015, 0.015):
                p = np.array([hx, hy - 0.005, hz + dz])
                for c in ([1.0, 0, 0], [-1.0, 0, 0]):
                    q = G.hand_quat(a, c)
                    pre, hg = self.A.hand_from_tcp(p - a * 0.06, q), self.A.hand_from_tcp(p, q)
                    if self.feasible(arm, [pre]) and self.feasible(arm, [hg]):
                        found = (p, q, pre, hg)
                        break
                if found:
                    break
            if found:
                best.append(found)
            if len(best) >= 3:
                break
        return best

    def _dr_grasp_slide(self, arm, cand, v, dist):
        """one attempt: grasp the handle with candidate pose cand; if the fingers close on nothing, back off (moved=None);
        else slide in 2 cm steps, release, back off, home.  Returns (moved_cm or None, finger gap)."""
        p, q, pre, hg = cand
        self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}, {"type": "move", "pose": hg.tolist()},
                            {"type": "grip", "val": 0.0}]})
        gap = float(self.A.opening(arm))
        moved = None
        if gap >= GAP_MIN:
            moved = 0.0
            for k in range(1, int(dist // 2) + 1):
                c = self.run_seq({arm: [self.move_tcp(arm, p + v * 0.02 * k, q)]})
                if c[arm] != "OK":
                    break
                moved = 2.0 * k
        self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        t = self.tcp(arm)
        self.run_seq({arm: [self.move_tcp(arm, t + np.r_[0, -0.05, 0.04])]})
        self.call("home", {"arm": arm})
        return moved, gap

    def _articulate_slide(self, arm, obj, o, d, motion, dist):
        pts = np.asarray(o["pts"], float)
        face = float(d["face_y"])
        pro = pts[(pts[:, 1] < face - 0.006) & (pts[:, 2] > self.tz + 0.03)]
        hx, hy, hz = d["handle"]
        if len(pro) >= 15:
            zs = np.sort(pro[:, 2])
            groups, cur = [], [zs[0]]
            for z in zs[1:]:
                if z - cur[-1] > 0.02:
                    groups.append(cur); cur = [z]
                else:
                    cur.append(z)
            groups.append(cur)
            groups = [g for g in groups if len(g) >= DRAWER_HANDLE_MIN_PTS] or groups   # parent: >= 8 (the subsampled cloud often keeps fewer on the lowest handle)
            g0 = groups[0]
            sel = pro[(pro[:, 2] >= g0[0] - 1e-6) & (pro[:, 2] <= g0[-1] + 1e-6)]
            hx, hy, hz = float(np.median(sel[:, 0])), float(np.median(sel[:, 1])), float(np.median(sel[:, 2]))
            self.trace.append({"drawer_handles": len(groups), "chosen_cm": self.cm([hx, hy, hz])})
        raw = self._dr_raw_handles(pts, face)          # the full-resolution depth in front of the face (thin bars keep few points above)
        if raw is not None:
            hx, hy, hz = raw
        dist = float(dist) if 3.0 <= float(dist) <= 30.0 else 15.0
        if arm not in ARMS:
            arm = "left" if hx < 0 else "right"
        other = "left" if arm == "right" else "right"
        if self.holding.get(other) is None:
            self.call("home", {"arm": other})
        pull = str(motion) in ("pull", "slide_out")
        v = np.array([0.0, -1.0, 0.0]) if pull else np.array([0.0, 1.0, 0.0])
        f0 = self._dr_front(pts, hx, hz)
        rec = {"front0_cm": None if f0 is None else round(f0 * 100, 1)}
        cands = self._dr_handle_poses(arm, hx, hy, hz)
        if not cands:
            self.trace.append({"drawer_slide": rec})
            return {arm: {"code": "UNREACHABLE", "why": f"cannot reach the drawer handle of {obj} with the {arm} arm; try the other arm"}}
        moved, opened, o2, f1, tries = 0.0, None, None, None, []
        for i in range(DRAWER_TRIES):                         # the drawer front is measured after each attempt; stop once it moved
            mv, gap = self._dr_grasp_slide(arm, cands[i % len(cands)], v, dist)
            o2 = self.see(obj)
            f1 = None if (o2 is None or f0 is None) else self._dr_front(o2["pts"], hx, hz)
            opened = None if f1 is None else (f0 - f1) * 100 * (1 if pull else -1)
            tries.append({"slid_cm": mv, "gap": round(gap, 3), "open_cm": None if opened is None else round(opened, 1)})
            moved = mv or 0.0
            if not pull or opened is None or opened >= DRAWER_MIN_OPEN_CM:
                break
        rec["tries"] = tries
        retried = len(tries) > 1
        out = {"slid_cm": moved, "handle_cm": self.cm([hx, hy, hz])}
        if opened is None:
            out["code"] = "SHORT" if moved < min(10.0, dist) else "UNSURE"
            out["drawer_open_cm"] = None
            out["note"] = "the hand slid the handle, but the cameras could not measure the drawer front; check the images"
        else:
            out["drawer_open_cm"] = round(opened, 1)
            if opened >= min(10.0, 0.6 * dist):
                out["code"] = "EXECUTED"
            else:
                out["code"] = "SHORT"
                out["why"] = (f"the hand slid {moved:.0f} cm but the cameras see the drawer front moved only {opened:.0f} cm"
                              + (f" ({len(tries)} grasps of the handle were tried)" if retried else "") + "; check the images")
            out["note"] = "drawer_open_cm is measured by the cameras (drawer front before vs after)"
        self.trace.append({"drawer_slide": rec})
        self._dr_drawer = None
        if pull and opened is not None and opened >= DRAWER_MIN_OPEN_CM:
            m = opened / 100
            inside = np.r_[hx, f0 - m / 2 + 0.01]          # middle of the part now outside the cabinet front
            y_lo, y_hi = f0 - m + 0.015, f0 - 0.005
            st = {"pts": np.asarray(o2["pts"], float), "f0": f0, "f1": f1, "hx": hx, "hz": hz, "xy": inside, "x_lo": hx - 0.14, "x_hi": hx + 0.14, "y_lo": y_lo, "y_hi": y_hi, "obj": obj, "floor": None, "rim": None}
            P_ = np.asarray(o2["pts"], float)
            reg = P_[(P_[:, 0] > hx - 0.12) & (P_[:, 0] < hx + 0.12) & (P_[:, 1] > y_lo + 0.01) & (P_[:, 1] < f0 - 0.025)]
            if len(reg) >= 30:
                st["rim"] = float(np.percentile(reg[:, 2], 97))
                st["floor"] = float(np.percentile(reg[:, 2], 5))
            fr = P_[(np.abs(P_[:, 0] - hx) < DRAWER_BAND_X) & (np.abs(P_[:, 1] - f1) < 0.015) & (P_[:, 2] < hz + 0.10)]
            if len(fr) >= 20:                                # the drawer's front panel top bounds the rim too
                st["rim"] = max(st["rim"] or 0.0, float(np.percentile(fr[:, 2], 97)))
            self._dr_drawer = st
            out["drawer_inside_xy_cm"] = r1(inside * 100)
            if st["rim"] is not None:
                out["drawer_rim_z_cm"] = round((st["rim"] - self.tz) * 100, 1)
            if st["floor"] is not None:
                out["drawer_floor_z_cm"] = round((st["floor"] - self.tz) * 100, 1)
            out["hint"] = (f"to put something into this drawer: place(arm, object, ref=\"table\", xy={r1(inside * 100)}) with the arm that "
                           "holds it: it is carried above the drawer's rim, lowered into the drawer and released")
        return {arm: out}

    def _dr_link_points(self, arm, hand_pose):
        """world points of this arm's links (incl. its wrist camera) at the joint solution the planner finds for hand_pose."""
        try:
            res = self.planner(arm)(list(map(float, hand_pose)))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(res, dict) or res.get("status") != "Success" or res.get("position") is None or not len(res["position"]):
            return None
        rb = self.A.T.robot
        ent = rb.left_entity if arm == "left" else rb.right_entity
        pl = rb.left_planner if arm == "left" else rb.right_planner
        pm = self.__dict__.setdefault("_dr_pm", {}).get(id(ent))
        if pm is None:
            pm = ent.create_pinocchio_model()
            self._dr_pm[id(ent)] = pm
        idx = [pl.all_joints.index(n) for n in pl.active_joints_name if n in pl.all_joints]
        full = np.asarray(ent.get_qpos(), float).copy()
        full[idx] = np.asarray(res["position"][-1], float)[: len(idx)]
        pm.compute_forward_kinematics(full)
        root = ent.get_root_pose() if hasattr(ent, "get_root_pose") else ent.get_pose()
        names = [l.get_name() for l in ent.get_links()]
        pre = ("fl_" if arm == "left" else "fr_")
        mine = [i for i, n in enumerate(names) if n.startswith(pre) or n.startswith(arm + "_camera")]
        P, prev = [], None
        for i in mine:
            p = np.asarray((root * pm.get_link_pose(i)).p, float)
            P.append(p)
            if prev is not None and np.linalg.norm(p - prev) < 0.4:
                for f in (0.25, 0.5, 0.75):
                    P.append(prev + f * (p - prev))
            prev = p
        return np.asarray(P)

    def _dr_hits(self, arm, poses, st, rim):
        """cabinet points (camera) next to the arm's links at the given hand poses; the open drawer's inside (above its floor,
        below its rim) is free space."""
        try:
            from scipy.spatial import cKDTree
            C = st["pts"]
            free = ((C[:, 0] > st["x_lo"] + 0.02) & (C[:, 0] < st["x_hi"] - 0.02) & (C[:, 1] > st["y_lo"]) & (C[:, 1] < st["y_hi"] - 0.01)
                    & (C[:, 2] < rim + 0.005))
            C = C[~free]
            if not len(C):
                return 0
            tree = cKDTree(C)
            n = 0
            for hp in poses:
                A = self._dr_link_points(arm, hp)
                if A is None or not len(A):
                    continue
                d, _ = tree.query(A, k=1)
                n += int((d < DRAWER_CLEAR_M).sum())
            return n
        except Exception as e:  # noqa: BLE001
            self.trace.append({"drawer_hits_error": repr(e)[:160]})
            return 0

    def _dr_front_now(self, st):
        o = self.see(st["obj"])
        return None if o is None else self._dr_front(o["pts"], st["hx"], st["hz"])

    def _dr_in(self, st, xy):
        return st["x_lo"] <= xy[0] <= st["x_hi"] and st["y_lo"] - 0.02 <= xy[1] <= st["y_hi"] + 0.02

    def place(self, arm, obj, ref, relation="on", xy=None, release=True, *args, **kwargs):
        st = getattr(self, "_dr_drawer", None)
        explicit = any(v not in (None, False, "", 0) for v in args[1:]) or any(
            kwargs.get(k) not in (None, False, "", 0) for k in kwargs if k != "height_cm")   # turn_deg / align / held_part / face / face_now / ...
        if (st is None or not release or explicit or self.holding.get(arm) != obj
                or str(relation or "on").lower() not in ("on", "in", "into", "inside")):
            return _DrawerParent.place(self, arm, obj, ref, relation, xy, release, *args, **kwargs)
        if ref == st["obj"] and xy is None:
            txy = np.asarray(st["xy"], float)
        elif ref in (None, "table", st["obj"]) and xy is not None and self._dr_in(st, np.asarray(xy, float) / 100):
            txy = np.asarray(xy, float)[:2] / 100
        else:
            return _DrawerParent.place(self, arm, obj, ref, relation, xy, release, *args, **kwargs)
        self.trace.append({"drawer_place": "drawer", "txy_cm": r1(txy * 100)})
        return self._dr_place(arm, obj, txy, st)

    def _dr_obj_out(self, arm, q, dest, st):
        """held-object points (in-hand cloud moved with the hand) outside the opened part of the drawer, in plan view."""
        h = self.hold.get(arm) or {}
        ph = h.get("pts_h")
        if ph is None or st.get("f1") is None:
            return 0
        R = Gm.mat(self.A.hand_from_tcp(dest, q))[:3, :3]
        W = (R @ np.asarray(ph, float).T).T + dest
        bad = (W[:, 1] > st["f0"] - 0.01) | (W[:, 1] < st["f1"] + 0.012) | (W[:, 0] < st["x_lo"]) | (W[:, 0] > st["x_hi"])
        return int(bad.sum())

    def _dr_place(self, arm, obj, txy, st):
        est = self.held_est(arm)
        if est is None:
            o = self.see(obj)
            if o is None:
                return {arm: {"code": "NOT_SEEN", "why": f"cannot see the held {obj}"}}
            ow, obot = self.world(o["center"]), float(self.world(o["lo"])[2])
        else:
            ow, obot = est
        tcp = self.tcp(arm)
        floor = st["floor"] if st["floor"] is not None else self.tz + 0.01
        rim = st["rim"] if st["rim"] is not None else floor + 0.15
        q0 = self.A.hand_pose(arm)[3:]
        from transforms3d.axangles import axangle2mat
        from transforms3d.quaternions import mat2quat
        rdir = G.unit(np.r_[txy - G.ARM_BASE[arm], 0.0])
        kax = G.unit(np.cross([0, 0, 1.0], rdir))
        rots = [(y, 0) for y in (0, 20, -20, 40, -40, 60, -60, 90, -90)] + [(0, t) for t in (20, -20, 35, -35)]
        shifts = [(0.0, 0.0), (0.0, -0.015), (0.0, 0.015), (0.0, -0.03), (0.03, 0.0), (-0.03, 0.0)]
        pre = []
        for i, (yaw, tilt) in enumerate(rots):        # cheap first: does the held object fit the opening at this turn / shift
            Rm = axangle2mat([0, 0, 1.0], np.radians(yaw)) if not tilt else axangle2mat(kax, np.radians(tilt))
            q = G.quat_mul(np.asarray(mat2quat(Rm)), q0)
            off = Rm @ (ow - tcp)
            boff = Rm @ (np.r_[ow[:2], obot] - tcp)
            for j, (sx, sy) in enumerate(shifts):
                t_ = txy + np.r_[sx, sy]
                if not self._dr_in(st, t_):
                    continue
                dest = np.r_[t_ - off[:2], floor + 0.01 - boff[2]]
                pre.append((self._dr_obj_out(arm, q, dest, st), i, j, yaw if not tilt else f"tilt {tilt}", q, dest, float(-boff[2])))
        pre.sort(key=lambda c_: (c_[0], c_[1], c_[2]))
        cands, checked = [], 0
        for out_, i, j, pose, q, dest, hang in pre:
            if checked >= DRAWER_MAX_CHECK or (cands and cands[0][0] == 0):
                break
            for lift_ in (DRAWER_RIM_CLEAR, 0.02):
                safe = max(rim + lift_ + hang, dest[2] + 0.06)
                hd, hs = self.A.hand_from_tcp(dest, q), self.A.hand_from_tcp(np.r_[dest[:2], safe], q)
                checked += 1
                if not (self.feasible(arm, [hd]) and self.feasible(arm, [hs])):
                    continue
                score = self._dr_hits(arm, [hd, hs], st, rim) + 5 * out_
                cands.append((score, pose, q, dest, safe, hang, out_))
                cands.sort(key=lambda c_: c_[0])
                break
        self.trace.append({"drawer_clear": {"fit_checked": len(pre), "ik_checked": checked, "feasible": len(cands),
                                          "best": None if not cands else {"score": cands[0][0], "obj_out": cands[0][6], "pose": cands[0][1]}}})
        if not cands:
            return {arm: {"code": "UNREACHABLE", "why": f"the {arm} arm cannot reach above the open drawer with {obj}; bring it closer or use the other arm"}}
        rec = {"rim_cm": round((rim - self.tz) * 100, 1), "floor_cm": round((floor - self.tz) * 100, 1), "attempts": []}
        done_low = None
        for score, pose, q, dest, safe, hang, out_ in cands[:2]:   # a blocked descent goes back up and tries the next pose once
            t0 = self.tcp(arm)
            codes = self.run_seq({arm: [self.move_tcp(arm, np.r_[t0[:2], max(t0[2], safe)]), self.move_tcp(arm, np.r_[t0[:2], max(t0[2], safe)], q),
                                        self.move_tcp(arm, np.r_[dest[:2], safe], q)]})
            if codes[arm] == "PLAN_FAIL":
                c2, _ = self._transport(arm, dest[:2], safe, q)
                codes[arm] = "OK" if c2 == "OK" else "PLAN_FAIL"
            t = self.tcp(arm)
            at = {"pose": pose, "score": score, "over": codes[arm], "tcp_cm": self.cm(t), "dest_cm": self.cm(dest), "safe_z_cm": round((safe - self.tz) * 100, 1)}
            rec["attempts"].append(at)
            if codes[arm] == "PLAN_FAIL" or float(np.linalg.norm(t[:2] - dest[:2])) > 0.03:
                continue
            n = max(2, int(np.ceil((safe - dest[2]) / 0.03)))
            dc = []
            for k in range(1, n + 1):
                c = self.run_seq({arm: [self.move_tcp(arm, np.r_[dest[:2], safe + (dest[2] - safe) * k / n], q)]})
                dc.append(c[arm])
                if c[arm] != "OK":
                    break
            t = self.tcp(arm)
            at.update(down_codes=dc, down_tcp_cm=self.cm(t))
            low = float(t[2] - hang)
            if low <= rim + 0.01 and float(np.linalg.norm(t[:2] - dest[:2])) <= 0.03:
                done_low = (low, dest)
                break
            self.run_seq({arm: [self.move_tcp(arm, np.r_[t[:2], safe], q)]})     # back up above the rim before the next pose
            fn = self._dr_front_now(st)
            if fn is not None and fn - st["f1"] > DRAWER_PUSHED_M:               # the blocked hand pushed the drawer in: the opening shrank
                rec["drawer_moved_cm"] = round((st["f1"] - fn) * 100, 1)
                break
        if done_low is None:
            self.trace.append({"drawer_place": rec})
            t = self.tcp(arm)
            if "drawer_moved_cm" in rec:
                self._dr_drawer = None
                return {arm: {"code": "BLOCKED", "why": f"could not lower {obj} into the drawer, and the blocked hand pushed the drawer about "
                              f"{-rec['drawer_moved_cm']:.0f} cm back in (cameras); still holding {obj}: check the images and pull the drawer open again before placing"}}
            return {arm: {"code": "BLOCKED", "why": f"could not lower {obj} into the open drawer ({len(rec['attempts'])} poses tried; the hand is at "
                          f"{self.cm(t)}); still holding {obj}"}}
        low, dest = done_low
        why = self._dr_pre_release(arm, st)
        if why is not None:                                # unknown / pushed in / hand not over the current opening: keep holding
            rec["pre_release"] = why
            self.trace.append({"drawer_place": rec})
            return {arm: {"code": "BLOCKED", "why": f"{why}; still holding {obj} (not released)"}}
        self._rl_scope(arm, self._dr_release, arm)
        self.holding.pop(arm, None)
        self.hold.pop(arm, None)
        o3 = self.see(obj)
        out = {"target_cm": self.cm(np.r_[txy, floor]), "released_bottom_z_cm": round((low - self.tz) * 100, 1)}
        if o3 is None:
            out.update(code="RELEASED_NOT_SEEN", why=f"released inside the drawer opening, but the cameras cannot find {obj} now; check the images")
        else:
            c3 = self.world(o3["center"])
            out["object_now"] = {"center": r1(o3["center"]), "bottom_z": round(float(o3["lo"][2]), 1)}
            fn = self._dr_front_now(st)
            if fn is not None:
                out["drawer_open_cm_now"] = round((st["f0"] - fn) * 100, 1)
                if fn - st["f1"] > DRAWER_PUSHED_M:
                    self._dr_drawer = None               # pushed in: the opening it reported is stale
            verdict, why = self._dr_contained(st, fn, c3, float(self.world(o3["lo"])[2]))
            out["code"] = "OK" if verdict is True else "UNSURE"
            out["in_drawer"] = verdict                      # True / False / None (not measured)
            if verdict is not True:
                out["why"] = f"after release: {why}; check the images"
        rec["result"] = out["code"]
        self.trace.append({"drawer_place": rec})
        return {arm: out}

    def _dr_contained(self, st, fn, c, bottom):
        """camera evidence that a released object is inside the drawer as it is NOW: the current drawer front measured, the
        object's centre between that front and the cabinet face and within the opened width, its bottom below the measured rim
        and not below the measured floor.  Returns (True / False / None, why)."""
        if fn is None:
            return None, "the drawer front could not be measured after the release, so the drawer opening is not confirmed"
        if st.get("rim") is None or st.get("floor") is None:
            return None, "the drawer rim / floor height were not measured, so containment is not confirmed"
        if not (st["x_lo"] <= c[0] <= st["x_hi"] and fn + 0.01 < c[1] < st["f0"] - 0.005):
            return False, f"the object centre {r1(np.asarray(c[:2]) * 100)} is outside the drawer opening as measured now"
        if bottom > st["rim"] - 0.005:
            return False, f"the object bottom ({(bottom - self.tz) * 100:.0f} cm) is not below the drawer rim ({(st['rim'] - self.tz) * 100:.0f} cm)"
        if bottom < st["floor"] - 0.03:
            return False, f"the object bottom ({(bottom - self.tz) * 100:.0f} cm) is below the drawer floor: probably not this object or not in the drawer"
        return True, "inside"

    def _dr_pre_release(self, arm, st):
        """before letting go: the current drawer front must be measured, not pushed back in, and the held object must be over
        the opening it leaves.  Returns None when release is allowed, else the reason."""
        fn = None
        for _ in range(2):
            fn = self._dr_front_now(st)
            if fn is not None:
                break
        if fn is None:
            return "the drawer front could not be measured before releasing"
        if fn - st["f1"] > DRAWER_PUSHED_M:
            return f"the drawer was pushed about {(fn - st['f1']) * 100:.0f} cm back in (cameras): pull it open again before placing"
        est = self.held_est(arm)
        if est is not None and not (st["x_lo"] <= est[0][0] <= st["x_hi"] and fn + 0.01 < est[0][1] < st["f0"] - 0.005):
            return "the held object is not over the drawer opening as measured now"
        return None

    def _dr_release(self, arm):
        self.open_gently(arm)
        self.retreat(arm)


# lean placement keeps a flat object's face square to the support.
# The parent stands the held object up with the smallest rotation that turns its long axis to the lean direction; the
# rotation about that long axis is left to chance, so a flat object (phone, card, plate) that lay on the table at a yaw ends
# up with its face turned sideways to the support by that yaw (the table pose is random), rests on one corner and topples
# sideways once released. For a flat held object (thinnest extent of
# its in-hand camera points < 0.35 x the middle extent) the stand-up rotation now also turns the object's thin axis (the
# face normal, from the camera points stored at grasp time) to face the robot, square to the support, with the side the
# palm is on toward the robot (so the hand stays in front of the object, not between it and the support); of the two
# rotations that satisfy this (which end down) the smaller reachable one is used.  Everything else
# (target point, approach, release, returns) is the parent's.  A non-flat object takes the parent's rotation unchanged.
# Camera / proprioception only; no task names, no simulator truth, no checker thresholds.
_LeanFaceParent = Harness
LEAN_FLAT = 0.35


class Harness(_LeanFaceParent):
    def _lean_face(self, arm):
        """flat held object: (face normal, long axis) in the world from the in-hand camera points, else None."""
        h = self.hold.get(arm) or {}
        P_ = h.get("pts_h")
        if P_ is None or len(P_) < 30:
            return None
        P_ = np.asarray(P_, float)
        ev, evec = np.linalg.eigh(np.cov((P_ - P_.mean(0)).T))
        sd = np.sqrt(np.maximum(ev, 0.0))
        if not sd[0] < LEAN_FLAT * sd[1]:
            return None
        Rh = Gm.mat(self.A.hand_pose(arm))[:3, :3]
        return Rh @ evec[:, 0], Rh @ np.asarray(h["u"], float)

    def _place_lean(self, arm, obj, ref, lean_deg=None):
        face = self._lean_face(arm) if arm in self.hold else None
        if face is None:
            return _LeanFaceParent._place_lean(self, arm, obj, ref, lean_deg)
        lean_deg = lean_deg or self.LEAN_DEG
        from transforms3d.quaternions import mat2quat
        r = self._ref_obs(ref)
        if r is None:
            raise RuntimeError(f"cannot see {ref}; locate it with a visual description")
        est = self.held_est(arm)
        if est is None:
            return {arm: {"code": "NOT_SEEN", "why": f"no in-hand estimate of the held {obj}"}}
        ow, obot = est
        c = (np.asarray(r["lo"][:2], float) + np.asarray(r["hi"][:2], float)) / 200   # footprint box centre (world, m)
        T = np.r_[c[0], c[1] + self.LEAN_DY, self.world(r["hi"])[2] + self.LEAN_DZ]   # object centre near the top of the support
        tcp = self.tcp(arm)
        hp = self.A.hand_pose(arm)
        q0 = hp[3:]
        nw, u3 = face
        palm = -Gm.mat(hp)[:3, 0]                              # from the fingertips back toward the palm (world)
        choice = None
        for ld in lean_deg:
            v = np.array([0.0, np.sin(np.radians(ld)), np.cos(np.radians(ld))])     # long axis up, top leaning away from the robot
            f0 = np.array([0.0, -np.cos(np.radians(ld)), np.sin(np.radians(ld))])   # face toward the robot, square to the support
            B = np.c_[v, f0, np.cross(v, f0)]
            rots = []
            for sgn in (1.0, -1.0):
                a1 = G.unit(sgn * u3)
                a2 = nw - (nw @ a1) * a1
                if float(a2 @ palm) < 0:                                            # the palm side of the object faces the robot
                    a2 = -a2
                a2 = G.unit(a2)
                rots.append(B @ np.c_[a1, a2, np.cross(a1, a2)].T)
            rots.sort(key=lambda M: -float(np.trace(M)))                            # smallest rotation first
            for Rm in rots:
                q = G.quat_mul(np.asarray(mat2quat(Rm)), q0)
                off = Rm @ (ow - tcp)
                dest = T - off
                app = dest + np.array([0.0, -0.06, 0.05])
                if self.feasible(arm, [self.A.hand_from_tcp(dest, q)]) and self.feasible(arm, [self.A.hand_from_tcp(app, q)]):
                    choice = (ld, q, dest, app)
                    break
            if choice:
                break
        if choice is None:
            self.trace.append({"lean_face": "no reachable square pose; parent rotation"})
            return _LeanFaceParent._place_lean(self, arm, obj, ref, lean_deg)
        ld, q, dest, app = choice
        self.trace.append({"lean_face": {"lean_deg": ld}})
        hi_z = max(tcp[2], app[2] + 0.06)
        codes = self.run_seq({arm: [self.move_tcp(arm, np.r_[tcp[:2], hi_z]), self.move_tcp(arm, np.r_[app[:2], hi_z], q),
                                    self.move_tcp(arm, app, q), self.move_tcp(arm, dest, q)]})
        self.trace.append({"place_lean": codes[arm], "lean_deg": ld, "target_cm": self.cm(T), "dest": self.cm(dest), "tcp": self.cm(self.tcp(arm))})
        out = {"code": "OK" if codes[arm] != "PLAN_FAIL" else "PLAN_FAIL", "target": "stood up leaning against the support",
               "target_cm": self.cm(T), "xy_corrections_cm": [], "turned_deg": f"stand up, lean {ld:.0f}"}
        if self.LEAN_PRESS:
            self.run_seq({arm: [self.move_tcp(arm, self.tcp(arm) - [0, 0, self.LEAN_PRESS], q)]})
        self.run_seq({arm: [{"type": "grip", "val": 1.0}]})
        t = self.tcp(arm)
        back = -Gm.mat(self.A.hand_pose(arm))[:3, 0]          # straight out along the fingers, then up
        self.run_seq({arm: [self.move_tcp(arm, t + 0.05 * back), self.move_tcp(arm, t + 0.06 * back + [0, 0, 0.06])]})
        self.holding.pop(arm, None)
        self.hold.pop(arm, None)
        o3 = self.see(obj)
        if o3 is None:
            out.update(code="OFF_TARGET", off_target={"dx_cm": None, "dy_cm": None, "dyaw_deg": None},
                       why="released, but the cameras no longer see the object at the support; it may have fallen")
            return {arm: out}
        oc = np.asarray(G.body_center(o3["pts"]), float)[:2]
        d = oc - T[:2]
        upright = not (float(o3["lo"][2]) < 1.5 and float(o3["size"][2]) < 5.0)     # lying flat on the table = fell
        out["object_now"] = {"center": r1(o3["center"]), "bottom_z": round(float(o3["lo"][2]), 1)}
        out["placed_offset_cm"] = r1(d * 100)
        if not upright or float(np.linalg.norm(d)) > 0.06:
            out.update(code="OFF_TARGET", off_target={"dx_cm": round(float(d[0]) * 100, 1), "dy_cm": round(float(d[1]) * 100, 1), "dyaw_deg": None},
                       why="released, but the object is not standing at the support (" + ("fell over" if not upright else "off to the side") + ")")
        return {arm: out}


# (pour over)
# a held container whose UPRIGHT pose above a container standing beside / below the table is out of reach can
# often still be tipped straight into a POURING pose above it (the RoboTwin expert reaches only the tipped pose, in one
# joint-space move). Gated: (1) a carry with relation "over" that returns UNREACHABLE for a container whose opening rim is
# below the table top gets one extra field "pour_option" (code and every other field unchanged); (2) pour(..., over=<ref>)
# runs the new path. Without over= pour is the parent's; every other tool is the parent's.
# shake / return_upright kept (return_upright+over refused before any motion); every later
# motion is checked (a failure stops and returns PARTIAL with the last estimated pose); the mouth is re-estimated after every
# step and must stay inside the opening; the current opening is preferred and an earlier one is used only while it is still
# consistent (same held object, no other tool since, current view absent or overlapping it).
# Return text: no "moved into / reached" wording unless the
# first tipped move returned OK; tipped_deg labelled as the planned tilt; trace keys of cache/lifecycle housekeeping that can run
# outside a triggering call use their own prefix.
from transforms3d.axangles import axangle2mat as _po_axangle  # noqa: E402
from transforms3d.quaternions import mat2quat as _po_m2q  # noqa: E402
_PourParent = Harness
_POUR_DIRS = {"front": [0, -1.0, 0], "back": [0, 1.0, 0], "left": [-1.0, 0, 0], "right": [1.0, 0, 0]}
_POUR_BUDGET_S = 60.0
_POUR_KEEP = ("carry", "pour", "locate", "lift", "move", "home", "verify")   # an earlier opening stays valid only across calls that move
# nothing but the arm itself (any other tool may move the container or release the held object)


def _po_area(o):
    d = np.asarray(o["box_cm"][1], float) - np.asarray(o["box_cm"][0], float)
    return float(max(d[0], 0.0) * max(d[1], 0.0))


def _po_overlap(a, b):
    """True when the two measured boxes (cm, xy) overlap."""
    alo, ahi = np.asarray(a["box_cm"][0], float), np.asarray(a["box_cm"][1], float)
    blo, bhi = np.asarray(b["box_cm"][0], float), np.asarray(b["box_cm"][1], float)
    return bool(np.all(np.minimum(ahi[:2], bhi[:2]) > np.maximum(alo[:2], blo[:2])))


class Harness(_PourParent):
    def call(self, name, a):
        if name not in _POUR_KEEP and self.__dict__.get("_po_open"):
            self.__dict__["_po_open"] = {}               # any other tool may have moved things; forget earlier openings
            self.trace.append({"pour_life_open_cleared": name})   # housekeeping, not a trigger
        if name == "pour" and isinstance(a, dict) and a.get("over") not in (None, ""):
            self._po_over = str(a["over"])
            try:
                return _PourParent.call(self, name, {k: v for k, v in a.items() if k != "over"})
            finally:
                self._po_over = None
        n0 = len(self.trace)
        r = _PourParent.call(self, name, a)
        if name == "carry":
            try:
                self._po_carry_note(a, r, n0)
            except Exception as e:  # noqa: BLE001
                self.trace.append({"pour_life_error": repr(e)[:200]})   # housekeeping, not a trigger
        return r

    def _po_carry_note(self, a, r, n0):
        arm, obj, ref = a.get("arm"), a.get("object"), a.get("ref")
        out = r.get(arm) if isinstance(r, dict) else None
        if not isinstance(out, dict) or out.get("code") != "UNREACHABLE":
            return
        if str(a.get("relation") or "").lower() != "over" or ref in (None, "", "table") or self.holding.get(arm) != obj:
            return
        rims = [t["opening"].get("rim_cm") for t in self.trace[n0:] if isinstance(t, dict) and isinstance(t.get("opening"), dict)]
        rims = [x for x in rims if x is not None]
        if not rims or max(rims) >= 0.0:                 # only a container whose opening is below the table top
            return
        self._po_h = float(a.get("height_cm", 10) or 10)
        ops = [t["opening"] for t in self.trace[n0:] if isinstance(t, dict) and isinstance(t.get("opening"), dict)
               and t["opening"].get("rim_cm") is not None and t["opening"]["rim_cm"] < 0.0]
        if ops:                                          # keep the widest opening measured for this container in this carry
            best = max(ops, key=_po_area)
            store = self.__dict__.setdefault("_po_open", {})
            prev_b = store.get(ref)
            if prev_b is not None and (prev_b.get("exd4_arm") != arm or prev_b.get("exd4_obj") != obj or not _po_overlap(prev_b, best)):
                prev_b = None                            # a different hold or a box elsewhere replaces the earlier one
            if prev_b is None or _po_area(best) > _po_area(prev_b):
                store[ref] = dict(best, exd4_arm=arm, exd4_obj=obj)
        out["pour_option"] = (f"if the aim is to pour the contents of {obj} into {ref}: pour(arm=\"{arm}\", direction=..., degrees=..., "
                              f"over=\"{ref}\") moves {obj} straight into a tipped pouring pose above {ref} (holding it upright above "
                              f"{ref} is out of reach for this arm; a tipped pose may still be reachable)")
        self.trace.append({"pour_carry_note": {"ref": ref, "rim_cm": max(rims)}})

    def pour(self, arm, direction="left", degrees=100.0, shake=True, back=False):
        ref = getattr(self, "_po_over", None)
        if not ref:
            return _PourParent.pour(self, arm, direction, degrees, shake, back)
        r = self._po_pour_over(arm, ref, direction, float(degrees), bool(shake), bool(back))
        self.trace.append({"pour_pour_result": {k: (v.get("code") if isinstance(v, dict) else v) for k, v in r.items()}})
        return r

    def _po_mouth(self, arm, up_h, half):
        """estimated mouth point (world) from the gripper pose and the in-hand pose at grasp time."""
        return np.asarray(self.held_est(arm)[0], float) + self._po_up_now(arm, up_h) * half

    def _po_pour_over(self, arm, ref, direction, degrees, shake, back):
        obj = self.holding.get(arm)
        est = self.held_est(arm)
        if not obj or est is None:
            return {arm: {"code": "NOT_HOLDING", "why": f"the {arm} arm holds nothing to pour"}}
        dname = str(direction).lower()
        if dname not in _POUR_DIRS:
            return {arm: {"code": "BAD_ARG", "why": "direction must be left, right, front or back"}}
        if ref == obj or ref in (None, "table"):
            return {arm: {"code": "BAD_ARG", "why": "over must name the container to pour into"}}
        if back:                                         # refused before any motion (no upright return is planned on this path)
            return {arm: {"code": "BAD_ARG", "why": "return_upright=true is not available together with over; nothing was moved. "
                                                    "Call pour with over and return_upright=false (it stays tipped), or without over",
                          "still_holding": obj}}
        self._place_arm = arm
        n0 = len(self.trace)
        prev = getattr(self, "_primitive_scope", False)
        self._primitive_scope = True
        why_ns = None
        try:
            _BASE.Harness._opening_target(self, obj, ref, "over")
        except RuntimeError as e:
            why_ns = str(e)
        finally:
            self._primitive_scope = prev
        ops = [t["opening"] for t in self.trace[n0:] if isinstance(t, dict) and isinstance(t.get("opening"), dict)]
        hfp = (self.hold.get(arm) or {}).get("fp")
        need = 0.5 * float(min(hfp)) if hfp is not None else 0.0    # an opening much narrower than the held object is not the container

        def ok_(o):
            return (o.get("rim_cm") is not None and o["rim_cm"] < 0.0
                    and min(np.asarray(o["box_cm"][1], float) - np.asarray(o["box_cm"][0], float)) >= need)
        cur = ops[-1] if ops else None
        old = (self.__dict__.get("_po_open") or {}).get(ref)
        if old is not None and (old.get("exd4_arm") != arm or old.get("exd4_obj") != obj):
            old = None                                   # an earlier opening is only trusted for the same continuous hold
        op, op_src = None, None
        if cur is not None and ok_(cur):                 # the current measurement first
            op, op_src = cur, "measured now"
        elif old is not None and ok_(old):
            if cur is None:
                op, op_src = old, "measured earlier (not measured in the current view)"
            elif _po_overlap(cur, old):
                op, op_src = old, "measured earlier (the current view sees a smaller part of it at the same place)"
            else:
                self.trace.append({"pour_opening_conflict": {"now": cur.get("box_cm"), "earlier": old.get("box_cm")}})
                return {arm: {"code": "NOT_SEEN", "why": f"the opening of {ref} measured now {cur.get('box_cm')} cm does not match the one "
                                                        f"measured earlier {old.get('box_cm')} cm; nothing was moved; locate {ref} again",
                              "still_holding": obj}}
        if op is None:
            return {arm: {"code": "NOT_SEEN", "why": why_ns or f"cannot find an opening of {ref} below the table top that is wider than "
                                                        f"{obj}; locate it with a visual description"}}
        self.trace.append({"pour_opening": {"source": op_src, "box_cm": op["box_cm"], "rim_cm": op["rim_cm"],
                                            "earlier_box_cm": None if old is None else old.get("box_cm")}})
        lo, hi = np.asarray(op["box_cm"][0], float) / 100, np.asarray(op["box_cm"][1], float) / 100
        lo2, hi2 = lo + 0.03, hi - 0.03                  # the mouth stays inside the measured footprint
        for k in range(2):
            if lo2[k] > hi2[k]:
                lo2[k] = hi2[k] = (lo[k] + hi[k]) / 2

        def inside(p):                                   # estimated mouth over the measured opening footprint
            return bool(np.all(p[:2] >= lo - 1e-6) and np.all(p[:2] <= hi + 1e-6))
        ctr = (lo2 + hi2) / 2
        gx = np.linspace(lo2[0], hi2[0], max(1, int((hi2[0] - lo2[0]) / 0.04) + 1))
        gy = np.linspace(lo2[1], hi2[1], max(1, int((hi2[1] - lo2[1]) / 0.04) + 1))
        xys = [ctr] + sorted([np.array([x, y]) for x in gx for y in gy], key=lambda p: float(np.linalg.norm(p - ctr)))[:6]
        c, _ = est
        t0, q0 = self.tcp(arm), np.asarray(self.A.hand_pose(arm)[3:], float)
        up_h = self._po_up_hand(arm)                     # the container's own up axis in the hand frame (upright when grasped)
        half = float(np.asarray(self.hold[arm]["size"], float)[2]) / 200
        h0 = max(30.0, float(getattr(self, "_po_h", 30.0) or 30.0))
        heights = (h0 + 8.0, h0 + 4.0, h0)               # held-object centre above the table: a little higher than a plain carry
        d_req = float(degrees)
        d0 = float(np.clip(degrees, 60.0, 150.0))
        thetas = [x for x in (d0 + 15.0, d0 + 5.0, d0 - 5.0, d0 - 15.0) if 60.0 <= x <= 150.0]   # past the request first: a held container
        # sags ~8 deg in the fingers; loose contents stayed in at ~100-112 deg true tilt and fell at once at >=117 (bench)
        dirs = [dname] + [x for x in ("front", "left", "back", "right") if x != dname]
        tried, execs, t_start = 0, 0, time.time()
        order = []                                       # (direction, tilt) actually checked, in order
        for th in thetas:
            for dn in dirs:
                order.append(f"{dn} {th:.0f}")
                Rk = self._po_rot(arm, up_h, dn, th)     # final tilt measured from upright, not from the current pose
                q = G.quat_mul(np.asarray(_po_m2q(Rk)), q0)
                mouth = (Rk @ self._po_up_now(arm, up_h)) * half
                for hc in heights:
                    for xy in xys:
                        if time.time() - t_start > _POUR_BUDGET_S:
                            break
                        cn = np.r_[xy - mouth[:2], self.tz + hc / 100]
                        tcp = cn + Rk @ (t0 - c)
                        hand = self.A.hand_from_tcp(tcp, q)
                        tried += 1
                        if not self.feasible(arm, [hand]):
                            continue
                        execs += 1
                        code = self.run_seq({arm: [{"type": "move", "pose": [float(v) for v in hand]}]})[arm]
                        self.trace.append({"pour_pour_over": {"dir": dn, "deg": th, "centre_h_cm": hc, "mouth_cm": r1(xy * 100), "code": code,
                                                              "tried": tried, "tcp_cm": self.cm(self.tcp(arm))}})
                        if code == "PLAN_FAIL":
                            if execs >= 3:
                                return {arm: {"code": "PLAN_FAIL", "why": f"tipped poses above {ref} plan individually but the motion failed {execs} times",
                                              "still_holding": obj}}
                            t0, q0 = self.tcp(arm), np.asarray(self.A.hand_pose(arm)[3:], float)
                            c = self.held_est(arm)[0]
                            continue
                        return self._po_after_main(arm, ref, obj, op, op_src, code, dn, th, dname, d_req, d0, xy, up_h, half,
                                                   inside, shake, order)
        return {arm: {"code": "UNREACHABLE", "why": f"no reachable tipped pose of {obj} above {ref} ({tried} poses checked: "
                                                     f"{'/'.join(dirs)}, {thetas[-1]:.0f}-{thetas[0]:.0f} deg, centre {heights[-1]:.0f}-{heights[0]:.0f} cm above the table)",
                      "still_holding": obj}}

    def _po_after_main(self, arm, ref, obj, op, op_src, code, dn, th, dname, d_req, d0, xy, up_h, half, inside, shake, order):
        """everything after the first tipped move; each motion is checked and the mouth must stay over the opening."""
        steps = [{"step": "tip", "deg": th, "code": code}]
        fail = None if code == "OK" else f"the move into the tipped pose returned {code}"
        m_now = self._po_mouth(arm, up_h, half)
        if fail is None and not inside(m_now):
            fail = f"after the tipped move the mouth is estimated at {r1(m_now[:2] * 100)} cm, outside the opening"
        extra, skipped = None, []
        if fail is None:
            for th2 in (th + 25.0, th + 15.0):           # then tip further where reachable: contents caught at the gripped rim
                if th2 > 160.0:                          # (Flash smoke: 2-3 of 5 stayed at the fingers at ~122 deg) slide out
                    continue
                R2 = self._po_rot(arm, up_h, dn, th2)
                q2 = G.quat_mul(np.asarray(_po_m2q(R2)), np.asarray(self.A.hand_pose(arm)[3:], float))
                c2 = np.asarray(self.held_est(arm)[0], float)
                m2 = c2 + (R2 @ self._po_up_now(arm, up_h)) * half       # mouth after this extra tip about the held centre
                if not inside(m2):
                    skipped.append(f"{th2:.0f} deg (mouth would leave the opening: {r1(m2[:2] * 100)} cm)")
                    continue
                tcp2 = c2 + R2 @ (self.tcp(arm) - c2)
                hand2 = self.A.hand_from_tcp(tcp2, q2)
                if not self.feasible(arm, [hand2]):
                    skipped.append(f"{th2:.0f} deg (not reachable)")
                    continue
                c2_ = self.run_seq({arm: [{"type": "move", "pose": [float(v) for v in hand2]}]})[arm]
                self.trace.append({"pour_extra_tip": {"deg": th2, "code": c2_}})
                steps.append({"step": "extra tip", "deg": th2, "code": c2_})
                if c2_ != "OK":
                    fail = f"the extra tip to {th2:.0f} deg returned {c2_}"
                    break
                m_now = self._po_mouth(arm, up_h, half)
                if not inside(m_now):
                    fail = f"after the extra tip the mouth is estimated at {r1(m_now[:2] * 100)} cm, outside the opening"
                    break
                extra = th2
                break
        jogs = 0
        if fail is None and shake:
            hold = self.tcp(arm)
            for jog in ([0, -0.02, -0.015], [0, 0.02, -0.015], [0, 0, -0.015]):   # a jog that keeps the mouth over the opening
                if inside(self._po_mouth(arm, up_h, half) + np.asarray(jog)):
                    break
            else:
                jog = None
            if jog is None:
                skipped.append("shake (no jog keeps the mouth over the opening)")
            else:
                for _ in range(8):                       # hold the pose with small jogs (as the demos do) so loose contents can fall out
                    for p_ in (hold + jog, hold):
                        cj = self.run_seq({arm: [self.move_tcp(arm, p_)]})[arm]
                        if cj != "OK":
                            fail = f"shake jog {jogs + 1} of 16 returned {cj}"
                            steps.append({"step": "shake", "jogs_ok": jogs, "code": cj})
                            break
                        jogs += 1
                        m_now = self._po_mouth(arm, up_h, half)
                        if not inside(m_now):
                            fail = f"during the shake the mouth is estimated at {r1(m_now[:2] * 100)} cm, outside the opening"
                            break
                    if fail is not None:
                        break
                if fail is None:
                    steps.append({"step": "shake", "jogs_ok": jogs, "code": "OK"})
        un = self._po_up_now(arm, up_h)
        tilt = float(np.degrees(np.arccos(np.clip(un[2], -1.0, 1.0)))) if up_h is not None else None
        m_now = self._po_mouth(arm, up_h, half)
        out = {"code": "OK" if fail is None else ("PARTIAL" if code == "OK" else code),
               "tipped_deg": extra if extra is not None else th, "direction": dn, "over": ref,
               "tilt_est_deg": None if tilt is None else round(tilt), "tilt_source": "gripper pose and the in-hand pose at grasp time (not seen)",
               "held_bottom_cm": round((self.held_est(arm)[1] - self.tz) * 100, 1),
               "mouth_target_cm": r1(xy * 100), "mouth_est_cm": r1(m_now[:2] * 100),
               "mouth_source": "gripper pose and the in-hand pose at grasp time (not seen)", "mouth_inside_opening": inside(m_now),
               "opening_box_cm": op["box_cm"], "opening_source": op_src, "shaken": jogs if shake else "not requested",
               "steps": steps, "still_holding": obj, "upright_again": False,
               "tipped_deg_source": "planned tilt from upright of the last executed step (tilt_est_deg is the estimate)",
               "note": (f"{obj} was moved straight into a tipped pose above {ref} (not carried upright first); whether the "
                        "contents fell out is not measured; judge it from the images") if code == "OK" else
                       (f"a move of {obj} straight into a tipped pose above {ref} was attempted and returned {code}; whether it got there "
                        "is not confirmed; whether any contents fell out is not measured; judge it from the images")}
        if fail is not None:
            out["why"] = fail + "; stopped there"
        if skipped:
            out["skipped"] = skipped
        tries = order[:order.index(f"{dn} {th:.0f}") + 1] if f"{dn} {th:.0f}" in order else order
        if d0 != d_req:
            out["degrees_limited"] = f"requested {d_req:.0f} deg; this path uses {d0:.0f} deg as the reference (range 60-150)"
        if dn != dname or out["tipped_deg"] != d0:
            if code == "OK":
                out["changed"] = (f"tipped {dn} to {out['tipped_deg']:.0f} deg from upright; requested {dname} {d0:.0f} deg. Poses checked in "
                                  f"order before this one was reached: {', '.join(tries)} (the exact requested pose is not tried on its own)")
            else:
                out["changed"] = (f"attempted {dn} {out['tipped_deg']:.0f} deg from upright (planned, not confirmed); requested {dname} {d0:.0f} deg. "
                                  f"Poses checked in order before this attempt: {', '.join(tries)} (the exact requested pose is not tried on its own)")
        return {arm: out}

    def _po_up_hand(self, arm):
        h = self.hold.get(arm) or {}
        if "c" not in h or "b" not in h:
            return None
        v = np.asarray(h["c"], float) - np.asarray(h["b"], float)      # centre minus bottom point: vertical when grasped
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-3 else None

    def _po_up_now(self, arm, up_h):
        if up_h is None:
            return np.array([0.0, 0.0, 1.0])
        return Gm.mat(self.A.hand_pose(arm))[:3, :3] @ up_h

    def _po_rot(self, arm, up_h, dn, th):
        """world rotation taking the container's current up axis to "tilted th degrees from upright toward dn"."""
        d = np.asarray(_POUR_DIRS[dn], float)
        tgt = np.cos(np.radians(th)) * np.array([0, 0, 1.0]) + np.sin(np.radians(th)) * d
        u = self._po_up_now(arm, up_h)
        ax = np.cross(u, tgt)
        n = float(np.linalg.norm(ax))
        ang = float(np.arccos(np.clip(u @ tgt, -1.0, 1.0)))
        if n < 1e-6:
            return np.eye(3) if ang < 1e-3 else _po_axangle(G.unit(np.cross([0, 0, 1.0], d)), ang)
        return _po_axangle(ax / n, ang)


if __name__ == "__main__":
    main()
