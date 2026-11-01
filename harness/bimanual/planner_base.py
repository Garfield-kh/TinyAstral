#!/usr/bin/env python
"""Base layer of the RoboTwin harness H*: the tool bodies that planner.py builds on (not run directly).

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
Perception: SAM3 + native head / wrist RGB-D only (perception_base.py). No task names, no asset contact / functional points,
no simulator object poses are used by any tool. Simulator success is never shown to the planner. Privileged adapter
queries (grasp/place plans, object poses, boxes, joints, ...) are sealed (_seal_privileged): calling them raises."""
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
import geometry as G  # noqa: E402  (planner.py uses the same geometry module)
import perception_base as PC  # noqa: E402
from adapter_robotwin import RoboTwinAdapter  # noqa: E402

from request_trace import record_request  # noqa: E402

M = R.M
P = _paths.OUT
SEEDS = _paths.SEEDCACHE
CAP, WALL, MAX_REJECT, KEEP_IMG_TURNS = 30, 900, 2, 2
MEMDIR = _paths.MEMORY   # note sets
_MEMDIR_AT = _paths.MEMORY     # note sets
MEMSET = "robotwin"
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
- place(arm, object, ref, relation="on", xy=null, align=false, turn_deg=null) -> carry the HELD object to the reference and release it. align=true turns the object so its long side is parallel to the long side of ref (use it when the object must line up with a pad, mat, tray or slot). turn_deg turns the held object about the vertical axis by that many degrees before putting it down (positive = counter-clockwise seen from above); judge the needed turn from the images. With a reference object, xy=[x, y] (cm, table frame) picks the point on / in ref (e.g. one half of a tray); without xy the tool picks the centre or a free part. relation: "on" (on top of ref, e.g. stacking or onto a pad/mat/coaster), "in" (inside a container: basket, pan, bowl, bin), "left_of" / "right_of" / "front_of" / "behind" (on the table beside ref on that side, a clear gap of about 5 cm, same row), or ref="table" with xy=[x, y] cm, or "drop" (let it fall in from just above a container's opening, also a bin standing beside the table below its top). The tool centres the object over ref with camera feedback, lowers it and opens the gripper. relation="against" stands the held object up and leans it against a tall support (a stand or rack); this is chosen automatically when ref is tall and its top is too small to hold the object. After releasing, the tool measures where the object ended up: if it is off the target it re-grasps and places once more, and if it is still off it returns code OFF_TARGET with the measured offset (dx_cm, dy_cm, dyaw_deg) instead of OK.
- carry(arm, object, ref, relation="on", height_cm=10, turn_deg=null) -> like place but keeps holding the object height_cm above the target. relation="over": above a container's opening (also a bin beside the table), e.g. before pour.
  place and carry also take face="front" / "back" / "left" / "right": the tool turns the held object so that its flat face (its thinnest side, e.g. a sign, a board, a fan) faces that direction ("front" = toward the robot). If the images show the wrong side would face that way, also pass turn_deg=180. Better: also pass face_now="front" / "back" / "left" / "right" = the direction that flat face (the side that must end up facing) points to now, as seen in the images; the tool then turns that side to face (up to a full half turn) instead of guessing which of the two flat sides is meant. Whenever an object must end up facing a direction (facing the robot, facing left ...), use face instead of estimating turn_deg yourself; a task note that says "turn" or "rotate" to face something means face.
  place also takes low_end="left" / "right" / "front" / "back": a long object whose ends differ in height (e.g. a shoe: the lower end is the toe) is turned so its lower end points there. After face= / low_end= the tool re-measures the orientation once released and corrects a remaining turn once (orientation_check).
  place and carry also take held_part="left end" / "right end" / "front end" / "back end": that end of the held object (as it points now, see the images) is brought over the target instead of its middle (e.g. the head of a held tool onto a target).
- present(arm, xy=[x, y], height_cm=15, turn_deg=null) -> hold the HELD object up in the air, its centre over table point xy (cm), its bottom height_cm above the table, upright, still held (nearest reachable point if needed). Two arms at once: bimanual with two present calls.
- pour(arm, direction="left", degrees=100, shake=true, return_upright=false) -> tip the HELD object (a container) about a horizontal axis so its top tilts toward direction (left/right/front/back), optionally shake it a little, and keep holding it (tilted unless return_upright=true). Carry it over the destination first.
- handover(object, to_arm, mode="table") -> passes the object held by the other arm to to_arm. mode "table": puts it down between the arms, the other arm backs off, to_arm grasps it. mode "air" (only if a task note asks for it or the object would fall over when put down): to_arm takes it in the air (falls back to "table"). Result OK = to_arm holds it.
- lift(arm, height_cm) -> raise that gripper (and what it holds) to this height above the table (if it is already that high it stays; use move dz to go further up). arm may be "both" to lift both grippers together (for an object held with two hands).
- articulate(arm="auto", object, motion="open", degrees=80) -> opens (or closes) a hinged front door (microwave, cabinet): the tool finds the handle and hinge from the cameras, picks the arm on the hinge side when arm="auto" (a named arm is always used as given), grasps the handle and swings it along the door arc, then releases. For a low flat object with a lid on a horizontal hinge it pinches the lid edge, lifts the lid about the hinge and keeps holding it. If the swing is SHORT, look at the images and repeat with arm="auto" or the arm named in the hint. A drawer: motion="pull" slides the lowest drawer straight out (degrees = cm, default 15) and reports drawer_inside_xy_cm; "push_in" closes it.
- press(arm, object, direction=null) -> press the object with the closed gripper and come back (the gripper opens again afterwards). direction "down" presses its top (buttons, staplers); "into_face" pushes a thin upright plate (a rocker switch) along its own normal on its raised half; "from_front" / "from_left" / "from_right" / "from_back" pushes horizontally toward that side. Without direction a top press that cannot be reached falls back to tilted and horizontal pushes.
- push(arm, object, direction, distance_cm) -> push the object along a direction with the closed gripper.
- move(arm, dx, dy, dz) -> move that gripper by an offset in cm (anything held moves with it).
- release(arm) -> open that gripper and back off upward.
- home(arm) -> return that arm to its raised observation pose (do this to get an arm out of the way).
- bimanual(calls: [call, call]) -> run two calls at the same time, one per arm (each {"name", "arguments"} of grasp/place/lift/move/release/home). Use it to grasp one large object with both hands (e.g. two grasps with part "handle" or "left end"/"right end"), then lift(arm="both").
- verify(checks) / done(checks) -> checks are [[predicate, args...], ...] evaluated from the cameras and gripper sensors. Predicates: on(a, b), in(a, b), on_table(a), left_of/right_of/front_of/behind(a, b), near(a, b), same_row(a, b), at(a, x, y, radius_cm), upright(a), height(a, cm) (bottom of a above the table), lifted(a, cm) (held and bottom at least cm high), holding(arm, a), released(arm), pressed(a). done ends the episode if all checks pass. When no predicate measures the goal, judge the images and call done(checks=[]). A check with ok=null could not be measured reliably (not seen, or an implausible measurement): judge that part from the images and leave it out of done if the images show it is fine.
- give_up(reason).

Finished steps are listed under "Done so far". Grasping such an object again, or pressing an object a second time, returns CONFIRM_NEEDED: if the images show the step is still done, leave it; otherwise repeat the call with "confirm": true.

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
                              drop_robot=self._drop and not self.holding, below_names=set(getattr(self, "below", set())))
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
        lt = self.ledger_text() if hasattr(self, "ledger_text") else ""
        return "Arms: " + "; ".join(parts) + "." + (("\n" + lt) if lt else "")

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
    def _see_clear(self, n):
        """an object the cameras cannot find may be hidden behind an empty gripper: move the empty arms to their
        observation pose and look once more (no planner call is spent)."""
        o = self.see(n)
        if o is None and not getattr(self, "_raw", False):
            free = [a for a in ARMS if not self.holding.get(a)]
            if free:
                self.trace.append({"not_seen_retry": n, "arms_cleared": free})
                self.start()
                o = self.see(n)
        return o

    def grasp_plan(self, arm, obj, part=None, exclude=(), own_side=False):
        o = self._see_clear(obj)
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
        sz = np.asarray(o["size"], float)
        tall = sz[2] >= 10.0 and sz[2] >= 2.0 * max(sz[0], sz[1])
        zlo, zhi = self.world(o["lo"])[2], self.world(o["hi"])[2]
        # a tall thin object that is out of reach for the side grasp at 72 % height is tried again at other heights and
        # a lower tilt (second pass only: when the first pass finds a grasp, nothing changes)
        passes = [[(0.72, (90.0, 75.0, 60.0))]] + ([[(0.85, (90.0, 75.0, 60.0)), (0.55, (90.0, 75.0, 60.0)), (0.72, (45.0,)), (0.85, (45.0,)), (0.85, (25.0, 0.0))]] if tall else [[("steep", (60.0, 70.0))]])
        for pi, variants in enumerate(passes):
            tried_p = 0
            for si, s in enumerate(specs):
                vlist = variants if (tall and s.get("out") is None) else ([None] if pi == 0 else (variants if s.get("out") is None else []))
                for v in vlist:
                    if v is not None and v[0] == "steep":   # an object at the edge of reach: approaches leaning further along the reach
                        s2 = s
                        cands = G.candidates(s, arm, tilts=v[1])
                    elif v is not None:            # tall thin object: side grasp on the upper part (tilted/top grasps topple it)
                        s2 = dict(s)
                        s2["p"] = np.r_[s["p"][:2], zlo + v[0] * (zhi - zlo)]
                        cands = G.candidates(s2, arm, tilts=v[1])
                    else:
                        s2 = s
                        cands = G.candidates(s, arm)
                    for al, a, c in cands:
                        key = (si, al) if (pi == 0) else (si, al, v[0] if v else None)
                        if key in exclude:
                            continue
                        q = G.hand_quat(a, c)
                        p = s2["p"]
                        pre = p - a * PRE
                        hp, hg = self.A.hand_from_tcp(pre, q), self.A.hand_from_tcp(p, q)
                        tried += 1
                        tried_p += 1
                        if self.feasible(arm, [hp]) and self.feasible(arm, [hg]):
                            if pi:
                                self.trace.append({"tall_fallback_grasp": {"height_frac": v[0], "tilt": al}})
                            return {"arm": arm, "obj": obj, "pre": hp, "grasp": hg, "q": q, "key": key, "why": s2["why"], "obs": o, "spec_p": p,
                                    "pregrip": float(s2.get("pregrip", 1.0)),
                                    "tilt": al, "point_cm": self.cm(p), "shape": s2["shape"]}, None
                        if tried_p >= 24:
                            break
                    if pi and tried_p >= 24:
                        break
                if pi and tried_p >= 24:
                    break
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

    # completion-state ledger
    def _ledger(self):
        if not hasattr(self, "_led"):
            self._led = {}
        return self._led

    def _support(self, obj, what):
        """a sub-goal the tools measured as achieved (object put on / in a reference and measured there; a press that
        measurably changed the object). Kept until a gripper touches the object again or a check measures it false."""
        self._ledger()[obj] = {"state": "supported", "what": what, "call": int(getattr(self, "k", -1))}

    def _refute(self, obj, why):
        led = self._ledger()
        if obj in led:
            led[obj] = {**led[obj], "state": "refuted", "why": why}

    def _guard(self, obj, action, confirm):
        """undoing a supported sub-goal needs an explicit confirm (one extra call), so a single occlusion or a jumpy
        re-detection does not make the planner destroy a finished step. Not a done gate: nothing blocks done."""
        e = self._ledger().get(obj)
        if confirm or e is None or e.get("state") != "supported" or e.get("warned"):
            return None
        e["warned"] = True
        return {"code": "CONFIRM_NEEDED", "why": f"{obj}: {e['what']} (measured at call {e['call']}) and nothing has moved it since; "
                f"{action} it again may undo that. Check the images: if that step still looks done, leave {obj} alone and go on "
                f"(or call done). If the images show it is NOT done, repeat this call with confirm=true."}

    def ledger_text(self):
        led = [f"{o}: {e['what']}" for o, e in self._ledger().items() if e.get("state") == "supported"]
        return ("Done so far (measured by the tools): " + "; ".join(led) + ".") if led else ""

    def grasp(self, arm, obj, part=None, confirm=False):
        gd = self._guard(obj, "grasping", confirm)
        if gd is not None:
            return {arm: gd}
        self._ledger().pop(obj, None)
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
        if rel in ("drop", "drop_in", "drop_into", "over", "above_opening"):
            return self._opening_target(obj, ref, rel)
        for ha, ho in self.holding.items():           # reference held by the other hand: use its in-hand geometry
            if ho == ref and ha in self.hold and rel in ("in", "into", "inside", "on", "onto", "on_top_of", "at"):
                hp = self.A.hand_pose(ha)
                fl = self.A.tcp_point(hp) + Gm.mat(hp)[:3, :3] @ self.hold[ha]["floor"]
                self._raw = True                                  # re-measure the held container with the cameras now (it may
                try:                                              # have swung about its handle since the grasp)
                    rc = self.perceive([ref]).get(ref)
                finally:
                    self._raw = False
                if rc is not None and len(rc["pts"]) >= 40:
                    cc = np.asarray(G.body_center(rc["pts"]), float)[:2]
                    fz = G.floor_z(rc["pts"], cc)
                    d = float(np.linalg.norm(cc - fl[:2]))
                    self.trace.append({"held_ref_remeasure_cm": r1(cc * 100), "rigid_cm": r1(fl[:2] * 100), "diff_cm": round(d * 100, 1)})
                    if d < 0.10:
                        return cc, max(fl[2], fz if fz is not None else fl[2]) + 0.015, "held container floor (measured now)"
                return fl[:2], fl[2] + 0.015, "held container floor"
        r = self._see_clear(ref)
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

    def _see_below(self, n):
        """(re-)detect n allowing points below the table top beyond its side edges (a bin standing beside the table)."""
        if not hasattr(self, "below"):
            self.below = set()
        if n not in self.below:
            self.below.add(n)
            self.perc.pop(n, None)
            self.first.pop(n, None)
        held = [o for o in self.holding.values() if o != n]
        if not held:
            return self.see(n)
        # a held object of the same kind (a small bin in the hand) must not be taken for n: detect them together so each
        # name gets its own cluster (the held one is favoured near its in-hand position)
        self._drop = False
        try:
            res = self.perceive([n] + held)
        finally:
            self._drop = True
        o = res.get(n)
        for h in held:
            for a_, ob in self.holding.items():
                if ob == h and o is not None:
                    est = self.held_est(a_)
                    if est is not None and float(np.linalg.norm(self.world(o["center"])[:2] - est[0][:2])) < 0.08:
                        self.trace.append({"below_ref_is_held_object": h})
                        return None
        return o

    def _below_table_blob(self):
        """camera points below the table top and beyond its left / right edge (not the robot's own body in front of the
        table), grouped on a 4 cm grid; the largest connected group = a bin / box standing on the floor beside the table."""
        try:
            views = PC.capture(self.sim.rd)
        except Exception:  # noqa: BLE001
            return None
        xr = PC.table_x_extent(views, self.tz)
        if xr is None:
            return None
        P_ = np.concatenate([v["xyz"][v["valid"]] for v in views])
        sel = (P_[:, 2] < self.tz - 0.005) & (P_[:, 2] > self.tz - 0.60) & ((P_[:, 0] < xr[0] - 0.01) | (P_[:, 0] > xr[1] + 0.01)) \
            & (P_[:, 1] > -0.30) & (P_[:, 1] < 0.50) & (np.abs(P_[:, 0]) < 0.95)
        Q = P_[sel]
        if len(Q) < 200:
            return None
        cell = np.floor(Q[:, :2] / 0.04).astype(int)
        keys, inv, cnt = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        occ = {tuple(k): i for i, k in enumerate(keys)}
        seen, best = set(), None
        for i, k in enumerate(keys):
            if i in seen:
                continue
            comp, stack = [], [i]
            seen.add(i)
            while stack:
                j = stack.pop(); comp.append(j)
                kx, ky = keys[j]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        jj = occ.get((kx + dx, ky + dy))
                        if jj is not None and jj not in seen:
                            seen.add(jj); stack.append(jj)
            n = int(cnt[comp].sum())
            if best is None or n > best[0]:
                best = (n, comp)
        if best is None or best[0] < 200:
            return None
        pts = Q[np.isin(inv.ravel(), best[1])]
        if len(pts) > 4000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        lo_c, hi_c = (lo - [0, 0, self.tz]) * 100, (hi - [0, 0, self.tz]) * 100
        return {"center": (lo_c + hi_c) / 2, "size": hi_c - lo_c, "lo": lo_c, "hi": hi_c, "pts": pts}

    def _opening_target(self, obj, ref, rel):
        """a point above the opening of a container (also one standing beside / below the table): the opening is the
        ring of points near its rim height; the target is the point of that opening, shrunk by the held object's half size and
        a margin, nearest to this arm (bins beside the table are reached only near their edge); the held object's bottom goes a
        few cm above the rim."""
        r = self._see_below(ref)
        if r is not None and len(r["pts"]) >= 20 and float(np.median(np.asarray(r["pts"])[:, 2])) > self.tz - 0.02:
            blob = self._below_table_blob()               # a big container standing beside the table outranks an on-table detection
            if blob is not None and len(blob["pts"]) >= 1000:
                self.trace.append({"opening_prefers_below_table": True, "sam_center_cm": r1(r["center"])})
                r = None
        if r is None or len(r["pts"]) < 20:
            r = self._below_table_blob()                  # geometry fallback: the biggest thing standing below the table top beside it
            if r is None:
                raise RuntimeError(f"cannot see {ref}; locate it with a visual description")
            self.trace.append({"opening_from_geometry": True, "center_cm": r1(r["center"]), "size_cm": r1(r["size"])})
            self.perc[ref] = r
        pts = np.asarray(r["pts"], float)
        rim = float(np.percentile(pts[:, 2], 95))
        lo, hi = np.percentile(pts[:, :2], 3, axis=0), np.percentile(pts[:, :2], 97, axis=0)     # the whole footprint (its rim is often seen only in part)
        hfp = (self.hold.get(self._place_arm) or {}).get("fp") if self._place_arm else None
        half = (min(float(hfp[0]), float(hfp[1])) / 200) if hfp is not None else 0.03
        mg = half + 0.015
        lo2, hi2 = lo + mg, hi - mg
        for k in range(2):
            if lo2[k] > hi2[k]:
                lo2[k] = hi2[k] = (lo[k] + hi[k]) / 2
        base = np.asarray(G.ARM_BASE.get(self._place_arm, np.zeros(2)), float)
        gx = np.linspace(lo2[0], hi2[0], max(1, int((hi2[0] - lo2[0]) / 0.03) + 1))
        gy = np.linspace(lo2[1], hi2[1], max(1, int((hi2[1] - lo2[1]) / 0.03) + 1))
        cands = sorted([np.array([x, y]) for x in gx for y in gy], key=lambda p: float(np.linalg.norm(p - base)))
        k = int(getattr(self, "_open_k", 0))
        c = cands[min(k, len(cands) - 1)]
        self.trace.append({"opening": {"rim_cm": round((rim - self.tz) * 100, 1), "box_cm": [r1(lo * 100), r1(hi * 100)], "target_cm": r1(c * 100), "k": k, "n": len(cands)}})
        z = rim + (0.06 if rel.startswith("drop") else 0.10)
        return c, z, "dropped above the container opening" if rel.startswith("drop") else "held above the container opening"

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

    DIRS = {"front": [0, -1.0], "back": [0, 1.0], "left": [-1.0, 0], "right": [1.0, 0]}

    @staticmethod
    def _mar(xy):
        """minimum-area rectangle of footprint points, 2 deg search refined to 0.25 deg; returns long axis u, short axis m
        long / short extents (m)."""
        X = np.asarray(xy, float) - np.median(xy, 0)

        def area(th):
            e1 = np.array([np.cos(th), np.sin(th)]); e2 = np.array([-e1[1], e1[0]])
            a, b = X @ e1, X @ e2
            l1 = np.percentile(a, 97) - np.percentile(a, 3); l2 = np.percentile(b, 97) - np.percentile(b, 3)
            return l1 * l2, e1, e2, l1, l2
        best = min((area(t) for t in np.radians(np.arange(0, 90, 2.0))), key=lambda r: r[0])
        th0 = np.arctan2(best[1][1], best[1][0])
        best = min((area(th0 + np.radians(d)) for d in np.arange(-2.0, 2.01, 0.25)), key=lambda r: r[0])
        _, e1, e2, l1, l2 = best
        return (e1, e2, l1, l2) if l1 >= l2 else (e2, -e1, l2, l1)

    def _orient(self, P_):
        """horizontal orientation of an object from its points: over height slices
        the slice whose footprint is most elongated defines the long axis u and the flat-face normal m (sign free). low_end:
        the direction along u of the lower end (the higher points' centroid sits toward the other end)."""
        P_ = np.asarray(P_, float)
        z0, z1 = float(np.percentile(P_[:, 2], 2)), float(np.percentile(P_[:, 2], 98))
        best = None
        for name, lo, hi in (("whole", 0.0, 1.01), ("upper", 0.35, 1.01), ("top", 0.55, 1.01), ("base", 0.0, 0.30), ("middle", 0.30, 0.60)):
            Q = P_[(P_[:, 2] >= z0 + lo * (z1 - z0)) & (P_[:, 2] <= z0 + hi * (z1 - z0))]
            if len(Q) < 40:
                continue
            u, m, L, W = self._mar(Q[:, :2])
            r = L / max(W, 1e-3)
            if best is None or r > 1.15 * best["aspect"]:
                best = {"u": u, "m": m, "aspect": float(r), "slice": name, "L_cm": round(L * 100, 1), "W_cm": round(W * 100, 1)}
        if best is None:
            return None
        u = best["u"]
        c = np.median(P_[:, :2], 0)
        hi_ = P_[P_[:, 2] >= np.percentile(P_[:, 2], 80)]
        s = float((np.median(hi_[:, :2], 0) - c) @ u) if len(hi_) >= 10 else 0.0
        best["low_end"] = -np.sign(s) * u if abs(s) > 0.005 else None
        best["low_shift_cm"] = round(abs(s) * 100, 1)
        return best

    def _face_turn(self, arm, obj, face, face_now=None, low_end=None):
        """yaw (deg) that turns the held object's flat face (thinnest side) toward the world direction `face`
        (|yaw| <= 90 without face_now; full range with face_now), or (low_end=dir) turns its long axis so its lower end
        points to dir (full range, the lower end measured from the in-hand points)."""
        if arm not in self.hold or "pts_h" not in self.hold[arm]:
            return None
        o = self._orient(self._tracked(arm, obj)["pts"])
        if o is None:
            return None
        if low_end:
            d = self.DIRS.get(str(low_end).lower())
            if d is None or o["low_end"] is None:
                self.trace.append({"low_end_turn": None, "low_shift_cm": o["low_shift_cm"]})
                return None
            n = o["low_end"]
            y = (np.degrees(np.arctan2(d[1], d[0])) - np.degrees(np.arctan2(n[1], n[0])) + 180) % 360 - 180
            self.trace.append({"low_end_turn": round(float(y), 1), "low_shift_cm": o["low_shift_cm"], "slice": o["slice"], "aspect": round(o["aspect"], 2)})
            return float(y)
        d = self.DIRS.get(str(face).lower())
        if d is None:
            return None
        m = o["m"]
        cur = np.degrees(np.arctan2(m[1], m[0]))
        want = np.degrees(np.arctan2(d[1], d[0]))
        y = (want - cur + 90) % 180 - 90
        dn = self.DIRS.get(str(face_now or "").lower())
        if dn is not None:                                 # the side that points toward face_now now is the one to turn (signed, full range)
            n = np.asarray(m[:2], float) * (1.0 if float(np.dot(m[:2], dn)) >= 0 else -1.0)
            cur = np.degrees(np.arctan2(n[1], n[0]))
            y = (want - cur + 180) % 360 - 180
        self.trace.append({"face_turn": round(float(y), 1), "thin_axis_deg": round(float(cur), 1), "face_now": face_now, "face_slice": o["slice"],
                           "face_aspect": round(o["aspect"], 2)})
        return float(y)

    def _yaw_error(self, obj, face=None, low_end=None):
        """after release (no gripper in the way) measure the object's orientation again: the remaining turn (deg) that
        would bring its flat face to `face` (sign-free, within +-90) or its lower end to `low_end` (full range)."""
        o3 = self.see(obj)
        if o3 is None or len(o3["pts"]) < 60:
            return None, None
        o = self._orient(o3["pts"])
        if o is None or o["aspect"] < 1.25:
            return None, o
        if low_end:                                      # smoke: the end polarity at rest flips between views -> axis only (sign-free)
            d = self.DIRS.get(str(low_end).lower())
            if d is None:
                return None, o
            u = o["u"]
            return float((np.degrees(np.arctan2(d[1], d[0])) - np.degrees(np.arctan2(u[1], u[0])) + 90) % 180 - 90), o
        d = self.DIRS.get(str(face).lower())
        if d is None:
            return None, o
        m = o["m"]
        return float((np.degrees(np.arctan2(d[1], d[0])) - np.degrees(np.arctan2(m[1], m[0])) + 90) % 180 - 90), o

    YAW_TOL = {"face": 6.0, "low_end": 10.0}
    FIX_ASPECT = 2.5          # smoke: only a clearly elongated footprint at rest is trusted for the in-place correction

    def place(self, arm, obj, ref, relation="on", xy=None, release=True, height_cm=10.0, align=False, turn_deg=None, held_part=None, face=None, face_now=None,
              low_end=None):
        """wrapper: (1) with face= / low_end=, after the release the object's orientation is measured again from the
        cameras (no gripper in the way); if the remaining turn is above a few degrees and the object sits at the target, it is
        re-grasped where it stands and put down again turned by that amount (once); the measured error is reported.
        (2) a release measured on / in a reference object is recorded in the completion ledger."""
        r = self._place_v(arm, obj, ref, relation, xy, release, height_cm, align, turn_deg, held_part, face, face_now, low_end)
        out = r.get(arm) if isinstance(r, dict) else None
        if str(relation or "").lower() in ("drop", "drop_in", "drop_into", "over", "above_opening") and isinstance(out, dict) and out.get("code") == "UNREACHABLE":
            for k in (1, 2, 4, 6, 9):                      # other points of the opening, nearest to this arm first
                self._open_k = k
                try:
                    r = self._place_v(arm, obj, ref, relation, xy, release, height_cm, align, turn_deg, held_part, face, face_now, low_end)
                finally:
                    self._open_k = 0
                out = r.get(arm) if isinstance(r, dict) else None
                if not (isinstance(out, dict) and out.get("code") == "UNREACHABLE"):
                    break
        if release and (face or low_end) and isinstance(out, dict) and out.get("code") in ("OK", "OFF_TARGET") and not self.holding.get(arm):
            kind = "low_end" if low_end else "face"
            err, o = self._yaw_error(obj, face, low_end)
            out["orientation_check"] = {"remaining_turn_deg": None if err is None else round(err, 1),
                                        "slice": None if o is None else o["slice"], "aspect": None if o is None else round(o["aspect"], 2)}
            off = out.get("off_target") or {}
            near = out.get("code") == "OK" or (off.get("dx_cm") is not None and float(np.hypot(off["dx_cm"], off.get("dy_cm") or 0)) <= 6.0)
            lim = 45.0
            if err is not None and self.YAW_TOL[kind] < abs(err) <= lim and near and o is not None and o["aspect"] >= self.FIX_ASPECT:
                g = self.grasp(arm, obj, confirm=True)
                if (g.get(arm) or {}).get("code") == "OK":
                    r2 = self._place_once(arm, obj, ref, relation, xy, True, height_cm, False, float(err), held_part)
                    o2 = r2.get(arm) or {}
                    err2, _ = self._yaw_error(obj, face, low_end) if not self.holding.get(arm) else (None, None)
                    o2["orientation_check"] = {"first_remaining_turn_deg": round(err, 1), "corrected_by_deg": round(err, 1),
                                               "remaining_turn_deg": None if err2 is None else round(err2, 1)}
                    r, out = r2, o2
                    self.trace.append({"yaw_fix": round(err, 1), "after": None if err2 is None else round(err2, 1)})
                else:
                    out["orientation_check"]["fix"] = f"re-grasp for the turn failed: {(g.get(arm) or {}).get('code')}"
                    return r
        if release and isinstance(out, dict) and out.get("code") == "OK" and ref not in (None, "table"):
            rel = str(relation or "on").lower().replace("_", " ")
            self._support(obj, f"put {rel} {ref} and measured there" + (f", facing {face}" if face else ""))
        return r

    def _place_v(self, arm, obj, ref, relation="on", xy=None, release=True, height_cm=10.0, align=False, turn_deg=None, held_part=None, face=None, face_now=None,
                 low_end=None):
        self._forget_held_contents(arm)                    # a carried container takes its contents along
        if ref not in (None, "table"):
            self._forget_release(ref)                      # putting something onto / into ref may nudge it
        if (face or low_end) and self.holding.get(arm) == obj:          # face -> a turn about the vertical (plus the planner's extra turn_deg)
            ft = self._face_turn(arm, obj, face, face_now, low_end)
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
            g = self.grasp(arm, obj, confirm=True)
            if (g.get(arm) or {}).get("code") == "OK":
                td2 = None                                   # the object already carries the first turn; face is re-solved from the new hold
                if face or low_end:
                    td2 = self._face_turn(arm, obj, face, face if face_now else None, low_end)
                if td2 is None and (face or low_end or turn_deg is not None):
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
        hz_ = max(tcp[2], safe)
        if str(why).startswith("held above") and isinstance(yaw, str) and \
                self.feasible(arm, [self.A.hand_from_tcp(np.r_[dest[:2] + (ow[:2] - tcp[:2]) * 0, hz_], q0)]):
            # a held container is tilted only once it is over the opening (tilting it on the way spills its contents)
            codes = self.run_seq({arm: [self.move_tcp(arm, np.r_[tcp[:2], hz_]), self.move_tcp(arm, np.r_[dest[:2], hz_], q0),
                                        self.move_tcp(arm, np.r_[dest[:2], safe], q)]})
        else:
            codes = self.run_seq({arm: [self.move_tcp(arm, np.r_[tcp[:2], hz_]), self.move_tcp(arm, np.r_[tcp[:2], hz_], q),
                                        self.move_tcp(arm, np.r_[dest[:2], safe], q)]})
        self.dbg_img("over")
        self.trace.append({"place_over": codes[arm], "yaw": yaw, "tcp": self.cm(self.tcp(arm)), "dest": self.cm(dest), "obj_est": self.cm(ow), "tcp0": self.cm(tcp)})
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
        self.dbg_img("down")
        if release and ((codes[arm] == "PLAN_FAIL" and float(self.tcp(arm)[2] - dest[2]) > 0.04) or
                        (codes[arm] == "POSE_ERR" and float(self.tcp(arm)[2] - dest[2]) > 0.08)):   # a short stop (resting on what is below) still releases # never let go high above the target
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
            if str(why).startswith("dropped"):           # it may now lie in a bin below the table top
                if not hasattr(self, "below"):
                    self.below = set()
                self.below.add(obj)
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
        elif w.startswith("dropped"):                    # inside the container's footprint and below its rim
            r = self._refv(ref)
            if r is not None and r is not o3:
                lo = np.asarray(r["lo"][:2], float) / 100
                hi = np.asarray(r["hi"][:2], float) / 100
                inside = bool(np.all(oc > lo - 0.01) and np.all(oc < hi + 0.01))
                below_rim = float(o3["lo"][2]) < float(r["hi"][2])
                off = not (inside and below_rim)
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
    def handover(self, obj, to_arm, mode="table"):
        """mode="air": the giver brings the object to the middle in the air, the receiver grasps the half away from the
        giver's fingers with a side approach, then the giver lets go (falls back to the table hand-over if no receiving grasp
        is reachable). mode="table" (default): put it down between the arms and regrasp from the table."""
        frm = "left" if to_arm == "right" else "right"
        if self.holding.get(frm) != obj:
            raise ValueError(f"the {frm} arm does not hold {obj}")
        if str(mode).lower() in ("air", "mid-air", "midair", "in_air"):
            ra = self.handover_air(frm, to_arm, obj)
            if ra is not None:
                if ra.get("code") == "OK":
                    self.trace.append({"handover_air": "OK"})
                return ra
            self.trace.append({"handover_air": "no receiving grasp; via the table"})
        o = self.see(obj)
        y = float(o["center"][1]) if o is not None else 0.0
        mid = [4.0 if to_arm == "right" else -4.0, min(max(y, -10.0), -2.0)]
        r1_ = self._place_once(frm, obj, "table", "table", xy=mid)
        if r1_[frm]["code"] not in ("OK", "OFF_TARGET", "MEASURE_UNSURE"):          # the hand-over put-down point need not be exact
            return {"code": "HANDOVER_FAIL", "put_down": r1_[frm]}
        self.call("home", {"arm": frm})
        r2 = self.grasp(to_arm, obj, confirm=True)
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
        hp_g = self.A.hand_pose(frm)
        Rg_, tg_ = Gm.mat(hp_g)[:3, :3], self.A.tcp_point(hp_g)
        # receiver point: on the object's receiver side / lower half, away from the giver's fingers
        cand = []
        lowz = bot + 0.3 * hgt if hgt > 0.07 else c[2]
        for gp in (np.r_[c[:2], lowz], np.r_[c[0] + sx * 0.25 * size[0], c[1], c[2]]):
            for tilt in (0.0, 25.0, 45.0):
                tl = np.radians(tilt)
                a = G.unit(np.array([-sx * np.cos(tl), 0.0, -np.sin(tl)]))
                for cv in ([0, 1.0, 0], [0, -1.0, 0]):
                    cand.append((gp, a, np.asarray(cv)))
        for gp in (np.r_[c[:2], lowz], np.r_[c[:2], bot + 0.5 * hgt] if hgt > 0.07 else np.r_[c[:2], c[2]]):   # also from the robot side
            for tilt in (0.0, 30.0, 50.0):
                tl = np.radians(tilt)
                a = G.unit(np.array([-sx * 0.35 * np.cos(tl), np.cos(tl), -np.sin(tl)]))
                for cv in ([1.0, 0, 0], [-1.0, 0, 0]):
                    cv_ = np.asarray(cv) - a * float(np.asarray(cv) @ a)
                    cand.append((gp, a, G.unit(cv_)))
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
            Rg, tg = Rg_, tg_                              # the giver's hand frame at the moment of the hand-over
            nh = {"c": R.T @ (c - tt), "b": R.T @ (np.r_[c[:2], bot] - tt), "size": hd["size"]}
            for k in ("floor",):                          # points: giver hand frame -> world -> receiver hand frame
                if k in hd:
                    nh[k] = R.T @ (tg + Rg @ hd[k] - tt)
            if "u" in hd:
                nh["u"] = R.T @ (Rg @ hd["u"])
            if "top" in hd:
                nh["top"] = float((tg[2] + hd["top"]) - tt[2])
            if "fp" in hd:
                nh["fp"] = hd["fp"]
            if "pts_h" in hd:
                nh["pts_h"] = (R.T @ ((Rg @ hd["pts_h"].T).T + tg - tt).T).T
            self.hold[to] = nh
            self._forget_release(obj)
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

    # present (hold in the air at a table-frame point)
    def _present_plan(self, arm, xy_cm, height_cm, turn_deg=None):
        """hand pose that puts the HELD object's centre over xy (cm, table frame) with its bottom height_cm above the
        table, turning it only about the vertical (never tilting). If the exact point is out of reach, the nearest reachable
        point (toward this arm's base, then a little higher / lower) is used and reported."""
        from transforms3d.axangles import axangle2mat
        from transforms3d.quaternions import mat2quat
        est = self.held_est(arm)
        if est is None:
            return None, {"code": "NOT_HOLDING", "why": f"the {arm} arm holds nothing"}
        c, bot = est
        tcp, q0 = self.tcp(arm), self.A.hand_pose(arm)[3:]
        want = np.r_[np.asarray(xy_cm, float)[:2] / 100, self.tz + float(height_cm) / 100]
        base = np.asarray(G.ARM_BASE[arm], float)
        tow = G.unit(base - want[:2])
        pts = [(0.0, 0.0)] + [(d, dz) for d in (0.03, 0.06, 0.09, 0.12) for dz in (0.0, 0.03, -0.03)]
        y0 = float(turn_deg or 0.0)
        yaws = [y0 + e for e in (0, 10, -10, 20, -20, 35, -35)]
        for d, dz in pts:
            goal = np.r_[want[:2] + tow * d, want[2] + dz]
            for yaw in yaws:
                Rm = axangle2mat([0, 0, 1.0], np.radians(yaw))
                q = G.quat_mul(np.asarray(mat2quat(Rm)), q0)
                off = Rm @ (c - tcp)
                boff = Rm @ (np.r_[c[:2], bot] - tcp)
                dest = np.r_[goal[:2] - off[:2], goal[2] - boff[2]]
                if self.feasible(arm, [self.A.hand_from_tcp(dest, q)]):
                    return {"arm": arm, "dest": dest, "q": q, "yaw": yaw, "goal": goal, "shift_cm": round(d * 100), "dz_cm": round(dz * 100)}, None
        return None, {"code": "UNREACHABLE", "why": f"the {arm} arm cannot hold its object at {list(xy_cm)} cm, {height_cm} cm high, or "
                                                    "within 12 cm of it; choose a point nearer this arm's side"}

    def present(self, reqs):
        """reqs = [(arm, xy_cm, height_cm, turn_deg)], one or two arms; both arms move at the same time. The path rises
        first (never sweeps low over the table), turns in place, crosses at that height, then settles at the point."""
        plans, out = {}, {}
        for arm, xy, h, td in reqs:
            if arm not in ARMS or not self.holding.get(arm):
                out[arm] = {"code": "NOT_HOLDING", "why": f"the {arm} arm holds nothing (grasp first)"}
                continue
            if xy is None:
                est = self.held_est(arm)
                xy = list(np.asarray(est[0][:2]) * 100) if est is not None else None
            pl, err = self._present_plan(arm, xy, h, td)
            if pl is None:
                out[arm] = err
            else:
                plans[arm] = pl
        if not plans:
            return out
        for a in plans:
            self._forget_held_contents(a)
        seqs = {}
        for a, pl in plans.items():
            t = self.tcp(a)
            zt = max(float(t[2]), float(pl["dest"][2]))
            seqs[a] = [self.move_tcp(a, np.r_[t[:2], zt]), self.move_tcp(a, np.r_[t[:2], zt], pl["q"]),
                       self.move_tcp(a, np.r_[pl["dest"][:2], zt], pl["q"]), self.move_tcp(a, pl["dest"], pl["q"])]
        codes = self.run_seq(seqs)
        self._raw, self._drop = True, False
        try:
            seen = self.perceive([self.holding[a] for a in plans])
        finally:
            self._raw, self._drop = False, True
        for a, pl in plans.items():
            obj = self.holding[a]
            est = self.held_est(a)
            gap = float(self.A.opening(a))
            miss = float(np.linalg.norm(est[0][:2] - pl["goal"][:2])) if est is not None else 1.0
            code_ = codes[a] if codes[a] != "POSE_ERR" else ("OK" if miss <= 0.03 else "SHORT")     # a move that ended short is reported
            r = {"code": code_, "object_center_cm": self.cm(est[0]), "object_bottom_cm": round((est[1] - self.tz) * 100, 1),
                 "finger_opening": round(gap, 3)}
            if pl["shift_cm"] or pl["dz_cm"]:
                r["note"] = f"the exact point was out of reach: held {pl['shift_cm']} cm closer to the {a} arm" + (f", {pl['dz_cm']:+d} cm in height" if pl["dz_cm"] else "")
            if pl["yaw"]:
                r["turned_deg"] = round(float(pl["yaw"]))
            if gap < GAP_MIN:
                r["code"] = "SLIPPED"
                r["why"] = "the fingers closed fully: the object was dropped"
            o = seen.get(obj)
            if o is not None and len(o["pts"]) >= 30 and r["code"] == "OK":   # hold check from the cameras
                cam = np.asarray(G.body_center(o["pts"]), float)
                dxy = float(np.linalg.norm(cam[:2] - est[0][:2]))
                r["camera_offset_cm"] = round(dxy * 100, 1)
                hs = np.asarray(self.hold[a]["size"], float)
                if hs[2] >= 1.5 * max(hs[0], hs[1]) and float(o["size"][2]) < 0.6 * hs[2]:
                    r["warning"] = "the cameras see the object much shorter than when grasped: it may have tilted in the fingers"
                if dxy > 0.06 and o["lo"][2] < 2.0:
                    r["code"], r["why"] = "SLIPPED", "the cameras see the object on the table, not in the gripper"
            out[a] = r
        return out

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
        out = {}
        for a in arms:
            out[a] = {"code": codes.get(a, "OK"), "gripper_z_cm": round((self.tcp(a)[2] - self.tz) * 100, 1)}
            if a not in seq:
                out[a]["note"] = f"already at or above {float(h):.0f} cm; not moved"
        return out

    PRESS_DIRS = {"from_front": [0, 1.0, 0], "from_back": [0, -1.0, 0], "from_left": [1.0, 0, 0], "from_right": [-1.0, 0, 0]}

    def _panel(self, pts):
        """a thin upright plate (a rocker switch, a sideways button panel): its face normal n pointing toward the robot
        side, and the push point on its more protruding half (the raised end of a rocker), from the object's points."""
        pts = np.asarray(pts, float)
        if len(pts) < 20:
            return None
        u, m, L, W = self._mar(pts[:, :2])
        hz = float(np.percentile(pts[:, 2], 98) - np.percentile(pts[:, 2], 2))
        if not (L >= 1.6 * W and hz >= W and W <= 0.05):
            return None
        n = np.r_[m, 0.0]
        if n[1] > 0:                                     # face the robot (-y)
            n = -n
        s = pts @ n
        zmid = float(np.median(pts[:, 2]))
        halves = [pts[pts[:, 2] >= zmid], pts[pts[:, 2] < zmid]]
        prot = [float(np.percentile(h @ n, 90)) if len(h) >= 5 else -9.0 for h in halves]
        hsel = halves[int(np.argmax(prot))]
        z_lo, z_hi = float(np.percentile(hsel[:, 2], 10)), float(np.percentile(hsel[:, 2], 90))
        band = hsel[np.abs(hsel[:, 2] - (z_lo + z_hi) / 2) <= max(0.006, (z_hi - z_lo) / 3)]
        band = band if len(band) >= 5 else hsel
        p = np.median(band, 0)
        p = p + n * (float(np.percentile(band @ n, 90)) - float(p @ n))      # on the front surface of that half
        return {"n": n, "p": p, "raised": "upper" if int(np.argmax(prot)) == 0 else "lower", "prot_diff_cm": round(abs(prot[0] - prot[1]) * 100, 1)}

    def press(self, arm, obj, direction=None, confirm=False):
        led = self._ledger().get(obj)
        if not confirm and led is not None and led.get("state") == "pressed" and not led.get("warned"):
            led["warned"] = True
            return {arm: {"code": "CONFIRM_NEEDED", "why": f"{obj} was already pressed at call {led['call']} (reached it). A second press can undo a "
                          "switch or a toggle. Look at the images: if it changed, do not press again (go on or call done). If it clearly did "
                          "not change, repeat this press with confirm=true."}}
        r = self._press(arm, obj, direction)
        if (r.get(arm) or {}).get("code") == "OK":
            self._ledger()[obj] = {"state": "pressed", "what": "pressed once", "call": int(getattr(self, "k", -1))}
        return r

    def _press(self, arm, obj, direction=None):
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
        if direction and direction not in ("down", "tilted", "into_face", *self.PRESS_DIRS):
            return {arm: {"code": "BAD_ARG", "why": "direction must be down, into_face, from_front, from_left, from_right or from_back"}}
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
                a = np.asarray(self.PRESS_DIRS.get(dname, [0, 1.0, 0]), float)
                pn = self._panel(pts) if dname in ("into_face", *self.PRESS_DIRS) else None
                if pn is not None and (dname == "into_face" or float(-pn["n"] @ a) > 0.5):
                    a = -pn["n"]                            # push along the plate's own normal (it may be turned), at its raised half
                    p = pn["p"]
                    self.trace.append({"press_panel": {"normal_deg": round(float(np.degrees(np.arctan2(a[1], a[0]))), 1), "raised": pn["raised"],
                                                       "prot_diff_cm": pn["prot_diff_cm"], "p_cm": self.cm(p)}})
                    deep = 0.04
                else:
                    if dname == "into_face":
                        tried.append("into_face: no thin upright plate found"); continue
                    sproj = pts @ a
                    near = pts[sproj <= np.percentile(sproj, 1) + 0.004]     # the most protruding patch (a toggle's raised end), not the whole face
                    if len(near) < 5:
                        near = pts[sproj <= np.percentile(sproj, 4) + 0.008]
                    p = np.median(near, 0)
                    deep = 0.02
                q = G.hand_quat(a, [0, 0, 1.0])
                start, end = p - a * 0.07, p + a * deep
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
        return {arm: {"code": "OK" if reached else "NOT_REACHED", "direction": dname, "pressed_point_cm": self.cm(p),
                      "reach_past_surface_cm": round(depth * 100, 1),
                      "note": "whether the button / switch changed is not measured; judge it from the images (if it changed, do not press it again: a second press can undo it)"}}

    def push(self, arm, obj, direction, dist_cm):
        self._forget_release(obj)
        self._ledger().pop(obj, None)
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
        self._ledger().pop(obj, None)
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
        if str(motion) in ("pull", "push_in", "slide_out", "slide_in"):   # drawer, straight slide
            return self._articulate_slide(arm, obj, o, d, motion, degrees)
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
        out = {"code": "EXECUTED" if done >= 50 else "SHORT", "swing_deg": done, "handle_found": d["found"],
               "note": "door angle is not measured; judge it from the images"}
        if arm != good:
            out["hint"] = f"the door swings toward the {good} side; the {good} arm can follow it further"
        return {arm: out}

    def _articulate_slide(self, arm, obj, o, d, motion, dist):
        """the handles are the points sticking out in front of the front face
        grouped by height; the lowest group is the drawer that is pulled (unless only one). Grasp it from the front (pitched
        down), slide straight toward the robot (pull) or away (push_in) in 2 cm steps, release and back off; report the
        opened drawer's inside (xy) so an object can be put into it."""
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
            groups = [g for g in groups if len(g) >= 8] or groups
            g0 = groups[0]
            sel = pro[(pro[:, 2] >= g0[0] - 1e-6) & (pro[:, 2] <= g0[-1] + 1e-6)]
            hx, hy, hz = float(np.median(sel[:, 0])), float(np.median(sel[:, 1])), float(np.median(sel[:, 2]))
            self.trace.append({"drawer_handles": len(groups), "chosen_cm": self.cm([hx, hy, hz])})
        dist = float(dist) if 3.0 <= float(dist) <= 30.0 else 15.0
        if arm not in ARMS:
            arm = "left" if hx < 0 else "right"
        other = "left" if arm == "right" else "right"
        if self.holding.get(other) is None:
            self.call("home", {"arm": other})
        pull = str(motion) in ("pull", "slide_out")
        v = np.array([0.0, -1.0, 0.0]) if pull else np.array([0.0, 1.0, 0.0])
        best = None
        for pitch in (60.0, 45.0, 75.0, 30.0):
            b = np.radians(pitch)
            a = np.array([0.0, np.cos(b), -np.sin(b)])
            for dz in (0.0, -0.015, 0.015):
                p = np.array([hx, hy - 0.005, hz + dz])
                for c in ([1.0, 0, 0], [-1.0, 0, 0]):
                    q = G.hand_quat(a, c)
                    pre, hg = self.A.hand_from_tcp(p - a * 0.06, q), self.A.hand_from_tcp(p, q)
                    if self.feasible(arm, [pre]) and self.feasible(arm, [hg]):
                        best = (p, q, pre, hg)
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            return {arm: {"code": "UNREACHABLE", "why": f"cannot reach the drawer handle of {obj} with the {arm} arm; try the other arm"}}
        p, q, pre, hg = best
        self.run_seq({arm: [{"type": "grip", "val": 1.0}, {"type": "move", "pose": pre.tolist()}, {"type": "move", "pose": hg.tolist()},
                            {"type": "grip", "val": 0.0}]})
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
        out = {"code": "EXECUTED" if moved >= min(10.0, dist) else "SHORT", "slid_cm": moved, "handle_cm": self.cm([hx, hy, hz]),
               "note": "the drawer position is from the handle and the slide; check the images"}
        if pull and moved > 0:
            inside = np.r_[hx, face - moved / 100 / 2 + 0.01]          # between the old front face and the pulled-out front
            out["drawer_inside_xy_cm"] = r1(inside * 100)
            out["hint"] = (f"to put something into this drawer: place(arm, object, ref=\"table\", xy={r1(inside * 100)}) with the arm that "
                           "holds it (it is lowered into the drawer and released)")
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
        out = {"code": "EXECUTED" if done >= 50 else "SHORT", "swing_deg": done, "handle_found": True,
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
            return self.grasp(arm, a["object"], a.get("part"), confirm=bool(a.get("confirm", False)))
        if name in ("place", "carry"):
            td = a.get("turn_deg")
            return self.place(arm, a["object"], a.get("ref", "table"), a.get("relation", "on"), a.get("xy"),
                              release=name == "place", height_cm=float(a.get("height_cm", 10)), align=bool(a.get("align", False)),
                              turn_deg=None if td in (None, "", 0) else float(td), held_part=a.get("held_part"), face=a.get("face"), face_now=a.get("face_now"),
                              low_end=a.get("low_end"))
        if name == "present":
            if arm not in ARMS:
                raise ValueError("arm must be 'left' or 'right' (use bimanual with two present calls for both arms)")
            td = a.get("turn_deg")
            return self.present([(arm, a.get("xy"), float(a.get("height_cm", 15)), None if td in (None, "", 0) else float(td))])
        if name == "pour":
            return self.pour(arm, a.get("direction", "left"), float(a.get("degrees", 100)), bool(a.get("shake", True)), bool(a.get("return_upright", False)))
        if name == "handover":
            to = a.get("to_arm") or a.get("arm")
            if to not in ARMS:
                raise ValueError("to_arm must be 'left' or 'right'")
            return self.handover(a["object"], to, a.get("mode", "table"))
        if name == "lift":
            if arm not in (*ARMS, "both"):
                raise ValueError("arm must be left, right or both")
            return self.lift(arm, a.get("height_cm", 15))
        if name in ("move", "lift", "pour", "home") and arm in ARMS:
            self._forget_held_contents(arm)
        if name == "move":
            t = self.tcp(arm)
            c = self.run_seq({arm: [self.move_tcp(arm, t + np.array([a.get("dx", 0), a.get("dy", 0), a.get("dz", 0)], float) / 100)]})
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
            return self.press(arm, a["object"], a.get("direction"), confirm=bool(a.get("confirm", False)))
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
        if names == {"present"}:                     # both arms hold their objects in the air at the same time
            reqs = []
            for _, x in cs:
                td = x.get("turn_deg")
                reqs.append((x.get("arm"), x.get("xy"), float(x.get("height_cm", 15)), None if td in (None, "", 0) else float(td)))
            return self.present(reqs)
        if names == {"grasp"} and len({x[1].get("object") for x in cs}) == 2:     # two objects -> two single grasps (each with retries)
            out = {}
            for n, x in cs:
                if self.holding.get(x["arm"]):
                    raise ValueError(f"{x['arm']} already holds {self.holding[x['arm']]}")
            for n, x in cs:
                out.update(self.grasp(x["arm"], x["object"], x.get("part"), confirm=bool(x.get("confirm", False))))
            return out
        if names == {"grasp"}:
            for _, x in cs:
                self._forget_release(x.get("object"))
                self._ledger().pop(x.get("object"), None)
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
            if ok is False and p in ("on", "in", "at", "at_xy", "upright", "left_of", "right_of", "front_of", "behind", "near") and args \
                    and isinstance(args[0], str):
                self._refute(args[0], f"{p} measured false")             # contrary evidence ends the ledger entry
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
            held_by = [x for x, o in self.holding.items() if o == args[0] and x in self.hold and self.A.opening(x) >= GAP_MIN]   # a gripper that closed fully dropped it
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
    a = ap.parse_args()
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
        H.k = k
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


if __name__ == "__main__":
    main()
