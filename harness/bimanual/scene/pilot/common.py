"""Shared pieces of the oracle-expert data pilot: prompt constants (the exact deployment text), the predicate
evaluator on simulator-truth snapshots, and formatting helpers.

Units everywhere: centimetres; x to the right, y away from the robot (world frame), z = height above the table top.
"""
from __future__ import annotations

import json
import re

# ============================================================================ tool documentation (deployment text)
TOOL_DOC = """Tools (call exactly one per turn as <tool_call>{"name": ..., "arguments": {...}}</tool_call>):
- locate(object, part=null) - find an object: centre [x,y,z], size [dx,dy,dz], top_z, and the parts it can be grasped by.
- grasp(arm, object, part=null) - pick the grasp pose itself, approach, close the gripper; reports whether the object is held.
- place(arm, object, target, relation="on") - put the held object on/in/at the target, using the target's functional point,
  and open the gripper. target is an object name, "table" (then give xy=[x,y]) or "air" (then give xyz=[x,y,z]);
  release=false keeps holding it (e.g. to bring it to a hand-over point).
- move(arm, dx, dy, dz) - move the fingertips by a world-frame offset in cm (x right, y away from the robot, z up).
- rotate(arm, axis, deg) - rotate the gripper in place about the world axis "x" / "y" / "z".
- gripper(arm, action) - action "open" or "close".
- home(arm) - move the arm back to its start pose, out of the way.
Every tool answers with JSON {"ok": bool, "note": "...", "holding": {"left": ..., "right": ...}, "tcp": {"left": [x,y,z], "right": [x,y,z]}}
(locate adds "object": {...}). Units: cm; x to the right, y away from the robot, z = height above the table top."""

RELATION_DOC = """Completion criteria (object-state relations; units cm):
  holding(arm, obj)      both fingers of the arm grip obj
  lifted(obj, Hcm)       obj's bottom is at least H cm above where it started
  on(a, b)               a rests on top of b (b may be "table")
  at(a, x, y)            a's centre is within 3 cm of (x, y)
  gripper_open(arm)      the arm's gripper is open
  released(obj)          no finger touches obj
  above(a, b)            a is higher than b and over it
Prefix "not " to negate, e.g. not holding(left, cup)."""

PLANNER_SYSTEM = """You are the PLANNER ("brain") of a two-arm table-top robot (left arm and right arm, each with a parallel gripper; the robot stands at the front edge of the table facing +y).
You see the scene from several cameras and receive the overall instruction. Write an ordered list of SUBGOALS. An executor carries out one subgoal at a time with these tools:
""" + TOOL_DOC + """

A checker decides whether a subgoal is finished by testing its completion criteria on the camera images.
""" + RELATION_DOC + """

Answer with ONE JSON object only:
{"analysis": "<short reasoning about the scene and the order of steps>", "subgoals": [{"goal": "<instruction for the executor>", "arm": "left|right|both", "success": ["<relation>", ...]}, ...]}"""

EXECUTOR_SYSTEM = """You are the robot arm controller (executor) of a two-arm robot (left arm, right arm). You carry out ONE subgoal at a time given by the planner. Look at the images, the arm state and the tool feedback, think briefly inside <think>...</think>, then call exactly one tool as <tool_call>{"name": "...", "arguments": {...}}</tool_call>. When the subgoal is achieved, write <think>...</think> followed by "Subgoal complete." on its own line, with no tool call. If it cannot be achieved, write "Subgoal failed." the same way.

""" + TOOL_DOC

CHECKER_SYSTEM = """You are the CHECKER of a two-arm table-top robot. You get camera images taken BEFORE and AFTER the executor worked on a subgoal, the subgoal and its completion criteria. For each criterion decide from the images whether it holds now.
""" + RELATION_DOC + """

Answer with ONE JSON object only:
{"criteria": [{"criterion": "<relation>", "status": "satisfied|unsatisfied", "evidence": "<one sentence>"}], "subgoal": "satisfied|unsatisfied"}"""

CAMERA_NOTE = {
    "agent": "agent_camera (overview from above the robot; grey grid on the table every 10 cm, cyan L / orange R = fingertips)",
    "top": "top_camera (straight-down map view, +x right, +y up)",
    "left": "left_camera (wrist camera of the left arm)",
    "right": "right_camera (wrist camera of the right arm)",
}


def fmt_call(name, args) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "</tool_call>"


def arm_state_text(st: dict) -> str:
    lines = []
    for arm in ("left", "right"):
        a = st["arms"][arm]
        g = "open" if a["gripper_cmd"] > 0.8 else ("closed" if a["gripper_cmd"] < 0.2 else "half-closed")
        h = st["held"].get(arm) or "nothing"
        lines.append(f"{arm} arm: fingertips at {a['tcp']} cm, gripper {g}, holding {h}")
    return "\n".join(lines)


def executor_first_text(instruction, k, n, sub, objects, st) -> str:
    return (f"Overall task: {instruction}\nCurrent subgoal ({k}/{n}): {sub['goal']} (arm: {sub['arm']})\n"
            f"Success criteria: {'; '.join(sub['success'])}\nObjects in the scene: {', '.join(objects)}\n"
            + arm_state_text(st))


def planner_user_text(instruction, objects, st) -> str:
    return f"Task: {instruction}\nObjects in the scene: {', '.join(objects)}\n" + arm_state_text(st)


# ============================================================================ predicates on truth snapshots
_RX = re.compile(r"^\s*(not\s+)?([a-zA-Z_]+)\s*\((.*)\)\s*$")
ARMS = ("left", "right")
RELATIONS = {"holding", "lifted", "on", "at", "gripper_open", "gripper_closed", "released", "above", "on_table"}


def parse(expr):
    m = _RX.match(expr or "")
    if not m:
        raise ValueError(f"cannot parse '{expr}'")
    args = []
    for a in [x.strip().strip("'\"") for x in m.group(3).split(",") if x.strip()]:
        a2 = a[:-2].strip() if a.endswith("cm") else a
        try:
            args.append(float(a2))
        except ValueError:
            args.append(a)
    return bool(m.group(1)), m.group(2).lower(), args


def validate(expr, objects):
    neg, name, a = parse(expr)
    if name not in RELATIONS:
        raise ValueError(f"unknown relation {name}")
    objs = set(objects) | {"table"}
    need = {"holding": [(0, "arm"), (1, "obj")], "lifted": [(0, "obj")], "on": [(0, "obj"), (1, "obj")],
            "at": [(0, "obj")], "gripper_open": [(0, "arm")], "gripper_closed": [(0, "arm")],
            "released": [(0, "obj")], "above": [(0, "obj"), (1, "obj")], "on_table": [(0, "obj")]}[name]
    for i, kind in need:
        if i >= len(a):
            raise ValueError(f"{expr}: missing argument")
        if kind == "arm" and a[i] not in ARMS:
            raise ValueError(f"{expr}: argument {i+1} must be left/right")
        if kind == "obj" and a[i] not in objs:
            raise ValueError(f"{expr}: unknown object {a[i]}")
    if name == "at" and (len(a) < 3 or not all(isinstance(x, float) for x in a[1:3])):
        raise ValueError(f"{expr}: at(obj, x, y) needs numbers")


def _o(S, n):
    return S["objects"][n]


def evaluate(expr, S, S0):
    """-> (value, evidence sentence) on a truth snapshot S (S0 = episode start)."""
    neg, name, a = parse(expr)
    v, ev = _eval(name, a, S, S0)
    if neg:
        v = not v
    return bool(v), ev


def _inside(c, o, margin):
    lo, hi = o["lo"], o["hi"]
    return lo[0] - margin <= c[0] <= hi[0] + margin and lo[1] - margin <= c[1] <= hi[1] + margin


def _eval(name, a, S, S0):
    if name == "holding":
        arm, obj = a[0], a[1]
        h = S["held"].get(arm)
        if h == obj:
            return True, f"both fingers of the {arm} gripper are closed on the {obj}"
        touching = [o for ar, o in S["contacts"] if ar == arm]
        if obj in touching:
            return False, f"the {arm} gripper touches the {obj} with only one finger, it is not held"
        return False, f"the {arm} gripper holds {h or 'nothing'}, not the {obj}"
    if name == "lifted":
        obj = a[0]
        h = float(a[1]) if len(a) > 1 else 5.0
        dz = _o(S, obj)["lo"][2] - _o(S0, obj)["lo"][2]
        return dz >= h, f"the {obj}'s bottom is {dz:.0f} cm above where it started (needs {h:.0f} cm)"
    if name == "on_table" or (name == "on" and len(a) > 1 and a[1] == "table"):
        o = _o(S, a[0])
        ok = -2.0 <= o["lo"][2] <= 2.0
        return ok, (f"the {a[0]} rests on the table" if ok else f"the {a[0]}'s bottom is {o['lo'][2]:.0f} cm above the table")
    if name in ("on", "above"):
        oa, ob = _o(S, a[0]), _o(S, a[1])
        c = oa["center"]
        dz = oa["lo"][2] - ob["hi"][2]
        inside = _inside(c, ob, 1.0 if name == "on" else 2.0)
        dxy = ((c[0] - ob["center"][0]) ** 2 + (c[1] - ob["center"][1]) ** 2) ** 0.5
        if name == "on":
            ok = inside and -3.0 <= dz <= 3.0
            if ok:
                return True, f"the {a[0]} sits on the {a[1]} ({dxy:.0f} cm off its centre)"
            if not inside:
                return False, f"the {a[0]} is {dxy:.0f} cm away from the {a[1]}'s centre, not over it"
            return False, f"the {a[0]} is over the {a[1]} but {dz:.0f} cm above its top"
        ok = inside and dz > 0
        return ok, (f"the {a[0]} is {dz:.0f} cm above the {a[1]}" if ok else f"the {a[0]} is not above the {a[1]}")
    if name == "at":
        o = _o(S, a[0])
        d = ((o["center"][0] - a[1]) ** 2 + (o["center"][1] - a[2]) ** 2) ** 0.5
        tol = float(a[3]) if len(a) > 3 else 3.0
        return d <= tol, f"the {a[0]}'s centre is {d:.0f} cm from ({a[1]:.0f}, {a[2]:.0f})"
    if name in ("gripper_open", "gripper_closed"):
        g = S["arms"][a[0]]["gripper_cmd"]
        ok = g > 0.8 if name == "gripper_open" else g < 0.2
        return ok, f"the {a[0]} gripper is {'open' if g > 0.8 else 'closed' if g < 0.2 else 'half-closed'}"
    if name == "released":
        t = [o for _, o in S["contacts"]]
        return a[0] not in t, (f"no finger touches the {a[0]}" if a[0] not in t else f"a finger still touches the {a[0]}")
    raise ValueError(name)
