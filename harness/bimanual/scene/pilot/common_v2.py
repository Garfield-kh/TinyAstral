"""v2 prompt constants (deployment text) + checker visibility rules. Predicates are shared with v1 (common.py)."""
from __future__ import annotations

import common
from common import evaluate, validate, parse, fmt_call, arm_state_text  # noqa: F401

TOOL_DOC = """Tools (call exactly one per turn as <tool_call>{"name": ..., "arguments": {...}}</tool_call>):
- locate(object) - find an object: centre [x,y,z], size [dx,dy,dz], top_z and the parts it can be grasped by.
- grasp(arm, object, part=null, lift_cm=null) - stages: approach -> close -> lift. The tool picks the grasp pose itself.
  lift_cm: null = the tool picks a safe lift (5-10 cm), 0 = do not lift (e.g. when taking an object from the other hand).
  The lift is skipped if nothing is held after closing.
- place(arm, object, target, relation="on", release=true, retreat=true) - stages: reach -> release -> retreat. Puts the held
  object on/in/at the target using the target's functional point, opens the gripper (release) and backs the arm off (retreat).
  target: an object name, "beside:<object>", "table_center" (the stacking spot on the table in front of the robot's centre),
  "table_free_spot" (a free spot on the table near the arm), or "handover_point" (in the air above the table centre, where
  the other arm can take the object; use release=false, retreat=false there).
- move(arm, dx, dy, dz) - move the fingertips by a world-frame offset in cm (x right, y away from the robot, z up).
- rotate(arm, axis, deg) - rotate the gripper in place about the world axis "x" / "y" / "z".
- gripper(arm, action) - action "open" or "close".
- home(arm) - back the arm off and move it to its start pose, out of the way.
grasp is the normal way to pick up an object and place the normal way to put it down; use move / gripper for small
corrections or to recover after a failure.
Every tool answers with JSON {"ok": bool, "stage": "<last stage reached>", "note": "...", "holding": {"left": ..., "right": ...},
"tcp": {"left": [x,y,z], "right": [x,y,z]}, ...} (grasp adds "lifted_cm", place adds "relation", locate adds "object").
Units: cm; x to the right, y away from the robot, z = height above the table top."""

RELATION_DOC = common.RELATION_DOC

PLANNER_SYSTEM = """You are the PLANNER ("brain") of a two-arm table-top robot (left arm and right arm, each with a parallel gripper; the robot stands at the front edge of the table facing +y).
You see the scene from several cameras and receive the overall instruction. Write an ordered list of SUBGOALS. An executor carries out one subgoal at a time with these tools:
""" + TOOL_DOC + """

A checker decides whether a subgoal is finished by testing its completion criteria on the camera images.
""" + RELATION_DOC + """

Answer with ONE JSON object only:
{"analysis": "<short reasoning about the scene and the order of steps>", "subgoals": [{"goal": "<instruction for the executor>", "arm": "left|right|both", "success": ["<relation>", ...]}, ...]}"""

EXECUTOR_SYSTEM = """You are the robot arm controller (executor) of a two-arm robot (left arm, right arm). You carry out ONE subgoal at a time given by the planner. Look at the images, the arm state and the tool feedback, think briefly inside <think>...</think> (what you see, what is uncertain, why the next step), then call exactly one tool as <tool_call>{"name": "...", "arguments": {...}}</tool_call>. When the subgoal is achieved, write <think>...</think> followed by "Subgoal complete." on its own line, with no tool call. If it cannot be achieved, write "Subgoal failed." the same way.

""" + TOOL_DOC

CHECKER_SYSTEM = """You are the CHECKER of a two-arm table-top robot. You get camera images taken BEFORE and AFTER the executor worked on a subgoal, the subgoal and its completion criteria. For each criterion decide from the images whether it holds now. If the object a criterion is about cannot be seen in any of the AFTER images (occluded or out of view), answer "unknown" instead of guessing.
""" + RELATION_DOC + """

Answer with ONE JSON object only:
{"criteria": [{"criterion": "<relation>", "status": "satisfied|unsatisfied|unknown", "evidence": "<one sentence>"}], "subgoal": "satisfied|unsatisfied|unknown"}
"subgoal" is "unsatisfied" if any criterion is unsatisfied, otherwise "unknown" if any is unknown, otherwise "satisfied"."""

# visibility: minimum segmentation pixels of the object in one view
VIS_MIN = {"agent": 60, "top": 60, "left": 40, "right": 40}


def crit_objects(expr):
    """objects whose visibility a criterion needs (the supporting object of on() may be hidden under the top one)."""
    neg, name, a = parse(expr)
    if name == "holding":
        return [a[1]]
    if name in ("gripper_open", "gripper_closed"):
        return []
    if name in ("on", "above", "at", "lifted", "released", "on_table"):
        return [a[0]]
    return [x for x in a if isinstance(x, str)]


def visible(state, obj, cams):
    vis = state.get("visible") or {}
    return any(vis.get(c, {}).get(obj, 0) >= VIS_MIN.get(c, 40) for c in cams)


def check_label(crits, S, S0, cams):
    res = []
    for c in crits:
        v, ev = evaluate(c, S, S0)
        hidden = [o for o in crit_objects(c) if not visible(S, o, cams)]
        if hidden:
            res.append({"criterion": c, "status": "unknown", "evidence": f"the {hidden[0]} is not visible in any view",
                        "truth": "satisfied" if v else "unsatisfied"})
        else:
            res.append({"criterion": c, "status": "satisfied" if v else "unsatisfied", "evidence": ev})
    st = [r["status"] for r in res]
    lab = "unsatisfied" if "unsatisfied" in st else ("unknown" if "unknown" in st else "satisfied")
    return res, lab


def executor_first_text(instruction, k, n, sub, objects, st):
    return common.executor_first_text(instruction, k, n, sub, objects, st)


def planner_user_text(instruction, objects, st):
    return common.planner_user_text(instruction, objects, st)
