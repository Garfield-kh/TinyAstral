#!/usr/bin/env python
"""UniBench ceiling bake-off with the generalised v3 harness (S6 harness: next-subgoal line + arm hand-off rule,
request_check + live checker, harness memory), oracle condition, Flash as every role.

Generalisations (task-agnostic, all patched here; v3 files untouched):
  * relation task_success() for steps no object relation describes (planner/checker docs); on() also covers "inside";
  * tools: move(arm="both") moves both arms together; press(arm, object, part, depth_cm) for buttons/bells/staplers;
    place targets "left_of:<obj>" / "right_of:<obj>".
Conditions: Z = zero-shot (no demo, only the general rules); X = one demo episode of a DIFFERENT task of the same kind
(from the oracle_v3_50 pilot, see XSRC).
Seeds: UniBench / RoboDawn valid seeds (episode_index 0-2 -> seeds as in ub4/fw4: seed_base_index=0).
Output: results/unibench/<lane>/<task>/shard_<k>/{results.json, episode_XXX/...} (ub4_summary format)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g50_build  # noqa: E402,F401  (relation patches)
import v3_closed_loop as V  # noqa: E402
import closed_loop as CL  # noqa: E402
import common_v2 as C  # noqa: E402
import v3_build as VB  # noqa: E402
import record_v2 as RV2  # noqa: E402
from record import ArmTag  # noqa: E402

import paths as _paths  # noqa: E402
P = _paths.OUT
SEEDS = _paths.SEEDCACHE
PILOT = _paths.OUT / "pilot"
XSRC = {"click_bell": "click_alarmclock", "press_stapler": "click_alarmclock", "open_microwave": "open_laptop",
        "open_laptop": "open_microwave", "grab_roller": "lift_pot", "lift_pot": "grab_roller",
        "place_a2b_left": "move_stapler_pad", "place_empty_cup": "place_container_plate",
        "place_object_basket": "place_bread_basket", "place_phone_stand": "place_object_stand",
        "beat_block_hammer": "stamp_seal", "stack_blocks_two": "stack_bowls_two", "stack_blocks_three": "stack_bowls_three",
        "stack_bowls_two": "stack_blocks_two", "handover_block": "pick_dual_bottles"}

# ------------------------------------------------------------------ prompt patches
REPL = [
    ('a rests on top of b (b may be "table")', 'a rests on top of b or inside it (b may be "table")'),
    ("  above(a, b)            a is higher than b and over it",
     "  above(a, b)            a is higher than b and over it\n"
     "  task_success()         the overall task instruction is fulfilled (for steps no other relation describes, e.g.\n"
     "                         pressing, opening, turning, shaking)"),
    ("- move(arm, dx, dy, dz) - move the fingertips by a world-frame offset in cm (x right, y away from the robot, z up).",
     "- move(arm, dx, dy, dz) - move the fingertips by a world-frame offset in cm (x right, y away from the robot, z up).\n"
     "  arm=\"both\" moves both arms together by the same offset (e.g. to lift an object held with two hands).\n"
     "- press(arm, object, part=null, depth_cm=2) - close the gripper, approach the object (or the named part) along the\n"
     "  tool's approach direction until a finger touches it, push depth_cm further, then return to the approach pose\n"
     "  (buttons, bells, staplers, switches)."),
    ('"beside:<object>",', '"beside:<object>", "left_of:<object>" / "right_of:<object>" (on the table about 12 cm to that\n'
                           '  side of the object),'),
]


def patch_prompts():
    n = {}
    for mod, attr in ((C, "PLANNER_SYSTEM"), (VB, "EXECUTOR_SYSTEM_V3"), (V, "CHECKER_SYS")):
        s = getattr(mod, attr)
        for a, b in REPL:
            n[(attr, a[:20])] = s.count(a)
            s = s.replace(a, b)
        setattr(mod, attr, s)
    return n


# ------------------------------------------------------------------ tools
REPL_V2 = [
    ("- grasp(arm, object, part=null, lift_cm=null) - stages: approach -> close -> lift. The tool picks the grasp pose itself.",
     "- grasp(arm, object, part=null, lift_cm=null) - stages: approach -> close -> lift. The tool picks the grasp pose itself.\n"
     "  part may be \"left end\" / \"right end\" for long objects; arm=\"both\" grasps a long object with both hands (left\n"
     "  hand at its left end, right hand at its right end, no lift). Articulated objects (doors, lids, handles) are not lifted."),
]
UL = ("upper half", "lower half", "upper", "lower", "top", "bottom")


class G50Tools(V.Tools3):
    V2 = False
    def t_move(self, arm, dx=0, dy=0, dz=0):
        if arm != "both":
            return super().t_move(arm, dx, dy, dz)
        d = [float(dx or 0), float(dy or 0), float(dz or 0)]
        if max(abs(x) for x in d) > 40:
            raise ValueError("each offset must be within +-40 cm")
        self.T.plan_success = True
        a1 = self.T.move_by_displacement(ArmTag("left"), x=d[0] / 100, y=d[1] / 100, z=d[2] / 100)
        a2 = self.T.move_by_displacement(ArmTag("right"), x=d[0] / 100, y=d[1] / 100, z=d[2] / 100)
        self.T.move(a1, a2)
        ok = bool(self.T.plan_success)
        self.T.plan_success = True
        S = self._state()
        h = [f"{a}: {S['held'][a]}" for a in ("left", "right") if self.held(S, a)]
        return RV2.fb(S, ok, "done", (f"moved both fingertips by ({d[0]:+.0f}, {d[1]:+.0f}, {d[2]:+.0f}) cm" if ok
                                      else "motion planner could not move both arms by that offset")
                      + (f"; holding {', '.join(h)}" if h else "")), S

    def _side_cps(self, actor, arm, part):
        try:
            pts = {i: np.asarray(p)[:3] for i, p in actor.iter_contact_points("list")}
        except Exception:  # noqa: BLE001
            return None
        if not pts:
            return None
        c = np.asarray(actor.get_pose().p)
        if part in ("left", "left end", "left side", "left half"):
            return [i for i in pts if pts[i][0] < c[0]] or None
        if part in ("right", "right end", "right side", "right half"):
            return [i for i in pts if pts[i][0] >= c[0]] or None
        other = "left" if arm == "right" else "right"
        S0 = self.ep.states[-1]
        name = next((n for n, v in self.sim.names.items() if v is actor), None)
        if name and self.held(S0, other) == name:
            t = S0["arms"][other]["tcp"]
            ot = np.array([t[0] / 100, t[1] / 100])
            return [i for i in pts if np.linalg.norm(pts[i][:2] - ot) > 0.08] or None
        return None

    def t_grasp(self, arm, object, part=None, lift_cm=None):
        if not self.V2:
            return super().t_grasp(arm, object, part, lift_cm)
        if arm == "both":
            fl, _ = self.t_grasp("left", object, "left end", 0)
            fr, S = self.t_grasp("right", object, "right end", 0)
            ok = bool(fl.get("ok") and fr.get("ok"))
            return RV2.fb(S, ok, "closed", f"left: {fl.get('note')}; right: {fr.get('note')}"), S
        actor = self._obj(object)
        from envs.utils.actor_utils import ArticulationActor
        if isinstance(actor, ArticulationActor) and lift_cm is None:
            lift_cm = 0
        cps = self._side_cps(actor, arm, part)
        if cps is not None:
            orig = self.T.grasp_actor

            def g(*a, **kw):
                if kw.get("contact_point_id") is None:
                    kw["contact_point_id"] = cps
                return orig(*a, **kw)
            self.T.grasp_actor = g
        try:
            return super().t_grasp(arm, object, part if part in UL else None, lift_cm)
        finally:
            if "grasp_actor" in vars(self.T):
                del self.T.grasp_actor

    def _cp(self, actor, part):
        if part in ("upper half", "lower half", "upper", "lower", "top", "bottom"):
            c = np.asarray(actor.get_pose().p)
            pts = {i: np.asarray(p)[:3] for i, p in actor.iter_contact_points("list")}
            up = part in ("upper half", "upper", "top")
            return [i for i in pts if (pts[i][2] >= c[2]) == up] or None
        return None

    def t_press(self, arm, object, part=None, depth_cm=2):
        tag, actor = self._arm(arm), self._obj(object)
        depth = min(max(float(depth_cm if depth_cm is not None else 2), 0.0), 6.0)
        self._run(self.T.close_gripper(tag))
        act = self.T.grasp_actor(actor, tag, pre_grasp_dis=0.1, grasp_dis=0.1, contact_point_id=self._cp(actor, part),
                                 gripper_pos=0.0)
        if act is None or act[0] is None or not act[1]:
            self.T.plan_success = True
            S = self._state()
            return RV2.fb(S, False, "approach", f"no reachable approach pose for {object} with the {arm} arm"), S
        moves = [a for a in act[1] if a.action == "move"]
        if not self._run((act[0], moves)):
            S = self._state()
            return RV2.fb(S, False, "approach", f"could not reach the approach pose of {object}"), S
        import transforms3d as t3d
        pose0 = np.asarray(self.T.get_arm_pose(tag), float)
        fwd = t3d.quaternions.quat2mat(pose0[3:])[:, 0]
        touched, travelled = False, 0.0
        for _ in range(8):                       # 2 cm steps along the approach direction until a finger touches
            st = 0.02
            if not self._run(self.T.move_by_displacement(tag, x=fwd[0] * st, y=fwd[1] * st, z=fwd[2] * st)):
                break
            travelled += st
            if any(o == object for a, o, *_ in self.sim.contacts() if a == arm):
                touched = True
                break
        if touched and depth > 0:
            self._run(self.T.move_by_displacement(tag, x=fwd[0] * depth / 100, y=fwd[1] * depth / 100, z=fwd[2] * depth / 100))
        S = self._state()                        # success is checked here (at the bottom of the press)
        self._run(self.T.move_to_pose(tag, pose0))
        S = self._state()
        note = (f"pressed {object}: a finger touched it after {travelled * 100:.0f} cm, pushed {depth:.0f} cm further and came back"
                if touched else f"moved {travelled * 100:.0f} cm toward {object} without touching it and came back")
        return RV2.fb(S, touched, "returned", note), S

    def t_place(self, arm, object, target, relation="on", release=True, retreat=True, **kw):
        tgt = str(target)
        if not (tgt.startswith("left_of:") or tgt.startswith("right_of:")):
            fb, S = super().t_place(arm, object, target, relation, release, retreat, **kw)
            if self.V2 and not fb.get("ok") and fb.get("stage") == "reach" and self.held(S, arm) == object:
                orig = self.T.place_actor

                def pa(*a, **k):
                    k["constrain"] = "align"
                    return orig(*a, **k)
                self.T.place_actor = pa
                try:
                    fb2, S2 = super().t_place(arm, object, target, relation, release, retreat, **kw)
                finally:
                    del self.T.place_actor
                fb2["note"] = "first attempt failed (" + str(fb.get("note")) + "); retried with the object aligned to the target: " + str(fb2.get("note"))
                return fb2, S2
            return fb, S
        side = -1 if tgt.startswith("left_of:") else 1
        b = self._obj(tgt.split(":", 1)[1])
        p = np.asarray(b.get_pose().p)
        orig = self.sim.names
        # reuse the beside: branch with the side forced by temporarily pretending the arm side
        tag, actor = self._arm(arm), self._obj(object)
        S0 = self.ep.states[-1]
        if self.held(S0, arm) != object:
            return RV2.fb(S0, False, "reach", f"the {arm} arm is not holding {object}"), S0
        bias = float(getattr(self.T, "table_z_bias", 0.0))
        tp = [p[0] + side * 0.12, p[1], 0.74 + bias + 0.01, 0, 1, 0, 0]
        act = self.T.place_actor(actor, tag, target_pose=tp, functional_point_id=0, pre_dis=0.05, dis=0.0,
                                 is_open=bool(release), pre_dis_axis="fp")
        if act is None or act[0] is None or not act[1]:
            self.T.plan_success = True
            S = self._state()
            return RV2.fb(S, False, "reach", f"no reachable place pose for {object} at {tgt}"), S
        acts = list(act[1])
        n_move = sum(1 for a in acts if a.action == "move")
        if not self._run((act[0], acts[:n_move])):
            S = self._state()
            return RV2.fb(S, False, "reach", f"could not reach the place pose at {tgt}"), S
        if not release:
            S = self._state()
            return RV2.fb(S, self.held(S, arm) == object, "reached", f"{object} is at {tgt} and still held"), S
        self._run((act[0], acts[n_move:]))
        S = self._state()
        rv = C.evaluate(f"on({object}, table)", S, self.ep.states[0])[0]
        stage = "released"
        if retreat is not False:
            self._run(self.T.move_by_displacement(tag, z=0.07))
            S = self._state()
            stage = "retreated"
        del orig
        return RV2.fb(S, bool(rv), stage, f"put {object} on the table {tgt.replace(':', ' the ').replace('_', ' ')}, released it"
                      if rv else f"released {object} but it is not resting on the table",
                      {"relation": f"on({object}, table) = {str(bool(rv)).lower()}"}), S


_at = VB.Memory.after_tool


def _after_tool(self, c, S):
    if isinstance(c.get("args"), dict) and c["args"].get("arm") == "both":
        for a in ("left", "right"):
            _at(self, dict(c, args=dict(c["args"], arm=a)), S)
        return None
    return _at(self, c, S)


VB.Memory.after_tool = _after_tool
_checker = V.checker


def _checker2(ep, sub, frames_idx, tools):
    if sub.get("arm_main") not in ("left", "right") and sub.get("arm") not in ("left", "right"):
        sub = dict(sub, arm_main="right")
    return _checker(ep, sub, frames_idx, tools)


V.checker = _checker2
_pc = CL.plan_call


def plan_call(ep, instr, demos, history=None):
    subs = _pc(ep, instr, demos, history)
    for s in subs or []:
        if any("task_success" in str(c) for c in s.get("success", [])) and "[overall task:" not in s["goal"]:
            s["goal"] = s["goal"] + f" [overall task: {instr}]"
    return subs


CL.plan_call = plan_call


# ------------------------------------------------------------------ demos
def x_demo(task):
    f = PILOT / "demos" / f"{task}_X.json"
    if f.exists():
        return json.loads(f.read_text())
    src = XSRC[task]
    V.V3 = PILOT
    d = V._demo_text(src, "S4")
    for k in ("planner", "executor"):
        d[k] = d[k].replace("of the same task", "of a DIFFERENT task of the same kind (" + src + ")")
    d["demo_task"] = src
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, ensure_ascii=False))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cond", choices=["Z", "X"], required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--api", default=_paths.DEFAULT_URL)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--tools", default="v1", choices=["v1", "v2"])
    a = ap.parse_args()
    assert 0 <= a.episode <= 2
    if a.tools == "v2":
        REPL.extend(REPL_V2)
        G50Tools.V2 = True
    print("prompt patches", patch_prompts(), "tools", a.tools, flush=True)
    V.NEXT_LINE = True
    V.HOME_RULE = V.HOME_RULE_TEXT
    CL.agent_loop = V.agent_loop
    CL.Tools = G50Tools
    demos = {"planner": "", "executor": "", "checker": ""} if a.cond == "Z" else x_demo(a.task)
    demos.setdefault("checker", "")
    V.CHECKER_DEMO = demos["checker"]
    sim = CL.SimEnv(a.task, seed_dir=str(SEEDS), task_config="demo_randomized", seed_base_index=0)
    root = P / a.lane / a.task / f"shard_{a.episode}"
    od = root / f"episode_{a.episode:03d}"
    CL.OUT = root
    r = CL.run_episode(sim, a.task, a.cond, a.episode, od, a.api, "qwen", demos)
    keep = ("episode_index", "seed", "instruction", "success", "finished_reason", "seconds", "model_calls", "prompt_tokens",
            "completion_tokens", "agent_claimed_done")
    ep = {k: r.get(k) for k in keep}
    ep["demo_task"] = demos.get("demo_task")
    (od / "result.json").write_text(json.dumps(r, ensure_ascii=False, default=str))
    (root / "results.json").write_text(json.dumps({"framework": a.lane.split("__")[0], "model": "qwen", "task": a.task,
                                                   "episodes": [ep]}, ensure_ascii=False, indent=1))
    print(json.dumps(ep), flush=True)
    try:
        sim.rd.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
