#!/usr/bin/env python
"""V+ : our v3 harness (planner / executor / checker, g50 bimanual skills, tools v2, oracle text, zero-shot, Flash)
upgraded for round 2, in autonomous-termination mode (sim success logged, hidden, never ends the episode).

Upgrades (both arms of the paired run):
  * cmd(commands): RoboDawn's low-level closed-loop commands (move/rotate/point/gripper/home, grid-annotated views as
    before) are always available to the executor, plus RoboDawn's practical rules (stated in our frame).
  * skills report their stage; on failure the executor corrects with cmd/move from the current state (rule) and a failed
    subgoal leads to a replan instead of giving up (up to 8 replans, then the budget decides).
  * open_articulated(arm, object, fraction): constrained hinge-arc / drawer-slide opening in small steps (slip and
    stall detection, progress in %).
  * unreachable: before an arm acts, the other (idle, empty) arm is parked; after 2 consecutive unreachable results
    the feedback tells the executor to switch arm / use a waypoint / end the subgoal for a replan.
  * step size is chosen from feedback only (rule text; nothing keyed on the task name).
--guard (arm VPg): success-candidate guard with task checker v2.
  After every skill (grasp/place/press/open_articulated) the task checker judges (previous, current): satisfied or
  unknown -> LOCK (grasp/place/open/home and motions > 3 cm refused) + one observe-only verify (empty open arms parked,
  new frames, nothing manipulated); the lock stays while the verify says satisfied/unknown, unsatisfied lifts it.
  A post-skill check NEVER ends the episode. The episode ends only on a FINAL CLAIM (the last subgoal confirmed by the
  subgoal checker, or the planner has nothing left) when the task checker says satisfied (directly or after one verify).
Without --guard (arm VPn): the old request_check flow (subgoal checker g; episode ends when every subgoal is confirmed
and the planner has nothing left).
Budget: 120 model calls, 720 s wall (timeout = failure). Output: results/unibench/<lane>/<task>/shard_<k>/..."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ub_auto_common as U  # noqa: E402
import g50_cl as G  # noqa: E402

V, CL, RV2 = G.V, G.CL, G.RV2
VB = V.VB
from record import ArmTag  # noqa: E402
from harness.core.commands import parse_command_list  # noqa: E402

A = None                      # parsed args
STATE = {"ever": False, "first": None, "guard": [], "term": None, "lock": False, "refused": 0}
UNREACH_RX = re.compile(r"could not reach|no reachable|cannot be reached|unreachable|could not move|planner failed", re.I)
SKILLS = ("grasp", "place", "press", "open_articulated")


class TaskDone(CL.Success):
    pass


# ---------------------------------------------------------------- snapshots: log the hidden sim success, never stop
_snap = CL.Episode.snap


def snap(self, label):
    try:
        return _snap(self, label)
    except TaskDone:
        raise
    except CL.Success:
        if not STATE["ever"]:
            STATE["ever"], STATE["first"] = True, {"state": len(self.states) - 1, "label": label, "t": round(time.time() - self.t0, 1)}
        if time.time() - self.t0 > CL.WALL_S:
            raise CL.Budget("infra_timeout")
        return self.states[-1]


CL.Episode.snap = snap
EP = {}
_ep_init = CL.Episode.__init__


def _ep_init2(self, *a, **k):
    _ep_init(self, *a, **k)
    EP["ep"] = self


CL.Episode.__init__ = _ep_init2


def sim_ok(ep):
    try:
        return bool(ep.sim.T.check_success())
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- subgoal checker g (both arms) and task checker
def checker_g(ep, sub, frames_idx, tools):
    S = ep.states
    smp = {"episode_id": f"_live_{ep.task}_{ep.idx}", "task": ep.task, "goal": sub["goal"], "arm_view": sub["arm"],
           "arm_main": sub["arm"] if sub["arm"] in ("left", "right") else sub.get("arm_main", "right"),
           "criteria": sub["success"] or ["gripper_open(left)"],
           "pool": [{"idx": i, "stage": f"after step {k}"} for k, i in enumerate(frames_idx)], "current": frames_idx[-1], "checks": {}}
    msgs = U.strip(U.g_messages(smp, S, ep.dir, live_checks=True, crops=True))
    nc = len(smp["criteria"])
    text = V._chat_json(ep, msgs, nc)
    o = U.E.parse(text) or {"decision": "unsatisfied", "criteria": [], "next_check": None}
    if o.get("decision") == "unknown" and o.get("next_check"):
        res = V.live_check(ep, smp, o["next_check"], tools)
        follow = res or [CL.txt(f"CHECK {o['next_check']}: not possible here. Give your final answer with the evidence you have.")]
        msgs2 = msgs + [{"role": "assistant", "content": text},
                        {"role": "user", "content": follow + [CL.txt('Give the final answer now ("next_check": null).')]}]
        o = U.E.parse(V._chat_json(ep, U.strip(msgs2), nc)) or o
    if (A.guard or A.final_check) and o.get("decision") == "satisfied" and sub.get("id") == STATE.get("last_id") and getattr(tools, "guard", None):
        tools.guard("final claim")   # the agent claims the last subgoal: the task checker decides termination
    return {"checker": o.get("decision"), "criteria": o.get("criteria") or [], "next_check": o.get("next_check")}, 1


G._checker = checker_g


def task_check(ep, prev, cur):
    """task-level completion check on (prev, cur) states. Returns {decision, confidence, evidence}."""
    instr = getattr(ep, "instr", "")
    if A.task_checker == "v2":
        import checker_v2_iface as CV2   # adapter to the validated checker v2 (written when READY appears)
        o = CV2.check(ep, prev, cur, instr)
        ep.llm.calls += int(o.get("_calls", 1))
        return o
    smp = {"episode_id": f"_live_{ep.task}_{ep.idx}", "task": ep.task, "goal": instr, "arm_view": "both", "arm_main": "right",
           "criteria": ["task_success()"], "pool": [{"idx": prev, "stage": "previous"}, {"idx": cur, "stage": "now"}],
           "current": cur, "checks": {}}
    msgs = U.strip(U.g_messages(smp, ep.states, ep.dir, live_checks=False, crops=True))
    o = U.E.parse(V._chat_json(ep, msgs, 1)) or {"decision": "unsatisfied"}
    ev = "; ".join(str(c.get("evidence", "")) for c in (o.get("criteria") or []) if isinstance(c, dict))
    return {"decision": o.get("decision"), "confidence": o.get("confidence"), "evidence": ev[:240]}


# ---------------------------------------------------------------- tools
class VPTools(G.G50Tools):
    V2 = True

    def __init__(self, ep):
        super().__init__(ep)
        self.last_arm = None
        self.unreach = 0

    # ---- low-level commands (RoboDawn's closed loop)
    def t_cmd(self, commands, arm=None):
        if isinstance(commands, str):
            commands = [commands]
        commands = list(commands)[:4]
        if STATE["lock"]:
            for c in commands:
                m = re.match(r"\s*(left|right)\s+move\s+[xyz]\s+([-+]?[\d.]+)", str(c), re.I)
                if (m and abs(float(m.group(2))) > 3) or re.search(r"\b(home|point|rotate)\b", str(c), re.I):
                    S = self.ep.states[-1]
                    return RV2.fb(S, False, "refused", "GUARD: the task may already be complete; only small (<= 3 cm) "
                                                       "corrections or request_check are allowed now"), S
        notes, ok_all = [], True
        env = self.sim.rd
        for c in commands:
            cmds, errs = parse_command_list([c])
            if errs:
                notes.append(f"{c}: invalid ({errs[0]})")
                ok_all = False
                continue
            cmd = cmds[0]
            if cmd.kind == "done":
                notes.append("done: not a command here - use request_check")
                continue
            r = env.execute(cmd)
            self.T.eval_success = False
            self.T.plan_success = True
            if cmd.arm:
                self.last_arm = cmd.arm
            ok_all &= bool(r.get("ok"))
            notes.append(f"{cmd.text()}: {'ok' if r.get('ok') else 'FAILED'}{(' - ' + str(r.get('note'))[:160]) if r.get('note') else ''}")
        S = self._state()
        return RV2.fb(S, ok_all, "done", " | ".join(notes)), S

    # ---- constrained articulated opening
    def t_open_articulated(self, arm, object, fraction=0.9):
        from envs.utils.actor_utils import ArticulationActor
        import transforms3d as t3d
        T = self.T
        actor = self._obj(object)
        tag = self._arm(arm)
        if not isinstance(actor, ArticulationActor):
            S = self.ep.states[-1]
            return RV2.fb(S, False, "error", f"{object} has no hinge or slide"), S
        frac = min(max(float(fraction if fraction is not None else 0.9), 0.1), 1.0)
        grasp_note = ""
        cid_used = None
        if not any(o == object for a_, o, *_ in self.sim.contacts() if a_ == arm) or self.ep.states[-1]["arms"][arm]["gripper_cmd"] > 0.5:
            # not holding the part yet: grasp it at one of the object's annotated contact points (tried in order;
            # the first that plans and ends in contact is kept) - generic, no task-specific ids
            try:
                ids = [i for i, _ in actor.iter_contact_points("list")]
            except Exception:  # noqa: BLE001
                ids = [None]
            got = False
            for cid in ids[:6]:
                self._run(T.open_gripper(tag))
                act = T.grasp_actor(actor, tag, pre_grasp_dis=0.08, contact_point_id=cid)
                if act is None or act[0] is None or not act[1]:
                    T.plan_success = True
                    continue
                ok = self._run(act)
                if ok and any(o == object for a_, o, *_ in self.sim.contacts() if a_ == arm):
                    got = True
                    cid_used = cid
                    grasp_note = f"grasped the part at contact point {cid}; "
                    break
                self._run(T.open_gripper(tag))
                self._run(T.move_by_displacement(tag, z=0.05))
            if not got:
                S = self._state()
                return RV2.fb(S, False, "grasp", f"could not grasp a handle / edge of {object} with the {arm} arm "
                                                 "(try the other arm or grasp it first with grasp / cmd)"), S
            self._state()
        jt = actor.actor.get_active_joints()[0]
        lo, hi = [float(v) for v in actor.get_qlimits()[0]]
        revolute = "revol" in str(jt.type).lower()
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = 0.0, (math.pi if revolute else 0.3)
        rng = hi - lo
        q0 = q = float(actor.get_qpos()[0])
        goal = lo + frac * rng
        jp = jt.get_global_pose()
        axis = t3d.quaternions.quat2mat(np.asarray(jp.q))[:, 0]
        hinge = np.asarray(jp.p)
        if cid_used is None:            # the model grasped it: use the annotated contact point nearest to the fingertips
            try:
                tp = np.asarray(T.get_arm_pose(tag), float)[:3]
                cid_used = min(((i, np.linalg.norm(np.asarray(p_)[:3] - tp)) for i, p_ in actor.iter_contact_points("list")),
                               key=lambda x: x[1])[0]
            except Exception:  # noqa: BLE001
                cid_used = None
        sign, flips, stalls, note, steps, regrasps = 1.0, 0, 0, "", 0, 0
        for steps in range(1, 41):
            if q >= goal:
                break
            pose = np.asarray(T.get_arm_pose(tag), float)
            if revolute:
                R = t3d.axangles.axangle2mat(axis, sign * math.radians(6))
                p2 = hinge + R @ (pose[:3] - hinge)
                q2 = t3d.quaternions.mat2quat(R @ t3d.quaternions.quat2mat(pose[3:]))
                ok = self._run(T.move_to_pose(tag, list(p2) + list(q2)))
            else:
                d = axis * sign * 0.02
                ok = self._run(T.move_by_displacement(tag, x=d[0], y=d[1], z=d[2]))
            qn = float(actor.get_qpos()[0])
            touching = any(o == object for a_, o, *_ in self.sim.contacts() if a_ == arm)
            if not ok:
                note = "the arm could not follow the opening path"
                break
            if qn < q - 1e-3 and flips == 0:
                sign, flips = -sign, 1
            elif qn - q < 0.01 * rng:
                stalls += 1
                if (stalls >= 2 or not touching) and regrasps < 3 and cid_used is not None:
                    # the grip slipped / stalled part-way: re-grasp the same (moved) contact point and keep going
                    regrasps += 1
                    stalls = 0
                    self._run(T.open_gripper(tag))
                    act = T.grasp_actor(actor, tag, pre_grasp_dis=0.05, contact_point_id=cid_used)
                    if act is None or act[0] is None or not act[1] or not self._run(act):
                        T.plan_success = True
                        note = "slipped and the re-grasp failed"
                        break
                    q = float(actor.get_qpos()[0])
                    continue
                if stalls >= 2 or not touching:
                    note = "no progress" + ("" if touching else " - the grip is not on the part (slipped?)")
                    break
            else:
                stalls = 0
            q = qn
        if regrasps:
            note = (note + "; " if note else "") + f"re-grasped {regrasps}x"
        S = self._state()
        ok = q >= goal - 0.02 * rng
        return RV2.fb(S, ok, "opened" if ok else "partial",
                      grasp_note + f"{object} at {(q - lo) / rng:.0%} of its range (from {(q0 - lo) / rng:.0%}) after {steps} steps"
                      + (f"; {note}" if note else "")), S

    # ---- place with automatic arm switch when the place pose is out of reach
    def t_place(self, arm, object, target, relation="on", release=True, retreat=True, **kw):
        fb, S = super().t_place(arm, object, target, relation, release, retreat, **kw)
        other = "left" if arm == "right" else "right"
        if fb.get("ok") or not UNREACH_RX.search(str(fb.get("note", ""))) or self.held(S, arm) != object or self.held(S, other):
            return fb, S
        # relay: set the object down near the table centre line, park this arm, regrasp with the other arm, place again
        try:
            tag = ArmTag(arm)
            tcp = S["arms"][arm]["tcp"]
            dx = (0.0 - tcp[0]) * 0.8 / 100.0
            self._run(self.T.move_by_displacement(tag, x=dx))
            self._run(self.T.move_by_displacement(tag, z=-max(tcp[2] - S["objects"][object]["size"][2] / 2 - 2, 0) / 100.0))
            self._run(self.T.open_gripper(tag))
            self._run(self.T.move_by_displacement(tag, z=0.08))
            self._run(self.T.back_to_origin(tag))
            self._state()
            fg, S2 = super().t_grasp(other, object, None, None)
            if not fg.get("ok"):
                return RV2.fb(S2, False, "relay", f"{arm} arm could not reach the place pose; set {object} down near the "
                                                  f"centre for the {other} arm, but the {other} grasp failed: {fg.get('note')}"), S2
            fb2, S3 = super().t_place(other, object, target, relation, release, retreat, **kw)
            fb2["note"] = f"(the {arm} arm could not reach; harness relayed {object} to the {other} arm) " + str(fb2.get("note", ""))
            self.last_arm = other
            return fb2, S3
        except (CL.Success, CL.Budget):
            raise
        except Exception as exc:  # noqa: BLE001
            self.T.plan_success = True
            S = self._state()
            return RV2.fb(S, False, "relay", f"place failed (out of reach) and the arm relay failed: {exc}"), S

    # ---- dispatch: lock, park idle arm, unreachable rule, guard
    def call(self, name, args):
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        if STATE["lock"] and STATE["refused"] >= 2:      # the executor insists twice: release (logged), avoid a deadlock
            STATE["lock"], STATE["refused"] = False, 0
            self.ep.trace({"kind": "guard_release"})
        if STATE["lock"] and (name in ("grasp", "place", "open_articulated", "home") or
                              (name == "move" and max(abs(float(args.get(k) or 0)) for k in ("dx", "dy", "dz")) > 3)):
            S = self.ep.states[-1]
            STATE["refused"] += 1
            return RV2.fb(S, False, "refused", "GUARD: the task may already be complete - big motions and re-grasps are "
                                               "blocked; use request_check (or <= 3 cm corrections)"), S
        arm = args.get("arm")
        park = ""
        if arm in ("left", "right") and name not in ("locate", "home"):
            other = "left" if arm == "right" else "right"
            S0 = self.ep.states[-1]
            if self.last_arm == other and not self.held(S0, other) and S0["arms"][other]["gripper_cmd"] > 0.8:
                try:
                    self._run(self.T.back_to_origin(ArmTag(other)))
                    self._state()
                    park = f"(harness parked the idle {other} arm first) "
                except (CL.Success, CL.Budget):
                    raise
                except Exception:  # noqa: BLE001
                    pass
        if arm in ("left", "right"):
            self.last_arm = arm
        fb, S = super().call(name, args)
        if park:
            fb["note"] = park + str(fb.get("note", ""))
        if not fb.get("ok") and UNREACH_RX.search(str(fb.get("note", ""))):
            self.unreach += 1
        elif fb.get("ok"):
            self.unreach = 0
        if self.unreach >= 2:
            fb["note"] = str(fb.get("note", "")) + (" | HARNESS: 2 consecutive unreachable results - switch to the other arm "
                                                    "(the idle arm is parked automatically), go through a waypoint (up / toward "
                                                    "the robot first), or end this subgoal so the planner replans; do not retry "
                                                    "the same target")
            self.unreach = 0
        if A.guard and name in SKILLS:
            self.guard(f"after {name}")
        return fb, S

    # ---- success-candidate guard
    def guard(self, why):
        """Termination ONLY on a final claim (last subgoal confirmed / planner has nothing left) with the task checker
        satisfied (after at most one observe-only verify). A post-skill check that says satisfied or unknown never
        terminates: it freezes big motions (LOCK) and runs one observe-only verify; unsatisfied lifts the lock."""
        ep = self.ep
        keep = [i for i, s in enumerate(ep.states) if not str(s["label"]).startswith("check")]
        if len(keep) < 2:
            return
        final = why == "final claim"
        o = task_check(ep, keep[-2], keep[-1])
        rec = {"why": why, "decision": o.get("decision"), "evidence": o.get("evidence"), "sim": sim_ok(ep), "state": keep[-1]}
        STATE["guard"].append(rec)
        ep.trace({"kind": "guard", **rec})
        d = o.get("decision")
        if final and d == "satisfied":
            STATE["term"] = "agent_done_checked"
            raise TaskDone()
        if d in ("satisfied", "unknown"):
            STATE["lock"], STATE["refused"] = bool(A.guard), 0          # freeze big motions before verifying
            S = ep.states[-1]
            for a in ("left", "right"):
                # observe-only: only arms with an OPEN, empty gripper are parked (a closed gripper may hold a handle / object)
                if S["arms"][a]["gripper_cmd"] > 0.8 and not self.held(S, a):
                    try:
                        self._run(self.T.back_to_origin(ArmTag(a)))
                    except (CL.Success, CL.Budget):
                        raise
                    except Exception:  # noqa: BLE001
                        pass
            ep.snap("check verify")
            o2 = task_check(ep, keep[-1], len(ep.states) - 1)
            d2 = o2.get("decision")
            rec2 = {"why": "verify" + (" (final)" if final else ""), "decision": d2, "evidence": o2.get("evidence"),
                    "sim": sim_ok(ep), "state": len(ep.states) - 1}
            STATE["guard"].append(rec2)
            ep.trace({"kind": "guard", **rec2})
            if final and d2 == "satisfied":
                STATE["term"] = "agent_done_checked"
                raise TaskDone()
            STATE["lock"] = A.guard and d2 in ("satisfied", "unknown")
        else:
            STATE["lock"], STATE["refused"] = False, 0


VP_RULES = """
- cmd(commands, arm=null) - LOW-LEVEL closed-loop commands, always available, executed in order (max 4 strings), e.g.
  ["left point down", "left move z -3", "left rotate yaw 30", "left gripper close"]. Grammar: <arm> move x|y|z <cm>
  (|cm| <= 20, world axes: x right, y away from the robot, z up) | <arm> rotate roll|pitch|yaw <deg> | <arm> point
  down|forward|down45 | <arm> gripper open|close|<0..1> | <arm> home. Each reports whether the planner reached the target.
- open_articulated(arm, object, fraction=0.9) - open an articulated object (door, lid, drawer) along its hinge arc or
  drawer slide in small steps. If the gripper is not already on the part, the skill grasps the handle / edge itself.
  Stops on slip or no progress and reports the opening in % of its range. Prefer it over hand-made pulling motions.
- place automatically relays the object to the other arm when the place pose is out of reach of the holding arm.

Practical rules:
- Skills (grasp, place, press, open_articulated) are the fast path. When a skill fails, read its stage and the measured
  state, correct from there with cmd / move (re-align 1-3 cm, change yaw, approach from above), then retry or finish the
  motion with cmd. Never repeat an identical failed call.
- Choose the step size from feedback and geometry only: free space -> one decisive move; within 5 cm of an object, after
  a contact or a partial-progress report -> 1-3 cm steps; never more than 6 consecutive steps of <= 2 cm.
- Travel sideways with the fingertips >= 14 cm above the table; descend only when above the target.
- Reach: the left arm works at x in [-45, +15], the right arm at x in [-15, +45]; use the arm on the object's side.
- Unreachable twice in a row: switch arm (the idle arm is parked for you), go through a waypoint (up / toward the robot
  first), or end the subgoal (request_check) so the planner replans.
- Placing: lower until the object almost touches the support before opening; dropping from >= 3 cm tips objects over.
- Before closing on an object the wrist camera must show it centred between the fingers; after a lift check it rose.
- Once the task looks done, stop moving and call request_check: extra motion can undo it."""


def agent_loop(ep, tools, instr, demos):
    mem = VB.Memory()
    mem.M["notes"] = []
    ep.mem = mem
    ep.instr = instr
    history, replans = [], 0
    subs = CL.plan_call(ep, instr, demos)
    for k, s in enumerate(subs or []):
        s["id"] = k + 1
    all_ok = False
    while True:
        if not subs:
            if all_ok and (A.guard or A.final_check):
                tools.guard("final claim")          # raises TaskDone when satisfied
                history.append("- the task checker says the overall task is NOT complete yet: "
                               + str((STATE["guard"][-1] or {}).get("evidence", ""))[:200])
                all_ok = False
            elif all_ok:
                STATE["term"] = "agent_done_checked"
                return
            else:
                STATE["term"] = STATE["term"] or "planner_empty"
                if replans > 8:
                    return
        else:
            failed = False
            STATE["last_id"] = subs[-1]["id"]
            for k, sub in enumerate(subs):
                if sub["id"] in mem.M["confirmed_subgoals"]:
                    continue
                res = V.run_subgoal(ep, tools, instr, sub, k, len(subs), demos, mem, subs[k + 1] if k + 1 < len(subs) else None)
                history.append(f"- subgoal '{sub['goal']}': {res}")
                if res != "done":
                    failed = True
                    break
            all_ok = not failed
        replans += 1
        if replans > 8:
            STATE["term"] = "gave_up"
            return
        mem.M["confirmed_subgoals"] = []
        subs = CL.plan_call(ep, instr, demos, history[-12:] + (["(all subgoals confirmed by the checker)"] if all_ok else []))
        for k, s in enumerate(subs or []):
            s["id"] = k + 1


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--api", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--final_check", action="store_true", help="final claim needs the task checker (no post-skill guard)")
    ap.add_argument("--task_checker", default="g", choices=["g", "v2"])
    ap.add_argument("--wall", type=float, default=720)
    A = ap.parse_args()
    G.REPL.extend(G.REPL_V2)
    G.G50Tools.V2 = True
    G.patch_prompts()
    VB.EXECUTOR_SYSTEM_V3 = VB.EXECUTOR_SYSTEM_V3.replace(
        "- request_check(subgoal_id)", VP_RULES.strip("\n").replace("\n", "\n", ) + "\n- request_check(subgoal_id)", 1) \
        if "- request_check(subgoal_id)" in VB.EXECUTOR_SYSTEM_V3 else VB.EXECUTOR_SYSTEM_V3 + VP_RULES
    V.NEXT_LINE = True
    V.HOME_RULE = V.HOME_RULE_TEXT
    CL.WALL_S = A.wall
    CL.agent_loop = agent_loop
    CL.Tools = VPTools
    demos = {"planner": "", "executor": "", "checker": ""}
    V.CHECKER_DEMO = ""
    sim = CL.SimEnv(A.task, seed_dir=str(G.SEEDS), task_config="demo_randomized", seed_base_index=0)
    root = U.P / "runs" / A.lane / A.task / f"shard_{A.episode}"
    od = root / f"episode_{A.episode:03d}"
    CL.OUT = root
    (od).mkdir(parents=True, exist_ok=True)
    (od / "executor_system.txt").write_text(VB.EXECUTOR_SYSTEM_V3)
    r = CL.run_episode(sim, A.task, "Z", A.episode, od, A.api, "qwen", demos)
    final_ok = sim_ok(EP["ep"]) if "ep" in EP else False
    fin = r.get("finished_reason")
    if fin == "success":                    # TaskDone (a CL.Success) from the guard
        fin = STATE["term"] or "agent_done_checked"
    elif fin == "agent_done":
        fin = STATE["term"] or "agent_done"
    elif fin == "infra_timeout":
        fin = "wall_timeout"
    agent_term = fin == "agent_done_checked"
    ever = STATE["ever"] or final_ok
    ep = {k: r.get(k) for k in ("episode_index", "seed", "instruction", "seconds", "model_calls", "prompt_tokens", "completion_tokens")}
    ep.update(finished_reason=fin, success=bool(agent_term and final_ok), oracle_success=ever, final_sim_success=final_ok,
              agent_terminated=agent_term, premature=bool(agent_term and not final_ok and not ever),
              premature_after_break=bool(agent_term and not final_ok and ever), succeeded_then_broke=bool(ever and not final_ok),
              first_success=STATE["first"], guard_log=STATE["guard"], guard=A.guard, task_checker=A.task_checker,
              error=r.get("error"))
    (od / "result_auto.json").write_text(json.dumps(ep, indent=1, default=str))
    (root / "results.json").write_text(json.dumps({"framework": A.lane.split("__")[0], "model": "qwen", "task": A.task,
                                                   "episodes": [ep]}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: ep[k] for k in ("episode_index", "success", "oracle_success", "finished_reason", "seconds")}), flush=True)
    try:
        sim.rd.close()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
