#!/usr/bin/env python
"""v3 build: episodes (clean + branches) -> executor conversations (with harness memory and request_check / checker
verdicts), checker samples (round-2 'e' input), planner jobs. Output: <out>/build/{conversations,checker,planner}.json
Annotation (seen/reported/assessment/uncertain/memory_delta/next) is added later by v3_annotate.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_v2 as RV2  # noqa: E402
import common_v2 as C  # noqa: E402
import ce1_build as LB  # noqa: E402  (observable labels)
import clean_v2  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/oracle_expert_v3")
TOOLS_V3 = C.TOOL_DOC.replace(
    "- home(arm) - back the arm off and move it to its start pose, out of the way.",
    "- home(arm) - back the arm off and move it to its start pose, out of the way.\n"
    "- request_check(subgoal_id) - ask the checker whether the current subgoal is achieved. Only the checker can complete a\n"
    "  subgoal: it answers {\"checker\": \"satisfied|unsatisfied|unknown\", \"criteria\": [...], \"next_check\": ...}.").replace(
    "(grasp adds \"lifted_cm\", place adds \"relation\", locate adds \"object\")",
    "(grasp adds \"lifted_cm\" and, when holding, \"object_offset_cm\" = object centre minus fingertip centre; place adds "
    "\"relation\"; locate adds \"object\")")
EXEC_OUT = """Answer with ONE JSON object only:
{"seen": ["<= 2 short facts visible in the images (say which view)"],
 "reported": ["<= 2 short facts from the measured state, tool feedback, checker or memory"],
 "assessment": "what the last step actually achieved: succeeded / failed / partly / unknown, and why (one sentence)",
 "uncertain": ["<= 1 thing that would change the next action, or empty"],
 "memory_delta": ["<= 2 new facts worth remembering (working parameters, failure conditions, outdated estimates), or empty"],
 "next": "purpose of the next call and the observable change you expect (one sentence)",
 "tool_call": {"name": "...", "arguments": {...}}}
Keep the text short (about 60-120 words in total). The subgoal is finished only when the checker answers "satisfied"
to request_check; you never declare completion yourself."""
EXECUTOR_SYSTEM_V3 = ("You are the robot arm controller (executor) of a two-arm robot (left arm, right arm). You carry out ONE "
                      "subgoal at a time given by the planner. Each turn you get the images, the measured state, the tool "
                      "feedback and the harness MEMORY (maintained by the harness; you only add memory_delta).\n\n"
                      + TOOLS_V3 + "\n\n" + EXEC_OUT)


def memory_text(M):
    return "MEMORY: " + json.dumps(M, ensure_ascii=False)


def offset_of(S, arm, obj):
    o, t = S["objects"][obj]["center"], S["arms"][arm]["tcp"]
    return [round(o[k] - t[k], 1) for k in range(3)]


def grasp_event(ep, c):
    """the recorded grasp_actor event of a grasp call (matched by its closed-gripper state)."""
    for e in ep["events"]:
        if e["api"] == "grasp_actor" and e.get("after") == c.get("closed"):
            return e
    return None


def enrich_feedback(ep, c):
    """grasp feedback gets the measured object offset in the hand and the centring error = how far the object sits from
    where the grasp skill planned it (actual offset minus planned offset; the skill knows its planned grasp point)."""
    fb = dict(c["feedback"])
    if c["tool"] == "grasp":
        S = ep["states"]
        arm, obj = c["args"]["arm"], c["args"]["object"]
        Sc = S[c.get("closed", c["after"])]
        if Sc["held"].get(arm) == obj:
            off = offset_of(Sc, arm, obj)
            e = grasp_event(ep, c)
            if e is not None:
                nom = e.get("nominal_offset_cm")
                if nom is None:
                    o0 = S[e["before"]]["objects"][obj]["center"]
                    nom = [o0[k] - e["grasp_tcp"][k] for k in range(3)]
                err = ((off[0] - nom[0]) ** 2 + (off[1] - nom[1]) ** 2) ** 0.5
                fb["grasp_centering_error_cm"] = round(err, 1)
            fb["object_offset_cm"] = offset_of(S[c["after"]], arm, obj) if S[c["after"]]["held"].get(arm) == obj else off
    return fb


class Memory:
    def __init__(self):
        self.M = {"confirmed_subgoals": [], "holding": {}, "grasp_params": [], "recent_failures": []}

    def copy(self):
        m = Memory()
        m.M = json.loads(json.dumps(self.M))
        return m

    def after_tool(self, c, S):
        M = self.M
        M["holding"] = {a: {"object": S["held"][a], "offset_cm": offset_of(S, a, S["held"][a])} for a in ("left", "right")
                        if S["held"].get(a) and S["arms"][a]["gripper_cmd"] <= 0.8}
        fb = c["feedback"]
        if c["tool"] == "grasp" and fb["ok"]:
            gp = {"object": c["args"]["object"], "arm": c["args"]["arm"], "part": c["args"].get("part"),
                  "lift_cm": c["args"].get("lift_cm", "auto"), "offset_cm": M["holding"].get(c["args"]["arm"], {}).get("offset_cm")}
            M["grasp_params"] = (M["grasp_params"] + [gp])[-3:]
        if not fb["ok"] and c["tool"] not in ("locate",):
            key = (c["tool"], c["args"].get("object"), c["args"].get("arm"))
            for f in M["recent_failures"]:
                if (f["tool"], f.get("object"), f.get("arm")) == key:
                    f["count"] += 1
                    f["note"] = fb["note"][:80]
                    break
            else:
                M["recent_failures"] = (M["recent_failures"] + [{"tool": c["tool"], "object": c["args"].get("object"),
                                                                 "arm": c["args"].get("arm"), "note": fb["note"][:80], "count": 1}])[-3:]

    def confirm(self, sid):
        if sid not in self.M["confirmed_subgoals"]:
            self.M["confirmed_subgoals"].append(sid)


# ----------------------------------------------------------------------------- checker verdicts
def verdict(ep, sub, start, frames_idx, current, cams_override=None, resolve=True):
    """Checker answer at `current` given the subgoal's frames; active checks are resolved (the harness runs the check),
    so the verdict is the observable status with all cameras of the current state."""
    S = ep["states"]
    pool = [LB.F(i, "frame") for i in frames_idx]
    if current not in frames_idx:
        pool.append(LB.F(current, "now"))
    smp = {"criteria": sub["success"], "current": current, "pool": pool, "arm_main": sub["arm"] if sub["arm"] in ("left", "right") else
           (sub.get("arm_main") or "right")}
    res, dec, phys = LB.label(smp, S)
    crit = []
    for r in res:
        ev = C.evaluate(r["criterion"], S[current], S[0])[1] if r["observable"] != "unknown" else "not visible in any view"
        crit.append({"criterion": r["criterion"], "status": r["observable"], "evidence": ev})
    nxt = None
    out = {"checker": dec, "criteria": crit, "next_check": None}
    if dec == "unknown" and resolve:
        # the checker asks the harness to move the arm away / look again: use the first later clean state in which the
        # objects have not moved and the decision is observable
        objs = sorted({x for c in sub["success"] for x in C.parse(c)[2] if isinstance(x, str) and x in S[current]["objects"]})
        branch_states = {i for b in ep.get("branches", []) for i in range(b["start_state"] + 1, b["end_state"] + 2)}
        # branch rollouts sit between clean states: skip them (do not count them towards the 5-state look-ahead),
        # otherwise a subgoal that ends right before a branch never gets resolved and its clean conversation is dropped
        n_clean = 0
        for j in range(current + 1, len(S)):
            if j in branch_states:
                continue
            n_clean += 1
            if n_clean > 5:
                break
            if any(max(abs(S[j]["objects"][o]["center"][k] - S[current]["objects"][o]["center"][k]) for k in range(3)) > 1.0 for o in objs):
                break
            v2, p2 = verdict(ep, sub, start, frames_idx + [j], j, resolve=False)
            if v2["checker"] != "unknown":
                v2["resolved_by"] = "move_arm_away"
                v2["first_answer"] = {"checker": "unknown", "next_check": "move_arm_away"}
                return v2, p2
        out["next_check"] = "move_arm_away"
    return out, phys


# ----------------------------------------------------------------------------- conversations
def translate_branch(ep, b):
    S, E = ep["states"], ep["events"]
    evs = [E[i] for i in b["event_idx"] if not E[i].get("injected")]
    special = []
    norm = []
    for e in evs:
        if e["api"] == "place_not_holding":
            special.append(e)
            norm.append({"api": "__marker__", "idx": len(special) - 1})
        else:
            norm.append(e)
    calls = []
    chunk = []

    def flush():
        if chunk:
            for c in RV2.translate(S, chunk):
                if c["tool"] == "locate" and c.get("synthetic") and not c.get("recovery") and calls == []:
                    continue
                calls.append(c)
            chunk.clear()
    for e in norm:
        if e["api"] == "__marker__":
            flush()
            sp = special[e["idx"]]
            St = S[sp["after"]]
            args = {"arm": sp["arm"], "object": sp["object"], "target": sp.get("target") or "table_center"}
            calls.append({"tool": "place", "args": args, "before": sp["before"], "after": sp["after"], "mid": None,
                          "expert_api": [], "recovery": False,
                          "feedback": RV2.fb(St, False, "reach", f"the {sp['arm']} arm is not holding {sp['object']}: nothing was moved")})
        else:
            chunk.append(e)
    flush()
    return calls


def keep_home_after_place(subs, S):
    """r3: RV2.group starts a new subgoal at every home(), so the expert's post-place home(same arm) ended up at the head
    of the NEXT subgoal (place -> request_check | home -> locate -> grasp). Move it back to the end of the place subgoal so
    the conversation is place -> home -> request_check, as in the expert trajectory (the arm clears the stack before the
    check and before the other arm comes in). Subgoals are re-described (arm 'both' -> the grasping arm)."""
    out = [dict(x) for x in subs]
    for k in range(1, len(out)):
        prev, cur = out[k - 1], out[k]
        if not cur["calls"] or cur["calls"][0]["tool"] != "home" or prev["kind"] != "place":
            continue
        pl = [c for c in prev["calls"] if c["tool"] == "place"]
        h = cur["calls"][0]
        if not pl or pl[-1]["args"].get("release") is False or pl[-1]["args"]["arm"] != h["args"]["arm"] or len(cur["calls"]) < 2:
            continue
        out[k - 1] = RV2.describe(list(prev["calls"]) + [h], S)
        out[k] = RV2.describe(list(cur["calls"][1:]), S)
    return out


def conversations(ep):
    S = ep["states"]
    clean_ev = [e for e in ep["events"] if e.get("branch") is None]
    calls = RV2.translate(S, clean_ev)
    subs = keep_home_after_place(RV2.group(calls, S), S)
    for k, s in enumerate(subs):
        s["id"] = k + 1
        if s["arm"] == "both":
            s["arm_main"] = next((c["args"]["arm"] for c in reversed(s["calls"]) if c["tool"] in ("place", "grasp")), "right")
    convs = []
    mem = Memory()
    mem_at = {}                                   # state idx -> memory snapshot before the call starting there
    for k, s in enumerate(subs):
        turns = []
        for c in s["calls"]:
            mem_at.setdefault(c["before"], mem.copy())
            c["feedback"] = enrich_feedback(ep, c)
            turns.append({"kind": "tool", "call": c, "memory_before": json.loads(json.dumps(mem.M))})
            mem.after_tool(c, S[c["after"]])
        frames = [s["start"]] + [c["after"] for c in s["calls"]]
        v, phys = verdict(ep, s, s["start"], frames[-3:], s["end"])
        turns.append({"kind": "check", "state": s["end"], "verdict": v, "memory_before": json.loads(json.dumps(mem.M))})
        if v["checker"] == "satisfied":
            mem.confirm(s["id"])
        convs.append({"conv_id": f"{ep['episode_id']}__sg{k+1}", "subgoal": k, "branch": None, "turns": turns,
                      "start": s["start"], "valid": v["checker"] == "satisfied"})
    # branches
    for b in ep.get("branches", []):
        if b.get("actual") in (None, "no_failure") or b.get("error"):
            continue
        restored = b["end_state"] + 1
        k = None
        pre = []
        for kk, s in enumerate(subs):
            if b["moment"].startswith("M5"):
                if s["end"] == b["start_state"] or any(c["after"] == b["start_state"] for c in s["calls"][-1:]):
                    k, pre = kk, list(s["calls"])
            else:
                for j, c in enumerate(s["calls"]):
                    if c["tool"] != "locate" and c["before"] <= restored <= max(c["before"], c.get("closed") or c["before"]):
                        k, pre = kk, s["calls"][:j]
                        break
            if k is not None:
                break
        if k is None:
            continue
        s = subs[k]
        bcalls = translate_branch(ep, b)
        if not bcalls and not b["moment"].startswith("M5"):
            continue
        m0 = mem_at.get(s["start"], Memory()).copy()
        turns = []
        for c in pre:
            turns.append({"kind": "tool", "call": c, "memory_before": json.loads(json.dumps(m0.M))})
            m0.after_tool(c, S[c["after"]])
        if b["moment"].startswith("M5"):
            # request_check sees the idle arm parked over the object (injected) -> unknown -> home(idle) -> check
            inj = [ep["events"][i] for i in b["event_idx"] if ep["events"][i].get("injected")]
            occl_state = inj[-1]["after"] if inj else b["start_state"]
            frames = [s["start"]] + [c["after"] for c in pre]
            v1, _ = verdict(ep, s, s["start"], frames[-2:] + [occl_state], occl_state, resolve=False)
            turns.append({"kind": "check", "state": occl_state, "verdict": v1, "memory_before": json.loads(json.dumps(m0.M)),
                          "before_state": occl_state})
            last = occl_state
        else:
            last = None
        for c in bcalls:
            c["feedback"] = enrich_feedback(ep, c)
            turns.append({"kind": "tool", "call": c, "memory_before": json.loads(json.dumps(m0.M))})
            m0.after_tool(c, S[c["after"]])
            last = c["after"]
        frames = [s["start"]] + [t["call"]["after"] for t in turns if t["kind"] == "tool"]
        crit_sub = dict(s)
        v, phys = verdict(ep, crit_sub, s["start"], frames[-3:], last)
        turns.append({"kind": "check", "state": last, "verdict": v, "memory_before": json.loads(json.dumps(m0.M))})
        convs.append({"conv_id": f"{ep['episode_id']}__sg{k+1}__b{b['id']}_{b['actual']}", "subgoal": k, "branch": b,
                      "turns": turns, "start": s["start"], "valid": v["checker"] == "satisfied" and bool(b.get("recovered"))})
    return subs, convs


def instruction_of(ep):
    fake = {"states": ep["states"], "objects": ep["objects"], "calls": [], "seed": ep["seed"]}
    pool = ep.get("instruction_candidates") or []
    import numpy as np
    instr0 = str(np.random.RandomState(ep["seed"]).choice(pool)) if pool else ep["task"].replace("_", " ")
    d, cf, _ = clean_v2.check(ep["task"], instr0, fake)
    if not d and not cf:
        return instr0, instr0, False
    ok = [c for c in pool if not any(clean_v2.check(ep["task"], c, fake)[:2])]
    if ep["task"] == "handover_block" and any("pad" in c.lower() for c in ok):
        ok = [c for c in ok if "pad" in c.lower()]
    return instr0, (str(np.random.RandomState(ep["seed"]).choice(sorted(set(ok)))) if ok else instr0), True


def main():
    out = OUT / "build"
    out.mkdir(parents=True, exist_ok=True)
    allc, plans, stats = [], [], {}
    for p in sorted((OUT / "episodes").glob("*.json")):
        ep = json.loads(p.read_text())
        subs, convs = conversations(ep)
        i0, instr, fixed = instruction_of(ep)
        for c in convs:
            c.update(episode_id=ep["episode_id"], task=ep["task"], seed=ep["seed"], instruction=instr,
                     subgoal_def={k: v for k, v in subs[c["subgoal"]].items() if k != "calls"}, n_subgoals=len(subs))
        allc += convs
        plans.append({"episode_id": ep["episode_id"], "task": ep["task"], "seed": ep["seed"], "instruction": instr,
                      "instruction_original": i0, "instruction_fixed": fixed,
                      "subgoals": [{k: v for k, v in s.items() if k != "calls"} for s in subs]})
        for b in ep.get("branches", []):
            key = (ep["task"], b["injection"], b.get("actual"), b.get("recovered"))
            stats[str(key)] = stats.get(str(key), 0) + 1
    (out / "conversations.json").write_text(json.dumps(allc, ensure_ascii=False, default=str))
    (out / "planner.json").write_text(json.dumps(plans, ensure_ascii=False))
    n_turn = sum(len(c["turns"]) for c in allc if c["valid"])
    n_rec = sum(len(c["turns"]) for c in allc if c["valid"] and c["branch"])
    print("episodes", len(plans), "conversations", len(allc), "valid", sum(c["valid"] for c in allc),
          "turns(valid)", n_turn, "branch turns", n_rec, f"({100*n_rec/max(n_turn,1):.0f}%)")
    print(json.dumps(stats, indent=0))


if __name__ == "__main__":
    main()
