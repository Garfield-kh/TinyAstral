#!/usr/bin/env python
"""Checker experiment 1 - build samples with two-layer labels from the recorded episodes.

Every sample = (a real simulator state = "current", the evidence pool the checker may use, criteria).
physical_truth  : criteria evaluated on the simulator state (common.evaluate).
observable_status: whether the pool supports the decision, by stage evidence rules (not pixel counts alone):
  holding(arm,o)     satisfied  : the arm rose >=3 cm with the gripper closed and o rose with it (follow evidence), or o hangs
                                  in the air (>=3 cm, nothing under it) with no other arm holding it; o visible in agent view
                     unsatisfied: gripper commanded open (measured state), or the arm rose closed and o stayed
                     else unknown (a closed gripper alone does not show a grasp)
  not holding(arm,o) satisfied  : gripper open and fingertips >6 cm from o in a current frame; unsatisfied: closed and gripping o
  lifted(o,h)        needs a current agent-view frame with o visible
  on(a,b)            current agent-view frame with a visible: gap >2 cm -> unsatisfied; placing arm released -> truth; else unknown
  at(a,x,y)          needs a current top-view frame with a visible
  gripper_open(arm)  measured state -> always observable
decision = unsatisfied if any observable unsatisfied, else unknown if any unknown, else satisfied.
Categories: positive (decision satisfied), hard negative (decision unsatisfied), insufficient evidence (decision unknown).
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import paths as _paths  # noqa: E402
OUT = _paths.OUT / "checker_exp1"
# self-check: after a missed (offset) grasp the object is often visibly knocked away from the fingers, so "unknown" is too
# conservative for U1m -> excluded from the selected sets (kept in samples_all.jsonl)
EXCLUDE_KINDS = {"U1m_closed_no_follow_missed"}
VIS = {"agent": 60, "top": 60, "left": 40, "right": 40}
ALLC = ["agent", "top", "left", "right"]


def vis(S, o, cam):
    return S.get("visible", {}).get(cam, {}).get(o, 0) >= VIS[cam]


def frame_cams(f):
    return f.get("cams", ALLC)


def current_frame(sample):
    return next((f for f in sample["pool"] if f["idx"] == sample["current"]), None)


def observable(c, smp, states):
    S0 = states[0]
    Sc = states[smp["current"]]
    cf = current_frame(smp)
    neg, name, a = common.parse(c)
    truth = common.evaluate(c, Sc, S0)[0]

    def agent_cur(o):
        return cf is not None and "agent" in frame_cams(cf) and vis(Sc, o, "agent")
    if name == "gripper_open":
        st = truth
    elif name == "holding" and not neg:
        arm, o = a[0], a[1]
        st = "unknown"
        if Sc["arms"][arm]["gripper_cmd"] > 0.8:
            st = False
        else:
            pf = [f for f in smp["pool"] if "agent" in frame_cams(f)]
            for i in range(len(pf)):
                for j in range(i + 1, len(pf)):
                    A, B = states[pf[i]["idx"]], states[pf[j]["idx"]]
                    rise = B["arms"][arm]["tcp"][2] - A["arms"][arm]["tcp"][2]
                    if rise >= 3 and A["arms"][arm]["gripper_cmd"] < 0.5 and B["arms"][arm]["gripper_cmd"] < 0.5 \
                            and vis(A, o, "agent") and vis(B, o, "agent") and pf[j]["idx"] == smp["current"]:
                        orise = B["objects"][o]["lo"][2] - A["objects"][o]["lo"][2]
                        st = orise >= rise - 2.5
            if st == "unknown" and agent_cur(o):
                ob = Sc["objects"][o]
                under = any(n != o and abs(ob["lo"][2] - x["hi"][2]) < 1.5 and x["lo"][0] - 1 <= ob["center"][0] <= x["hi"][0] + 1
                            and x["lo"][1] - 1 <= ob["center"][1] <= x["hi"][1] + 1 for n, x in Sc["objects"].items())
                other = "right" if arm == "left" else "left"
                if ob["lo"][2] >= 3 and not under and Sc["held"].get(other) != o and Sc["arms"][other]["gripper_cmd"] > 0.8:
                    st = Sc["held"].get(arm) == o
    elif name == "holding" and neg:
        arm, o = a[0], a[1]
        st = "unknown"
        if cf is not None and "agent" in frame_cams(cf):
            d = sum((Sc["arms"][arm]["tcp"][k] - Sc["objects"][o]["center"][k]) ** 2 for k in range(3)) ** 0.5
            if Sc["arms"][arm]["gripper_cmd"] > 0.8 and d > 6:
                st = True
            elif Sc["arms"][arm]["gripper_cmd"] < 0.5 and Sc["held"].get(arm) == o and vis(Sc, o, "agent"):
                st = False
    elif name == "lifted":
        st = truth if agent_cur(a[0]) else "unknown"
    elif name == "at":
        st = truth if (cf is not None and "top" in frame_cams(cf) and vis(Sc, a[0], "top")) else "unknown"
    elif name in ("on", "on_table"):
        o = a[0]
        st = "unknown"
        if agent_cur(o):
            ob = Sc["objects"][o]
            base = 0.0 if (name == "on_table" or a[1] == "table") else Sc["objects"][a[1]]["hi"][2]
            gap = ob["lo"][2] - base
            if gap > 2.0:
                st = False
            elif Sc["arms"][smp["arm_main"]]["gripper_cmd"] > 0.8:
                st = truth
    else:
        st = truth
    if st == "unknown":
        return "unknown", truth
    return ("satisfied" if st else "unsatisfied"), truth


def label(smp, states):
    res = []
    for c in smp["criteria"]:
        o, t = observable(c, smp, states)
        res.append({"criterion": c, "observable": o, "physical": "satisfied" if t else "unsatisfied"})
    obs = [r["observable"] for r in res]
    dec = "unsatisfied" if "unsatisfied" in obs else ("unknown" if "unknown" in obs else "satisfied")
    phys = "satisfied" if all(r["physical"] == "satisfied" for r in res) else "unsatisfied"
    return res, dec, phys


def F(idx, stage, cams=None):
    d = {"idx": idx, "stage": stage}
    if cams:
        d["cams"] = cams
    return d


def samples_of(ep):
    S = ep["states"]
    ev = ep["events"]
    out = []

    def add(kind, goal, arm_main, arm_view, crit, current, pool, checks, intended):
        out.append({"kind": kind, "goal": goal, "arm_main": arm_main, "arm_view": arm_view, "criteria": crit,
                    "current": current, "pool": pool, "checks": checks, "intended": intended})
    for i, e in enumerate(ev):
        nxt = ev[i + 1] if i + 1 < len(ev) else None
        if e["api"] == "grasp_actor" and e.get("reached") is not None:
            arm, o = e["arm"], e["object"]
            closed = S[e["after"]]
            held = closed["held"].get(arm) == o
            lift = nxt if (nxt and nxt["api"] == "move_by_displacement" and nxt["arm"] == arm and (nxt.get("delta") or [0, 0, 0])[2] >= 3) else None
            pick = [f"holding({arm}, {o})", f"lifted({o}, 5cm)"]
            goal = f"pick up the {o} with the {arm} arm"
            if held and lift and not lift.get("missed_lift"):
                add("P1_pick_done", goal, arm, arm, pick, lift["after"],
                    [F(e["before"], "before grasp"), F(e["after"], "gripper closed"), F(lift["after"], "after lift")], {}, "positive")
                add("N5_closed_not_lifted", goal, arm, arm, pick, e["after"],
                    [F(e["before"], "before grasp"), F(e["reached"], "at grasp pose"), F(e["after"], "gripper closed")],
                    {"lift_small": [lift["after"]]}, "negative")
                add("U1_closed_no_follow", f"grasp the {o} with the {arm} arm", arm, arm, [f"holding({arm}, {o})"], e["after"],
                    [F(e["before"], "before grasp"), F(e["reached"], "at grasp pose"), F(e["after"], "gripper closed")],
                    {"lift_small": [lift["after"]]}, "unknown")
            elif held and nxt and nxt["api"] == "open_gripper" and nxt["arm"] != arm:
                # hand-over: arm receives the object from the other arm
                other = nxt["arm"]
                j = i + 2
                home = None
                while j < len(ev) and ev[j]["arm"] == other and ev[j]["api"] in ("move_by_displacement", "back_to_origin"):
                    home = ev[j]
                    j += 1
                ho = [f"holding({arm}, {o})", f"not holding({other}, {o})"]
                g2 = f"hand the {o} over from the {other} arm to the {arm} arm"
                if home:
                    add("P3_handover_done", g2, arm, "both", ho, home["after"],
                        [F(e["before"], "before receiving"), F(e["after"], "receiver closed"), F(nxt["after"], "giver opened"),
                         F(home["after"], "giver backed off")], {}, "positive")
                    add("U4_handover_no_post", g2, arm, "both", ho, home["after"],
                        [F(e["before"], "before receiving"), F(e["reached"], "receiver at grasp pose"), F(e["after"], "receiver closed")],
                        {"observe_now": [nxt["after"], home["after"]]}, "unknown")
                add("N4a_receiver_not_closed", g2, arm, "both", ho, e["reached"],
                    [F(e["before"], "before receiving"), F(e["mid"], "receiver approaching"), F(e["reached"], "receiver at grasp pose")],
                    {}, "negative")
                add("N4b_giver_not_released", g2, arm, "both", ho, e["after"],
                    [F(e["before"], "before receiving"), F(e["reached"], "receiver at grasp pose"), F(e["after"], "receiver closed")],
                    {}, "negative")
            elif not held and lift and lift.get("missed_lift"):
                touching = any(x == [arm, o] for x in closed["contacts"])
                add("N3_one_finger_rise" if touching else "N1_missed_rise", goal, arm, arm, pick, lift["after"],
                    [F(e["before"], "before grasp"), F(e["after"], "gripper closed"), F(lift["after"], "after lift")], {}, "negative")
                add("U1m_closed_no_follow_missed", f"grasp the {o} with the {arm} arm", arm, arm, [f"holding({arm}, {o})"], e["after"],
                    [F(e["before"], "before grasp"), F(e["reached"], "at grasp pose"), F(e["after"], "gripper closed")],
                    {"lift_small": [lift["after"]]}, "unknown")
        if e["api"] == "place_actor" and e.get("release") and e.get("reached") is not None:
            arm, o = e["arm"], e["object"]
            tp = e["target_point"]
            B = S[e["before"]]
            tgt = None
            for n, x in B["objects"].items():
                if n != o and x["lo"][0] - 2 <= tp[0] <= x["hi"][0] + 2 and x["lo"][1] - 2 <= tp[1] <= x["hi"][1] + 2 and abs(tp[2] - x["hi"][2]) < 4:
                    tgt = n
            if tgt:
                crit = [f"on({o}, {tgt})", f"gripper_open({arm})"]
                goal = f"place the {o} on the {tgt} with the {arm} arm and release it"
            else:
                crit = [f"on({o}, table)", f"at({o}, {round(tp[0])}, {round(tp[1])})", f"gripper_open({arm})"]
                goal = f"put the {o} down on the table at ({round(tp[0])}, {round(tp[1])}) with the {arm} arm and release it"
            ret = nxt if (nxt and nxt["api"] == "move_by_displacement" and nxt["arm"] == arm) else None
            cur = (ret or e)["after"]
            post = [F(e["reached"], "at place pose, before release"), F(e["after"], "after release")] + ([F(ret["after"], "after retreat")] if ret else [])
            add("P2_place_done", goal, arm, arm, crit, cur, post, {}, "positive")
            add("N2_hover_above_target", goal, arm, arm, crit, e["mid"],
                [F(e["before"], "carrying"), F(e["mid"], "above the target")], {}, "negative")
            add("U2_no_post_release_frame", goal, arm, arm, crit, cur,
                [F(e["before"], "carrying"), F(e["mid"], "above the target"), F(e["reached"], "at place pose, before release")],
                {"observe_now": [e["after"]] + ([ret["after"]] if ret else [])}, "unknown")
            add("U3_top_view_only", goal, arm, arm, crit, cur, [dict(f, cams=["top"]) for f in post],
                {"show_view": ["agent", arm]}, "unknown")
    return out


def post_check_labels(smp, states):
    """observable decision after each available active check (the check adds frames; lift_small is undone, so the
    judged state stays the same except that the follow evidence is now visible)."""
    out = {}
    for name, idxs in (smp.get("checks") or {}).items():
        s2 = json.loads(json.dumps(smp))
        if name == "lift_small":
            # follow evidence: closed frame -> lifted frame; judge holding at the lifted frame, other criteria at the original state
            res = []
            for c in smp["criteria"]:
                cn = common.parse(c)[1]
                if cn == "holding":
                    s3 = dict(s2, current=idxs[0], pool=s2["pool"] + [F(idxs[0], "check: lift_small")])
                    res.append(observable(c, s3, states)[0])
                else:
                    res.append(observable(c, s2, states)[0])
            out[name] = "unsatisfied" if "unsatisfied" in res else ("unknown" if "unknown" in res else "satisfied")
            continue
        if name == "observe_now":
            s2["pool"] = s2["pool"] + [F(i, "check: observe_now") for i in idxs]
            s2["current"] = idxs[-1] if smp["current"] not in idxs else smp["current"]
        elif name == "show_view":
            s2["pool"] = [{k: v for k, v in f.items() if k != "cams"} for f in s2["pool"]]
        _, dec, _ = label(s2, states)
        out[name] = dec
    return out


def main():
    rows = []
    for p in sorted((OUT / "episodes").glob("*.json")):
        ep = json.loads(p.read_text())
        S = ep["states"]
        for k, smp in enumerate(samples_of(ep)):
            res, dec, phys = label(smp, S)
            smp["post_check_label"] = post_check_labels(smp, S)
            cat = {"satisfied": "positive", "unsatisfied": "negative", "unknown": "insufficient"}[dec]
            rows.append({"id": f"{ep['episode_id']}__{k:02d}_{smp['kind']}", "episode_id": ep["episode_id"], "task": ep["task"],
                         "seed": ep["seed"], "split": ep["split"], "perturbed": ep["perturbed"], **smp, "category": cat,
                         "observable_status": dec, "physical_truth": phys, "criteria_labels": res,
                         "intended_matches": cat == {"positive": "positive", "negative": "negative", "unknown": "insufficient"}[smp["intended"]]})
    bad = [r for r in rows if r["observable_status"] == "unsatisfied" and r["physical_truth"] == "satisfied"]
    print("observable-unsatisfied but physically satisfied (should be 0):", len(bad))
    print("intended vs label mismatch:", Counter(r["kind"] for r in rows if not r["intended_matches"]))
    # balanced selection: per split, ~1/3 per category, spread over tasks and kinds
    random.seed(0)
    target = {"test": 200, "val": 30}
    chosen = []
    for split, n in target.items():
        for cat in ("positive", "negative", "insufficient"):
            pool = [r for r in rows if r["split"] == split and r["category"] == cat and r["kind"] not in EXCLUDE_KINDS]
            by = defaultdict(list)
            for r in pool:
                by[(r["task"], r["kind"])].append(r)
            for v in by.values():
                random.shuffle(v)
            sel, keys = [], sorted(by)
            while len(sel) < n and any(by[k] for k in keys):
                for k in keys:
                    if by[k] and len(sel) < n:
                        sel.append(by[k].pop())
            chosen += sel
    with open(OUT / "samples_all.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in target:
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in chosen:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cnt = Counter((r["split"], r["category"]) for r in chosen)
    print("selected:", dict(cnt))
    print("by kind:", dict(Counter((r["split"], r["kind"]) for r in chosen)))
    print("all candidates:", dict(Counter((r["split"], r["category"]) for r in rows)))


if __name__ == "__main__":
    main()
