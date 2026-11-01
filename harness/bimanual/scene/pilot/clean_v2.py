#!/usr/bin/env python
"""Supervision-chain check (rules only, no LLM): instruction <-> scene <-> subgoals <-> criteria <-> expert actions.

Per episode:
  description_errors : colour words in the instruction that no task object has (measured colour from the first agent
                       frame + the block colours from the simulator). Material words (wooden, plastic, ...) are not checked
                       against colour: "wooden" can be a light-grey wooden coaster.
  goal_conflicts     : the instruction asks for a different END STATE than the expert reaches (e.g. never mentions the
                       pad while the expert puts the block on the pad), or names an arm the expert does not use for that step.
  method_unspecified : end state agrees but the instruction does not mention a step the expert takes (e.g. the hand-over,
                       or centring the first block) - harmless, kept as is.
  chain_errors       : subgoal criteria that do not hold at the end of their subgoal / expert actions not covered.
Decision: conflicts or description errors -> FIX by re-drawing the instruction from the same RoboTwin unseen template list
(RandomState(seed), only candidates that pass all checks); none left -> EXCLUDE. Writes ep["supervision"] and
ep["instruction_used"] in place and a table to <out>/supervision_check.json.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import paths as _paths  # noqa: E402
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else str(_paths.OUT / "oracle_expert_pilot_v2"))
COLORS = ["red", "green", "blue", "yellow", "gray", "grey", "white", "black", "brown", "pink", "purple", "orange", "cyan"]
SYN = {"grey": "gray", "light gray": "gray", "light blue": "blue"}
PLACE_WORDS = r"\b(pad|place|set|drop|put|rest|position)\b"
HANDOVER_WORDS = r"\b(pass|hand|handover|transfer|switch|give|right)\b"


def scene_colors(ep):
    cols = {}
    st0 = ep["states"][0]
    for n in ep["objects"]:
        m = re.match(r"(red|green|blue|yellow)_(block|pad)", n)
        if m:
            cols[n] = {m.group(1)}
    for n, d in (st0.get("colors") or {}).items():
        cols.setdefault(n, set()).add(d["name"])
    # the cup/coaster models are fixed: blue cup, light-grey (wooden) coaster; used if the measurement is missing
    cols.setdefault("cup", {"blue"})
    cols.setdefault("coaster", {"gray", "white"})
    return cols


RT = str(_paths.ROBODAWN / "RoboTwin")
FAMILY = {"blue": {"blue", "cyan"}, "cyan": {"blue", "cyan"}, "gray": {"gray", "white"}, "white": {"gray", "white"}}


def all_candidates(task, info, seed):
    """the whole unseen instruction pool of this episode (RoboDawn only draws one)."""
    import os
    import random
    cwd = os.getcwd()
    sys.path.insert(0, RT + "/description/utils")
    os.chdir(RT)
    try:
        from generate_episode_instructions import generate_episode_descriptions
        random.seed(seed)
        res = generate_episode_descriptions(task, [info], 300)
        return sorted(set(str(x) for x in res[0]["unseen"]))
    finally:
        os.chdir(cwd)


def check(task, instr, ep):
    t = instr.lower()
    desc, conflict, method = [], [], []
    cols = scene_colors(ep)
    allc = set().union(*cols.values()) if cols else set()
    for c in COLORS:
        if re.search(rf"\b{c}\b", t):
            cc = SYN.get(c, c)
            if not (FAMILY.get(cc, {cc}) & allc):
                desc.append(f"colour '{c}' matches no task object (scene: {sorted(allc)})")
    expert_arms = [c["args"].get("arm") for c in ep["calls"] if c["tool"] == "grasp"]
    if task == "place_empty_cup":
        if "cup" not in t:
            conflict.append("does not mention the cup")
        if "coaster" not in t:
            conflict.append("does not say where to put the cup (expert puts it on the coaster)")
    elif task == "stack_blocks_three":
        for c in ("red", "green", "blue"):
            if c not in t:
                conflict.append(f"does not mention the {c} block")
        if not re.search(r"cent", t):
            method.append("does not ask to centre the stack (expert builds it at the stacking spot)")
        for arm, col in zip(re.findall(r"\b(left|right)\b", t), []):
            pass
    elif task == "handover_block":
        if not re.search(PLACE_WORDS, t):
            conflict.append("never asks to put the block down (expert ends with the block on the blue pad)")
        elif "pad" not in t and not re.search(r"\b(set|put|place|drop)\b.*\b(down|it)\b", t):
            conflict.append("does not say where to put the block")
        if not re.search(HANDOVER_WORDS, t):
            method.append("does not mention the hand-over (end state still matches)")
        m = re.search(r"\b(left|right) (arm|hand)\b", t)
        if m and expert_arms and m.group(1) != expert_arms[0] and not re.search(r"\bto the (right|left)", t[:m.end()]):
            conflict.append(f"names the {m.group(1)} arm for grabbing; expert grabs with the {expert_arms[0]} arm")
    return desc, conflict, method


def chain(ep):
    errs = []
    for k, ok in enumerate(ep.get("canonical_criteria_hold_at_end") or []):
        if not ok:
            errs.append(f"subgoal {k+1}: a criterion is false at its end")
    covered = {i for s in ep["subgoals"] for i in s["call_idx"]}
    if len(covered) != len(ep["calls"]):
        errs.append("some tool calls are not in any subgoal")
    if any(c["tool"] == "unknown" for c in ep["calls"]):
        errs.append("untranslated expert call")
    return errs


def main():
    rows = []
    for p in sorted((OUT / "episodes").glob("*.json")):
        ep = json.loads(p.read_text())
        if not ep.get("keep", ep.get("expert_success")):
            continue
        task, instr = ep["task"], ep.get("instruction_original", ep["instruction"])
        desc, conflict, method = check(task, instr, ep)
        ch = chain(ep)
        decision, used = "keep", instr
        if desc or conflict:
            cands = []
            pool = ep.get("instruction_candidates") or []
            if len(pool) < 5:
                pool = all_candidates(task, ep.get("expert_info") or {}, ep["seed"])
                ep["instruction_candidates"] = pool
            for c in pool:
                d2, c2, _ = check(task, c, ep)
                if not d2 and not c2:
                    cands.append(c)
            if task == "handover_block" and any("pad" in c.lower() for c in cands):
                cands = [c for c in cands if "pad" in c.lower()]      # prefer the fully specified end state
            if cands:
                used = str(np.random.RandomState(ep["seed"]).choice(sorted(set(cands))))
                decision = "fixed"
            else:
                decision = "exclude"
        if ch:
            decision = "exclude"
        ep["instruction_original"] = instr
        ep["instruction_used"] = used
        ep["supervision"] = {"description_errors": desc, "goal_conflicts": conflict, "method_unspecified": method,
                             "chain_errors": ch, "decision": decision}
        p.write_text(json.dumps(ep, ensure_ascii=False, default=str))
        rows.append({"episode_id": ep["episode_id"], "task": task, "instruction": instr, "used": used, **ep["supervision"]})
    (OUT / "supervision_check.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    cnt = Counter((r["task"], r["decision"]) for r in rows)
    print(dict(cnt))
    for r in rows:
        if r["decision"] != "keep" or r["description_errors"]:
            print(r["episode_id"], r["decision"], "|", r["instruction"], "->", r["used"], "|", r["description_errors"], r["goal_conflicts"], r["chain_errors"])
    print("method_unspecified:", Counter(r["task"] for r in rows if r["method_unspecified"]))


if __name__ == "__main__":
    main()
