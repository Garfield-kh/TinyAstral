#!/usr/bin/env python
"""Score a SimplerEnv visual-matching run (result.json per episode) with the paper's official measure.

Official measure (paper, Table "zero-shot benchmarks"): success_final = the official SimplerEnv success flag when the agent
ends the episode. Task SR is pooled over a task's variants (Pick Coke Can = horizontal + vertical + standing;
Open/Close Drawer = open + close); Google VM = arithmetic mean of Pick Coke Can, Move Near, Open/Close Drawer;
WidowX = arithmetic mean of the four Bridge tasks. Also printed: success_strict (official success at the official step
horizon), the 4-task Google average (with Open Top Drawer and Place Apple), and episode counts.

  python score_simplerenv.py --out DIR [--json]"""
from __future__ import annotations

import argparse
import glob
import json
import os

TASK = {"coke": ["coke_h", "coke_v", "coke_s"], "move_near": ["move_near"], "drawer": ["drawer_open", "drawer_close"],
        "apple": ["apple_drawer"], "spoon": ["br_spoon"], "carrot": ["br_carrot"], "stack": ["br_stack"], "eggplant": ["br_eggplant"]}
NAMES = {"coke": "Pick Coke Can", "move_near": "Move Near", "drawer": "Open/Close Drawer", "apple": "Open Top Drawer and Place Apple",
         "spoon": "Put Spoon on Towel", "carrot": "Put Carrot on Plate", "stack": "Stack Green Block on Yellow Block",
         "eggplant": "Put Eggplant in Basket"}
FULL = {"coke": 300, "move_near": 240, "drawer": 216, "apple": 108, "spoon": 24, "carrot": 24, "stack": 24, "eggplant": 24}
AVGS = (("google_vm", ["coke", "move_near", "drawer"]), ("google_vm4", ["coke", "move_near", "drawer", "apple"]),
        ("widowx", ["spoon", "carrot", "stack", "eggplant"]))


def _load(f):
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def score(out):
    R = [x for x in (_load(f) for f in glob.glob(os.path.join(out, "*", "result.json"))) if x]
    res = {}
    for t, ks in TASK.items():
        L = [x for x in R if x["key"] in ks]
        res[t] = {"n": len(L), "full": FULL[t], "final": pct(sum(bool(x["success_final"]) for x in L), len(L)),
                  "strict": pct(sum(bool(x["success_strict"]) for x in L), len(L))}
    for name, ts in AVGS:
        res[name] = {"n": sum(res[t]["n"] for t in ts)}
        for k in ("final", "strict"):
            v = [res[t][k] for t in ts]
            res[name][k] = round(sum(v) / len(v), 1) if all(x is not None for x in v) else None
    res["n"] = len(R)
    res["errors_without_result"] = sum(1 for f in glob.glob(os.path.join(out, "*", "error.json"))
                                       if not os.path.exists(os.path.join(os.path.dirname(f), "result.json")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = score(a.out)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    f = lambda x: "-" if x is None else f"{x:.1f}"  # noqa: E731
    print(f"SimplerEnv visual matching: {r['n']}/960 episodes with result.json ({r['errors_without_result']} crashed, not counted)")
    print(f"{'task':36s} {'n':>9s} {'final':>6s} {'strict':>6s}")
    for t in TASK:
        print(f"{NAMES[t]:36s} {r[t]['n']:>4d}/{r[t]['full']:<4d} {f(r[t]['final']):>6s} {f(r[t]['strict']):>6s}")
    print()
    print(f"Google VM (3 tasks, official measure = final): {f(r['google_vm']['final'])}   strict {f(r['google_vm']['strict'])}")
    print(f"Google VM (4 tasks incl. apple):               {f(r['google_vm4']['final'])}   strict {f(r['google_vm4']['strict'])}")
    print(f"WidowX (4 tasks, official measure = final):    {f(r['widowx']['final'])}   strict {f(r['widowx']['strict'])}")


if __name__ == "__main__":
    main()
