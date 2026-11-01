#!/usr/bin/env python
"""Score an EB-Manipulation run (<run>/<eval_set>/epNNN/result.json) with the paper's official measure.

Official measure: success_any = the official per-step task_success latched over the episode (env-step budget 60),
pooled over all finished episodes (228 for the full set). Also printed: success_15 (success within the official
15-step horizon) and per-subset numbers.

  python score_eb.py --run <out-root>/<tag> [--json]"""
from __future__ import annotations

import argparse
import glob
import json
import os

SETS = {"base": 48, "common_sense": 48, "complex": 48, "spatial": 48, "visual": 36}


def _load(f):
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def score(run):
    R = [x for x in (_load(f) for f in glob.glob(os.path.join(run, "*", "ep*", "result.json"))) if x]
    out = {"n": len(R), "success": pct(sum(bool(x.get("success_any")) for x in R), len(R)),
           "success_15": pct(sum(bool(x.get("success_15")) for x in R), len(R))}
    for es, full in SETS.items():
        L = [x for x in R if x.get("eval_set") == es]
        out[es] = {"n": len(L), "full": full, "success": pct(sum(bool(x.get("success_any")) for x in L), len(L)),
                   "success_15": pct(sum(bool(x.get("success_15")) for x in L), len(L))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = score(a.run)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    f = lambda x: "-" if x is None else f"{x:.1f}"  # noqa: E731
    for es in SETS:
        print(f"{es:14s} {r[es]['n']:>3d}/{r[es]['full']:<3d} success {f(r[es]['success']):>5s}   success_15 {f(r[es]['success_15']):>5s}")
    print(f"\nEB-Manipulation official measure (success_any, {r['n']}/228 episodes): {f(r['success'])}   success_15 {f(r['success_15'])}")


if __name__ == "__main__":
    main()
