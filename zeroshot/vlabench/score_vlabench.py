#!/usr/bin/env python
"""Score a VLABench run (<out>/<track>/<task>/eKK/result.json) with the paper's official measure.

Official measure: the official success event (the task's termination condition, checked after every env.step; the
episode stops when it fires) = result.json["success"]. Episodes with error.json and no result.json count as failures.
Track SR = arithmetic mean of its task SRs (official per-task SR, then averaged); overall = mean of the tracks.
By default only episodes e00..e09 are scored (the paper's 500-episode set: first 10 per task over 5 tracks);
--max-ep 0 scores everything present. Also printed: success_strict (success within the official max_episode_length).

  python score_vlabench.py --out DIR [--max-ep 10] [--json]"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _load(f):
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def score(out, max_ep=10):
    res, tr_f, tr_s, n, ne = {}, [], [], 0, 0
    for trd in sorted(glob.glob(os.path.join(out, "track_*"))):
        tf, ts, tasks = [], [], {}
        for tad in sorted(glob.glob(os.path.join(trd, "*"))):
            eps = [e for e in glob.glob(os.path.join(tad, "e[0-9][0-9]")) if not max_ep or int(os.path.basename(e)[1:]) < max_ep]
            rs = [r for r in (_load(os.path.join(e, "result.json")) for e in eps if os.path.exists(os.path.join(e, "result.json"))) if r]
            er = sum(1 for e in eps if os.path.exists(os.path.join(e, "error.json")) and not os.path.exists(os.path.join(e, "result.json")))
            if not rs and not er:
                continue
            tot = len(rs) + er
            n += len(rs)
            ne += er
            f, s = sum(bool(r["success"]) for r in rs) / tot, sum(bool(r.get("success_strict")) for r in rs) / tot
            tf.append(f)
            ts.append(s)
            tasks[os.path.basename(tad)] = {"n": tot, "errors": er, "success": round(100 * f, 1), "strict": round(100 * s, 1)}
        if tf:
            k = os.path.basename(trd)
            res[k] = {"success": round(100 * sum(tf) / len(tf), 1), "strict": round(100 * sum(ts) / len(ts), 1), "tasks": tasks}
            tr_f.append(res[k]["success"])
            tr_s.append(res[k]["strict"])
    res["avg"] = {"success": round(sum(tr_f) / len(tr_f), 1) if tr_f else None,
                  "strict": round(sum(tr_s) / len(tr_s), 1) if tr_s else None, "n": n, "errors": ne, "tracks": len(tr_f)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ep", dest="max_ep", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = score(a.out, a.max_ep)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    for k, v in r.items():
        if k == "avg":
            continue
        print(f"{k:32s} success {v['success']:5.1f}   strict {v['strict']:5.1f}   ({sum(t['n'] for t in v['tasks'].values())} episodes)")
    av = r["avg"]
    print(f"\nVLABench official measure (mean over {av['tracks']} tracks): {av['success']}   strict {av['strict']}   "
          f"[{av['n']} episodes with result.json, {av['errors']} crashed = failures]")


if __name__ == "__main__":
    main()
