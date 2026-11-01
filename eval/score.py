#!/usr/bin/env python
"""Strict success S for the training benchmarks, as reported in the paper.

  python eval/score.py robotwin|guava|libero RUN_DIR [--csv out.csv]

Per episode of the fixed list (eval/episodes.py):
  S  the planner ended the episode itself with done (end_reason model_done or done_after_rejections) AND the simulator's
     success predicate holds at the end;
  U  outcome unknown: missing result, endpoint / infrastructure error, RoboTwin wall_timeout, LIBERO wall-clock budget expiry;
  F  otherwise.
Reported number = S / N with the fixed N (500 / 450 / 300); the upper bound (S + U) / N is printed as well."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import episodes as E  # noqa: E402

DONE = ("model_done", "done_after_rejections")
CONN = ("Connection refused", "Max retries", "URLError", "Connection reset", "RemoteDisconnected", "HTTP Error 5",
        "Name or service not known", "No route to host", "BadStatusLine")


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def judge_robotwin(root, task, k):
    r = _load(Path(root) / task / f"shard_{k}" / "results.json")
    if r is None or not r.get("episodes"):
        return "U", "missing"
    ep = r["episodes"][0]
    end = ep.get("finished_reason")
    if end == "wall_timeout":
        return "U", "wall_timeout"
    if end == "invalid_model_output" and any(c in str(ep.get("error") or "") for c in CONN):
        return "U", "endpoint"
    return ("S" if (end in DONE and ep.get("final_sim_success")) else "F"), end


def judge_unified(d, bench):
    d = Path(d)
    r = _load(d / "result.json")
    if r is None:
        e = _load(d / "error.json") or {}
        return "U", "missing" if not e else ("endpoint" if any(c in e.get("error", "") for c in CONN) else "error")
    if r.get("infra_error"):
        return "U", "endpoint"
    end = r.get("end_reason")
    if bench == "libero" and end == "budget":
        return "U", "budget"
    if end == "invalid_model_output":
        ep = _load(d / "episode.json") or {}
        txt = " ".join(str(s.get("error") or "") for s in (ep.get("steps") or [])[-1:])
        if any(c in txt for c in CONN) or (bench == "guava" and "timed out" in txt):
            return "U", "endpoint"
    return ("S" if (end in DONE and r.get("success")) else "F"), end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["robotwin", "guava", "libero"])
    ap.add_argument("root")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    rows = []
    if a.bench == "robotwin":
        for t, k, seed in E.robotwin_episodes():
            st, why = judge_robotwin(a.root, t, k)
            rows.append({"group": t, "episode": f"{t}:{k}", "seed": seed, "status": st, "why": why})
    elif a.bench == "guava":
        for t, k, seed, dn in E.guava_episodes():
            st, why = judge_unified(Path(a.root) / dn, "guava")
            rows.append({"group": "unseen" if t in E.GUAVA_UNSEEN else "seen", "episode": dn, "seed": seed, "status": st, "why": why})
    else:
        for t, s, part, dn in E.libero_episodes():
            st, why = judge_unified(Path(a.root) / dn, "libero")
            rows.append({"group": part, "episode": dn, "seed": s, "status": st, "why": why})
    n = len(rows)
    c = defaultdict(int)
    g = defaultdict(lambda: defaultdict(int))
    for r in rows:
        c[r["status"]] += 1
        g[r["group"]][r["status"]] += 1
    print(f"{a.bench}: S={c['S']} F={c['F']} U={c['U']} N={n}  ->  S/N = {100 * c['S'] / n:.1f}%  (upper bound (S+U)/N = {100 * (c['S'] + c['U']) / n:.1f}%)")
    if a.bench != "robotwin":
        for k, v in sorted(g.items()):
            m = sum(v.values())
            print(f"  {k:8s} S={v['S']}/{m} = {100 * v['S'] / m:.1f}%  U={v['U']}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
