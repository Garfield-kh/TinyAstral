#!/usr/bin/env python
"""RoboTwin-50 evaluation with H* (500 episodes: 50 tasks x episodes 0-9).

  python eval/run_robotwin.py --lane tinyastral4b [--url URL] [--model qwen] [--workers 4] [--tasks a,b] [--episodes 0-9]

Needs: the RoboTwin conda env (SAPIEN, Vulkan), ROBODAWN_REPO, a SAM3 server (PERC_SAM3, API of perception/sam3_srv.py),
and the planner endpoint. Output: $TINYASTRAL_OUT/<lane>/<task>/shard_<k>/{results.json, episode_<k>/episode.json, ...}.
Score with: python eval/score.py robotwin $TINYASTRAL_OUT/<lane>"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pool  # noqa: E402
import episodes as E  # noqa: E402


def rng(s):
    out = []
    for p in s.split(","):
        a, _, b = p.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, help="run name (output sub-directory)")
    ap.add_argument("--url", default=_pool.default_url())
    ap.add_argument("--model", default="qwen", help="served model name")
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tasks", default="", help="comma-separated subset (default: all 50)")
    ap.add_argument("--episodes", default="0-9")
    a = ap.parse_args()
    tasks = a.tasks.split(",") if a.tasks else E.robotwin_tasks()
    out = _pool.out_root()
    env = dict(os.environ, TINYASTRAL_OUT=str(out))
    for k in ("HARNESS_MEMORY", "HARNESS_NO_OBJECT_LIST"):      # evaluation protocol: task note on, object list given
        env.pop(k, None)
    jobs = []
    for t in tasks:
        for k in rng(a.episodes):
            argv = [_pool.py(), str(_pool.MAIN), "robotwin", "--task", t, "--episode", str(k), "--lane", a.lane,
                    "--url", a.url, "--model", a.model, "--thinking", str(a.thinking)]
            jobs.append((f"{t}_{k}", argv, env, str(_pool.ROOT), out / a.lane / t / f"shard_{k}" / "results.json"))
    sys.exit(1 if _pool.run_jobs(jobs, a.workers, out / a.lane / "_logs") else 0)


if __name__ == "__main__":
    main()
