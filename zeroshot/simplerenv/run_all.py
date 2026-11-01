#!/usr/bin/env python
"""Run the full SimplerEnv visual-matching set (960 episodes: Google Robot 864 incl. apple-in-drawer, WidowX 96) with a
simple local worker pool: one subprocess per episode (run_simplerenv.py), finished episodes (result.json) are skipped,
so the command can be re-run to resume / retry failed episodes.

  python run_all.py --out $TINYASTRAL_OUT/zeroshot/simplerenv/mymodel --workers 4 --gpus 0,1 [--keys coke_h,move_near]
                    [-- <extra args passed to run_simplerenv.py, e.g. --gen default --thinking 1>]
Each worker renders on one GPU (CUDA_VISIBLE_DEVICES round-robin over --gpus; SAPIEN uses cuda:0 of the visible set).
Score with: python score_simplerenv.py --out <same dir>"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import zs_common as ZC  # noqa: E402
import episodes_simplerenv as ES  # noqa: E402

KEYS = ["coke_h", "coke_v", "coke_s", "move_near", "drawer_open", "drawer_close", "apple_drawer",
        "br_spoon", "br_carrot", "br_stack", "br_eggplant"]


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ZC.out_root() / "zeroshot" / "simplerenv"))
    ap.add_argument("--url", default=ZC.planner_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--sam3", default=os.environ.get("SIMPLER_SAM3_URL", "http://127.0.0.1:8160"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--keys", default="", help="comma list of episode keys (default: all 11)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes of the (shuffled) list")
    a = ap.parse_args(argv)
    keys = a.keys.split(",") if a.keys else KEYS
    eps = [e for k in keys for e in ES.manifest(k)]
    random.Random(0).shuffle(eps)                # mixes tasks so partial runs cover every task
    if a.limit:
        eps = eps[:a.limit]
    out = Path(a.out)
    jobs = []
    for e in eps:
        name = f"{e['key']}__{e['idx']:04d}"
        if (out / name / "result.json").exists():
            continue
        if (out / name / "error.json").exists():
            (out / name / "error.json").unlink()   # retry crashed episodes
        jobs.append((name, [sys.executable, str(HERE / "run_simplerenv.py"), "--key", e["key"], "--idx", str(e["idx"]),
                            "--url", a.url, "--model", a.model, "--sam3", a.sam3, "--out", str(out)] + extra))
    print(f"{len(jobs)} episodes to run ({len(eps) - len(jobs)} already done)", flush=True)
    gpus = a.gpus.split(",")
    ZC.run_pool(jobs, a.workers, lambda i: {"CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)]}, log_dir=out / "_logs")


if __name__ == "__main__":
    main()
