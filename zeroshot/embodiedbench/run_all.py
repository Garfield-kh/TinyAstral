#!/usr/bin/env python
"""Run all 228 EB-Manipulation episodes (base / common_sense / complex / spatial: 48 each, visual: 36) with a simple local
worker pool: one subprocess per episode (run_eb.py), finished episodes (result.json) are skipped, so the command can be
re-run to resume / retry crashed episodes. CoppeliaSim renders on the CPU through the X display in $DISPLAY (Xvfb);
all workers may share one display.

  python run_all.py --tag mymodel --workers 8 --sam3 http://127.0.0.1:8830 [--out-root DIR] [--sets base,visual]
                    [-- <extra args passed to run_eb.py, e.g. --gen default>]
Results: <out-root>/<tag>/<eval_set>/epNNN/result.json.  Score with: python score_eb.py --run <out-root>/<tag>"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import zs_common as ZC  # noqa: E402

SETS = {"base": 48, "common_sense": 48, "complex": 48, "spatial": 48, "visual": 36}


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="harness")
    ap.add_argument("--out-root", default=str(ZC.out_root() / "zeroshot" / "embodiedbench"))
    ap.add_argument("--url", default=ZC.planner_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--sam3", default=os.environ.get("EB_SAM3_URL", "http://127.0.0.1:8830"), help="comma list (round-robin over workers)")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    root = Path(a.out_root).resolve()
    jobs = []
    for i in range(48):                                # index-major: every subset fills in evenly
        for es in a.sets.split(","):
            if i >= SETS[es]:
                continue
            d = root / a.tag / es / f"ep{i:03d}"
            if (d / "result.json").exists():
                continue
            if (d / "error.json").exists():
                (d / "error.json").unlink()
            jobs.append((f"{es}__{i:03d}", [sys.executable, "-u", str(HERE / "run_eb.py"), "--eval-set", es, "--indexes", str(i),
                                             "--url", a.url, "--model", a.model, "--tag", a.tag, "--out-root", str(root)] + extra))
    print(f"{len(jobs)} episodes to run", flush=True)
    sams = a.sam3.split(",")
    ZC.run_pool(jobs, a.workers, lambda i: {"EB_SAM3_URL": sams[i % len(sams)]}, log_dir=root / a.tag / "_logs")


if __name__ == "__main__":
    main()
