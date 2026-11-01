#!/usr/bin/env python
"""Run the VLABench zero-shot set with a simple local worker pool (one subprocess per episode, run_vlabench.py).

Default = the paper's 500-episode set: the first 10 official evaluation episodes of every task in
track_1_in_distribution, track_2_cross_category, track_3_common_sense, track_4_semantic_instruction,
track_6_unseen_texture. --max-ep 0 runs every episode of the track files. Episodes with result.json (or error.json,
which the scorer counts as a failure) are skipped; --retry-errors re-runs crashed episodes.

  python run_all.py --out $TINYASTRAL_OUT/zeroshot/vlabench/mymodel --workers 4 --gpus 0,1 [--sam3 http://127.0.0.1:8164,...]
                    [-- <extra args passed to run_vlabench.py, e.g. --gen default>]
Each worker renders with MuJoCo EGL on one GPU (MUJOCO_EGL_DEVICE_ID round-robin over --gpus).
Score with: python score_vlabench.py --out <same dir>"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import zs_common as ZC  # noqa: E402

TRACKS = ["track_1_in_distribution", "track_2_cross_category", "track_3_common_sense", "track_4_semantic_instruction",
          "track_6_unseen_texture"]


def jobs(tracks, tasks=None, max_ep=10):
    from env_vlabench import load_track
    out = []
    for tr in tracks:
        cfg = load_track(tr)
        for ta, eps in cfg.items():
            if tasks and ta not in tasks:
                continue
            n = len(eps) if not max_ep else min(len(eps), max_ep)
            out += [(tr, ta, k) for k in range(n)]
    out.sort(key=lambda j: (j[2], TRACKS.index(j[0]) if j[0] in TRACKS else 9, j[1]))   # interleave tasks
    return out


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ZC.out_root() / "zeroshot" / "vlabench"))
    ap.add_argument("--url", default=ZC.planner_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--sam3", default=os.environ.get("LIBERO_SAM3_URL", "http://127.0.0.1:8164"),
                    help="comma list of SAM3 servers (round-robin over workers)")
    ap.add_argument("--tracks", nargs="+", default=TRACKS)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--max-ep", dest="max_ep", type=int, default=10, help="episodes per (track, task); 0 = all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", default=os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"), help="EGL render devices (round-robin)")
    ap.add_argument("--retry-errors", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    J = []
    for tr, ta, k in jobs(a.tracks, a.tasks, a.max_ep):
        d = out / tr / ta / f"e{k:02d}"
        if (d / "result.json").exists():
            continue
        if (d / "error.json").exists():
            if not a.retry_errors:
                continue
            (d / "error.json").unlink()
        J.append((f"{tr}__{ta}__e{k:02d}", (tr, ta, k)))
    print(f"{len(J)} episodes to run", flush=True)
    sams, gpus = a.sam3.split(","), a.gpus.split(",")
    jobs_ = [(name, [sys.executable, str(HERE / "run_vlabench.py"), "--track", tr, "--task", ta, "--episodes", str(k),
                     "--out", str(out), "--url", a.url, "--model", a.model] + extra) for name, (tr, ta, k) in J]
    # per worker: its EGL device and its SAM3 endpoint (run_vlabench.py reads $LIBERO_SAM3_URL when --sam3 is not given)
    ZC.run_pool(jobs_, a.workers, lambda i: {"MUJOCO_EGL_DEVICE_ID": gpus[i % len(gpus)], "LIBERO_SAM3_URL": sams[i % len(sams)]},
                log_dir=out / "_logs")


if __name__ == "__main__":
    main()
