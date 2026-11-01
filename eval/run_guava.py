#!/usr/bin/env python
"""Guava evaluation with H* (450 episodes: 15 tasks x 30; seed of episode k of task i = 42 + 30 i + k).

  python eval/run_guava.py --label tinyastral4b [--url URL] [--model qwen] [--sam3 http://127.0.0.1:8314] [--workers 4]

Needs: the Guava conda env, GUAVA_REPO (the harness imports guava.*; episodes run with cwd = GUAVA_REPO), MuJoCo EGL
(MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID), Guava's own SAM3 server (python -m guava.scripts.sam3_server, API /segment) and the
planner endpoint. --gen default (the reported setting). Output: $TINYASTRAL_OUT/guava_<label>/<task>__k<k>_s<seed>/.
Score with: python eval/score.py guava $TINYASTRAL_OUT/guava_<label>"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import _pool  # noqa: E402
import episodes as E  # noqa: E402
import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default=_pool.default_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--sam3", default=os.environ.get("GUAVA_SAM3_URL", "http://127.0.0.1:8314"))
    ap.add_argument("--gen", default="default")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    out = _pool.out_root() / f"guava_{a.label}"
    guava = str(paths.GUAVA)
    env = dict(os.environ, GUAVA_SAM3_URL=a.sam3, PYTHONPATH=os.pathsep.join(x for x in (guava, os.environ.get("PYTHONPATH")) if x))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    jobs = []
    for t in (a.tasks.split(",") if a.tasks else E.GUAVA_TASKS):
        sb = E.guava_seed_base(t)
        for off in range(0, E.GUAVA_EPISODES, E.GUAVA_CHUNK):
            last = f"{t}__k{off + E.GUAVA_CHUNK - 1:03d}_s{sb + off + E.GUAVA_CHUNK - 1}"
            argv = [_pool.py(), "-u", str(_pool.MAIN), "guava", "--gen", a.gen, "--sam3", a.sam3, "--task", t,
                    "--seed-base", str(sb), "--episodes", str(E.GUAVA_CHUNK), "--offset", str(off), "--out", str(out),
                    "--model", a.model, "--url", a.url, "--thinking", "1"]
            jobs.append((f"{t}_o{off}", argv, env, guava, out / last / "result.json"))
    sys.exit(1 if _pool.run_jobs(jobs, a.workers, out / "_logs") else 0)


if __name__ == "__main__":
    main()
