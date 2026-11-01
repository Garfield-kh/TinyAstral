#!/usr/bin/env python
"""LIBERO-PRO evaluation with H* (300 episodes: base tasks 0-9/50-59/100-109 and their 120 perturbed variants, seeds 0 and 1).

  python eval/run_libero.py --label tinyastral4b [--url URL] [--model qwen] [--workers 4]

Needs: the LIBERO (MotorMind) conda env, MOTORMIND_REPO and ROBODAWN_REPO, MuJoCo EGL (MUJOCO_GL=egl,
MUJOCO_EGL_DEVICE_ID), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (LIBERO init states), a SAM3 server (LIBERO_SAM3_URL, API of
perception/sam3_srv.py) and the planner endpoint. --gen default and --xml-calls 1 (the reported setting).
Output: $TINYASTRAL_OUT/libero_<label>/t<task>_s<seed>/. Score with: python eval/score.py libero $TINYASTRAL_OUT/libero_<label>"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pool  # noqa: E402
import episodes as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--url", default=_pool.default_url())
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--gen", default="default")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--part", choices=["all", "base", "pert"], default="all")
    a = ap.parse_args()
    out = _pool.out_root() / f"libero_{a.label}"
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONUNBUFFERED="1")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.pop("LIBERO_CONFIG_PATH", None)
    jobs = []
    for t, s, part, dn in E.libero_episodes():
        if a.part != "all" and part != a.part:
            continue
        argv = [_pool.py(), "-u", str(_pool.MAIN), "libero", "--gen", a.gen, "--task", str(t), "--seed", str(s), "--out", str(out),
                "--url", a.url, "--model", a.model, "--thinking", "1", "--xml-calls", "1"]
        jobs.append((dn, argv, env, str(out.parent), out / dn / "result.json"))
    out.mkdir(parents=True, exist_ok=True)
    sys.exit(1 if _pool.run_jobs(jobs, a.workers, out / "_logs") else 0)


if __name__ == "__main__":
    main()
