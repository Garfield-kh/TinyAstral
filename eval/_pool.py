"""Minimal job pool: each job is one subprocess (one simulator per process); results are files, so a re-run skips done jobs."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness"
MAIN = HARNESS / "run.py"


def default_url():
    return os.environ.get("TINYASTRAL_URL", "http://127.0.0.1:8000/v1")


def out_root():
    return Path(os.environ.get("TINYASTRAL_OUT", str(ROOT / "runs")))


def run_jobs(jobs, workers, log_dir):
    """jobs: [(name, argv, env, cwd, done_marker_path)]"""
    log_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not Path(j[4]).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run with {workers} worker(s)", flush=True)

    def one(j):
        name, argv, env, cwd, _ = j
        t0 = time.time()
        with open(log_dir / f"{name}.log", "w") as lf:
            rc = subprocess.call(argv, env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} {time.time() - t0:.0f}s", flush=True)
        return rc

    with ThreadPoolExecutor(max(1, workers)) as ex:
        rcs = list(ex.map(one, todo))
    return sum(1 for r in rcs if r != 0)


def py():
    return os.environ.get("TINYASTRAL_PYTHON", sys.executable)
