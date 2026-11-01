"""Shared helpers for the zero-shot drivers (simplerenv/run_simplerenv.py, vlabench/run_vlabench.py,
embodiedbench/run_eb.py).

Every driver runs the TinyAstral harness H* through the public API of harness/single_arm/planner.py (HM below):
  harness : HM.make_harness(bench, A, prof, flags)
  loop    : HM.run_loop(D, H, url, model, thinking, max_tokens)
  SYSTEM  : HM.system_text(bench, base, flags)
bench = "guava" (SimplerEnv, EB-Manipulation: the single-arm Guava prompt / tool profile) or "libero" (VLABench: the
LIBERO prompt / tool profile). --gen default|none|all|f1,f2 ; omitted -> $HARNESS_GEN, else HM.GEN_DEFAULT
("default" = HM.GEN_DEFAULTS[bench]).
Per episode a side file harness_info.json records bench / flags / SHA of the SYSTEM actually sent (taken from the
loop's own first message); result.json and episode.json are the per-episode result files read by the scorers.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "harness"
DEFAULT_URL = "http://127.0.0.1:8000/v1"


def planner_url():
    """OpenAI-compatible base URL of the planner model ($TINYASTRAL_URL)."""
    return os.environ.get("TINYASTRAL_URL", DEFAULT_URL)


def out_root():
    """results root ($TINYASTRAL_OUT, default <repo>/runs)."""
    return Path(os.environ.get("TINYASTRAL_OUT", str(REPO / "runs")))


def third_party(env_var, name):
    """upstream benchmark clone: $<env_var>, else $TINYASTRAL_THIRD_PARTY/<name>, else <repo>/third_party/<name>."""
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    return Path(os.environ.get("TINYASTRAL_THIRD_PARTY", str(REPO / "third_party"))) / name


def load_harness(bench, front=False):
    """import the single-arm harness (harness/single_arm/planner.py) for the Guava / LIBERO profile ($HARNESS_BENCH set
    BEFORE the import; the RoboTwin harness and SAPIEN 3 are never loaded). front=True puts harness/ (and its adapters/,
    single_arm/) first on sys.path even if they are already on it (EB-Manipulation: its 'tools' package clashes with
    harness/tools.py)."""
    if bench not in ("guava", "libero"):
        raise ValueError(bench)
    cur = os.environ.get("HARNESS_BENCH")
    if cur not in (None, bench):
        raise RuntimeError(f"HARNESS_BENCH={cur!r} but this driver needs {bench!r}")
    os.environ["HARNESS_BENCH"] = bench
    if str(HARNESS) not in sys.path or front:
        sys.path.insert(0, str(HARNESS))
    import paths
    paths.use("single_arm", front=front)
    import planner as HM
    if HM.BENCH != bench or not hasattr(HM, "UnifiedHarness"):
        raise RuntimeError(f"single-arm planner loaded as {HM.BENCH!r} (needed {bench!r}); was it imported earlier in this process?")
    return HM


def add_gen_arg(ap):
    ap.add_argument("--gen", default=None,
                    help="generic harness improvements: default|none|all|f1,f2 (omitted: $HARNESS_GEN, else HM.GEN_DEFAULT)")


def gen_flags(HM, bench, spec):
    """(flags, spec) exactly as the harness's own CLI (harness/run.py) resolves them."""
    if spec is None:
        spec = os.environ.get("HARNESS_GEN", HM.GEN_DEFAULT)
    return set(HM.gen_flags(spec, bench)), spec


def _sha(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def check_sent(S, system):
    """the SYSTEM recorded must be the one the loop sent: the loop's first message IS the request's system."""
    if not S.get("msgs"):
        return None                       # the loop raised before returning (no request record to compare)
    sent = S["msgs"][0]["content"]
    if sent != system:
        raise RuntimeError("zero-shot driver: recorded SYSTEM differs from the one sent")
    return sent


def tag(d, HM, bench, flags, spec, system, driver):
    """side file next to result.json (not model-facing)."""
    info = {"harness": "harness/single_arm", "driver": Path(driver).name, "bench_profile": bench, "gen_spec": spec, "gen": sorted(flags),
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(), "system_len": len(system),
            "planner_sha256": _sha(HARNESS / "single_arm/planner.py"), "flags_sha256": _sha(HARNESS / "single_arm/flags.py"),
            "prompts_sha256": _sha(HARNESS / "single_arm/prompts.py"), "driver_sha256": _sha(driver),
            "zs_common_sha256": _sha(__file__), "planner_info": getattr(HM, "INFO", None)}
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "harness_info.json").write_text(json.dumps(info, indent=1))
    except Exception:  # noqa: BLE001
        pass


def run_pool(jobs, workers, env_for_worker=None, log_dir=None):
    """jobs = [(name, argv)]; runs them as subprocesses with `workers` parallel slots (slot i gets env_for_worker(i)).
    A job's stdout/stderr go to <log_dir>/<name>.log when log_dir is given."""
    it = iter(list(jobs))
    lock = threading.Lock()

    def slot(i):
        env = {**os.environ, **(env_for_worker(i) if env_for_worker else {})}
        while True:
            with lock:
                j = next(it, None)
            if j is None:
                return
            name, argv = j
            if log_dir is not None:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(log_dir) / f"{name}.log", "w") as f:
                    rc = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT, env=env)
            else:
                rc = subprocess.call(argv, env=env)
            print(f"[slot {i}] {name} rc={rc}", flush=True)

    th = [threading.Thread(target=slot, args=(i,)) for i in range(max(1, workers))]
    for t in th:
        t.start()
    for t in th:
        t.join()
