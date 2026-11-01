"""Filesystem locations and defaults used by the harness. Everything can be overridden with environment variables.

  TINYASTRAL_THIRD_PARTY  directory holding the upstream benchmark checkouts (default: <repo>/third_party)
  ROBODAWN_REPO           RoboDawn checkout (contains RoboTwin)          default: $TINYASTRAL_THIRD_PARTY/RoboDawn
  GUAVA_REPO              Guava checkout                                 default: $TINYASTRAL_THIRD_PARTY/guava
  MOTORMIND_REPO          MotorMind checkout (LIBERO-PRO protocol)       default: $TINYASTRAL_THIRD_PARTY/motormind
  TINYASTRAL_OUT          episode outputs                                default: <repo>/runs
  TINYASTRAL_URL          OpenAI-compatible endpoint of the planner     default: http://127.0.0.1:8000/v1
"""
import os
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
THIRD_PARTY = Path(os.environ.get("TINYASTRAL_THIRD_PARTY", str(ROOT / "third_party")))


def _repo(env, name, *alt):
    for e in (env,) + alt:
        if os.environ.get(e):
            return Path(os.environ[e])
    return THIRD_PARTY / name


ROBODAWN = _repo("ROBODAWN_REPO", "RoboDawn", "ROBODAWN_ROOT")
GUAVA = _repo("GUAVA_REPO", "guava")
MOTORMIND = _repo("MOTORMIND_REPO", "motormind")
OUT = Path(os.environ.get("TINYASTRAL_OUT", str(ROOT / "runs")))
SEEDCACHE = HARNESS / "data" / "robotwin_seedcache"      # valid RoboTwin seeds per task (episode k -> k-th valid seed)
MEMORY = HARNESS / "memory"                              # task notes: memory/<set>/<task>.json
DEFAULT_URL = os.environ.get("TINYASTRAL_URL", "http://127.0.0.1:8000/v1")


# ------------------------------------------------------------------ import layout
# The harness modules import each other by bare module name (the RoboTwin scene code under bimanual/scene/ does the same),
# so the directories below go on sys.path instead of the harness being a package. A package called "harness" would also
# clash with RoboDawn's own top-level package `harness`. bimanual/ and single_arm/ both have a planner.py: one process
# uses one of the two.
PARTS = ("bimanual", "single_arm")


def use(part, front=False):
    """put harness/<part> (bimanual | single_arm), harness/adapters and harness/ on sys.path (first, in that order).
    front=True inserts them even if they are already on sys.path (so they come before anything inserted since)."""
    if part not in PARTS:
        raise ValueError(f"unknown harness part {part!r}; expected one of {PARTS}")
    other = [str(HARNESS / p) for p in PARTS if p != part]
    if any(x in sys.path for x in other):
        raise RuntimeError(f"harness/{part} requested, but another harness part is already on sys.path ({other})")
    for d in (HARNESS, HARNESS / "adapters", HARNESS / part):
        if front or str(d) not in sys.path:
            sys.path.insert(0, str(d))
