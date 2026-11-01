#!/usr/bin/env python
"""run.py: the one entry point of the harness H* on the three training benchmarks.

  RoboTwin (2 arms)        run.py robotwin --task T --episode k --lane L --url U [--model qwen] ...
                           -> bimanual/planner.py main() (the RoboTwin harness: SYSTEM, tools, task memory, planner loop)
  Guava / LIBERO-PRO       run.py guava  [--gen default|none|all|f1,f2] --task T --seed-base S --episodes N --out DIR ...
  (1 Panda)                run.py libero [--gen default|none|all|f1,f2] --task 57 --seed 0 1 --out DIR ...
                           -> single_arm/planner.py main() (the single-arm planner loop and tool layer, plus the
                              benchmark-agnostic tool-layer improvements of single_arm/flags.py selected by --gen)

--gen: "default" = the per-benchmark default set (the setting used for all reported results), "none" = the plain
single-arm harness, "all" = every flag, or a comma-separated list. $HARNESS_GEN overrides the default. See
single_arm/planner.py for what the flags change and for library use by the zero-shot drivers.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paths  # noqa: E402


def main_robotwin(argv):
    """RoboTwin: the RoboTwin harness's own CLI (same arguments, same output files)."""
    for k in ("HARNESS_MEMORY", "HARNESS_NO_OBJECT_LIST"):    # evaluation protocol: task memory on, object list given
        os.environ.pop(k, None)
    paths.use("bimanual")
    import planner as HR
    old = sys.argv
    sys.argv = [str(HERE / "bimanual" / "planner.py")] + list(argv)
    try:
        HR.main()
    finally:
        sys.argv = old


def main_single_arm(argv):
    """Guava / LIBERO-PRO: <bench> [--gen G] <arguments of single_arm/planner.py>."""
    os.environ["HARNESS_BENCH"] = argv[0]                  # the sub-command decides the benchmark (planner.BENCH)
    paths.use("single_arm")
    import planner as U
    U.main(list(argv))


def _entry():
    argv = sys.argv[1:]
    if argv and argv[0] in ("guava", "libero"):
        main_single_arm(argv)
    elif argv and argv[0] == "robotwin":
        main_robotwin(argv[1:])
    else:
        raise SystemExit("usage: run.py robotwin|guava|libero ...")


if __name__ == "__main__":
    _entry()
