"""LIBERO-PRO episode protocol of the single-arm harness H* (MotorMind Table 3 protocol; used by single_arm/planner.py,
LiberoDriver; environment: env_libero.py, adapter: adapter_libero.py).

Protocol
  task = MotorMind `--suite libero_pro` aggregate index (0-9 spatial, 50-59 object, 100-109 goal; perturbations
  10-49 inside each block); seed k = LIBERO's saved initial state k (evaluation: 0, 1); reset -> set_init_state ->
  32 settle ticks; 450 s wall clock per episode (model + perception + simulation all counted); success = BDDL
  predicate. Termination is the AGENT's (done with checks / give_up / budget), not stop-on-success: the reported
  success is the predicate at the moment the episode ends; "ever" = the predicate was true after some call
  (= what a stop-on-success protocol would have scored), logged only.
The planner sees: the task sentence, the agentview camera image, gripper proprioception and the tool results.
No task names in this file. SYSTEM = single_arm/prompts.LIBERO_SYSTEM."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # harness/: paths, tools, geom
os.environ.setdefault("MUJOCO_GL", "egl")

import guava_protocol  # noqa: E402,F401  (import side effects shared with Guava: GUAVA_REPO on sys.path, tools._motion guard)
from prompts import LIBERO_SYSTEM as SYSTEM  # noqa: E402,F401

WALL_S = 450.0
MAX_CALLS = 40
