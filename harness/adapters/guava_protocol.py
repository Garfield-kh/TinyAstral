"""Guava episode protocol of the single-arm harness H* (used by single_arm/planner.py, GuavaDriver).

Protocol (as published for the Guava benchmark):
  episode k of task i (BENCHMARK index) uses seed seed_base + k; for evaluation seed_base = 42 + 30 i;
  push tasks: direction = left if k is even else right; camera sideview for the shell game, else frontview;
  action cap = decision cap = 30 for bin_and_tray_simple / set_table / shell_game / red_objects_in_basket else 25;
  motion speed 1, 512x512, 20 settle steps; success = task score() at stop, for physical tasks also after 3x5
  settle steps. The task score is never shown to the planner.
The planner sees: the task sentence, the current camera image, gripper proprioception, and the tool results.
Benchmark task names and episode limits are defined in BENCHMARK and CAPS. SYSTEM = single_arm/prompts.GUAVA_SYSTEM.

Importing this module also puts GUAVA_REPO on sys.path and installs the object-motion guard on tools._motion below
(objects first seen after a call are left out of the obj_motion report); LIBERO-PRO and the zero-shot drivers that
reuse the Guava / LIBERO prompts import it too, so all of them run with the guard."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # harness/: paths, tools, geom
os.environ.setdefault("MUJOCO_GL", "egl")
import paths as _paths  # noqa: E402
_paths.use("single_arm")
GUAVA = _paths.GUAVA
sys.path.insert(0, str(GUAVA))

import tools as _tools_mod  # noqa: E402
from prompts import GUAVA_SYSTEM as SYSTEM  # noqa: E402,F401  (episode.json records the module-level SYSTEM)

_motion_orig = _tools_mod._motion


def _motion_safe(s0, s1):
    """objects first seen after s0 have no reference pose: ignore them (perception-backed snapshots can differ)."""
    s1 = {**s1, "objects": {n: o for n, o in s1["objects"].items() if n in s0["objects"]}}
    return _motion_orig(s0, s1)


_tools_mod._motion = _motion_safe

# ====================================================================== benchmark tasks and limits
BENCHMARK = ['pick_up_carrot', 'tomato_near_potato', 'lemon_in_bin', 'push_pot',
             'cube_stack_reverse', 'apple_juice_reverse_order', 'bin_and_tray_simple',
             'set_table', 'can_in_bin', 'apple_juice_order', 'remove_cube_from_tray',
             'push_basket', 'close_drawer', 'shell_game', 'red_objects_in_basket']
UNSEEN = BENCHMARK[:8]
TRAIN = BENCHMARK[8:]
CAPS = {t: (30 if t in ('bin_and_tray_simple', 'set_table', 'shell_game', 'red_objects_in_basket') else 25) for t in BENCHMARK}
PHYSICAL = {'apple_juice_order', 'apple_juice_reverse_order', 'can_in_bin', 'push_basket', 'red_objects_in_basket'}


def img_url(rgb):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def r2(v, k=3):
    return [round(float(x), k) for x in v]


def _finish(engine, task, A, S, folder, model, seed, k, t0, extra=None, tag=""):
    from PIL import Image
    from guava.inference import benchmark_engine as tm
    final = tm.state(engine)
    audits = [final]
    if task in PHYSICAL:
        for _ in range(3):
            engine.env._settle(5)
            audits.append(tm.state(engine))
    success = all(a["success"] for a in audits)
    A.refresh()
    Image.fromarray(A.rgb()).save(folder / "img" / f"{S['prefix']}final.jpg", quality=90)
    res = {"task": task, "seed": seed, "episode": k, "instruction": S["instruction"], "success": success,
           "success_at_stop": final["success"], "end_reason": S["end"], "decisions": len(S["steps"]), "actions": S["n_actions"],
           "rejects": S["rejects"], "seconds": round(time.time() - t0, 1), "model": model,
           "objects": list(A.names), "scorer_final": final.get("scorer_audit"), **(extra or {})}
    (folder / f"{tag}episode.json").write_text(json.dumps({**res, "system": SYSTEM, "steps": S["steps"], "usage": S["usage"]},
                                                          default=lambda x: r2(x) if hasattr(x, "__len__") else float(x)))
    (folder / f"{tag}result.json").write_text(json.dumps(res, default=str))
    return res


def make_engine(task, sam3_url):
    import yaml
    from guava.inference.rollout_env import RolloutEngine
    cfg = yaml.safe_load((GUAVA / f"configs/{task}.yaml").read_text())
    cam = "sideview" if task == "shell_game" else "frontview"
    cfg["grasp"] = "pca"
    cfg["stop_on_reward"] = False
    cfg["sim"].update(perception="sam3", grasp="pca", motion_speed=1.0, visualize=False, camera_views=[cam])
    cfg["sim"]["camera"] = {"width": 512, "height": 512, "name": cam}
    cfg.setdefault("robot", {}).update(sam3_url=sam3_url, sam3_debug=False, bbox_debug=False,
                                       graspgen_url="tcp://127.0.0.1:5556")
    import tempfile
    f = Path(tempfile.mkstemp(suffix=".yaml")[1])
    f.write_text(yaml.safe_dump(cfg, sort_keys=False))
    eng = RolloutEngine(f, system_prompt="", camera_name=cam)
    eng.initial_settle_steps = 20
    eng.max_actions = eng.max_decisions = CAPS[task]
    eng.table_frame = True
    eng.cfg.stop_on_reward = False
    eng.cfg.sim.motion_speed = 1.0
    eng.env.cfg.motion_speed = 1.0
    return eng
