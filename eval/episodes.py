"""Fixed evaluation episode lists of the three training benchmarks (the episodes behind the paper's Table 2)."""
import json
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "harness"

# ------------------------------------------------------------------ RoboTwin-50: 50 tasks x episodes 0-9 = 500 episodes
# Episode k of a task is the k-th valid RoboTwin seed (>= 100000) of that task, read from harness/data/robotwin_seedcache.
ROBOTWIN_EPISODES = 10


def robotwin_tasks():
    return sorted(p.name.split("__")[0] for p in (HARNESS / "data/robotwin_seedcache").glob("*__demo_randomized__seed0.json"))


def robotwin_episodes():
    """[(task, episode, seed)]"""
    out = []
    for t in robotwin_tasks():
        seeds = json.loads((HARNESS / f"data/robotwin_seedcache/{t}__demo_randomized__seed0.json").read_text())["seeds"]
        out += [(t, k, seeds[k]) for k in range(ROBOTWIN_EPISODES)]
    return out


# ------------------------------------------------------------------ Guava: 15 tasks x 30 episodes = 450 (first 8 tasks = unseen)
GUAVA_TASKS = ['pick_up_carrot', 'tomato_near_potato', 'lemon_in_bin', 'push_pot',
               'cube_stack_reverse', 'apple_juice_reverse_order', 'bin_and_tray_simple',
               'set_table', 'can_in_bin', 'apple_juice_order', 'remove_cube_from_tray',
               'push_basket', 'close_drawer', 'shell_game', 'red_objects_in_basket']
GUAVA_UNSEEN = GUAVA_TASKS[:8]
GUAVA_EPISODES = 30
GUAVA_CHUNK = 5            # episodes per process (as in the reported runs)


def guava_seed_base(task):
    return 42 + 30 * GUAVA_TASKS.index(task)


def guava_episodes():
    """[(task, k, seed, dirname)]"""
    return [(t, k, guava_seed_base(t) + k, f"{t}__k{k:03d}_s{guava_seed_base(t) + k}") for t in GUAVA_TASKS for k in range(GUAVA_EPISODES)]


# ------------------------------------------------------------------ LIBERO-PRO: 300 episodes
# Task = MotorMind `--suite libero_pro` aggregate index: block of 50 per suite (0 spatial, 50 object, 100 goal); inside a block
# 0-9 original tasks, 10-19 language, 20-29 object, 30-39 swap, 40-49 task perturbation. Seeds 0 and 1 (LIBERO init states).
LIBERO_BASE = list(range(0, 10)) + list(range(50, 60)) + list(range(100, 110))
LIBERO_PERT = list(range(10, 50)) + list(range(60, 100)) + list(range(110, 150))
LIBERO_SEEDS = (0, 1)
LIBERO_SUITE = {0: "spatial", 1: "spatial_lan", 2: "spatial_object", 3: "spatial_swap", 4: "spatial_task",
                5: "object", 6: "object_lan", 7: "object_object", 8: "object_swap", 9: "object_task",
                10: "goal", 11: "goal_lan", 12: "goal_object", 13: "goal_swap", 14: "goal_task"}


def libero_episodes():
    """[(task, seed, part, dirname)]: base 30 tasks x 2 seeds + perturbed 120 x 2 = 300."""
    out = []
    for part, ts in (("base", LIBERO_BASE), ("pert", LIBERO_PERT)):
        for s in LIBERO_SEEDS:
            out += [(t, s, part, f"t{t}_s{s}") for t in ts]
    return out
