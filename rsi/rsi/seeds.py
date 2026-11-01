"""Seed splits.  train = data collection after the loop (never used here), dev = failure mining (read trajectories),
selection = paired acceptance tests, eval = final report only.

Ranges are half-open [lo, hi) and must be disjoint.  Blocks are taken with a cursor that only moves forward, so every
test draws FRESH seeds and no seed is reused for a second decision.  The loop never asks for train or eval seeds;
`pairs(..., "eval")` raises unless allow_eval=True, which only the final, post-loop evaluation passes.
"""
LOOP_SPLITS = ("dev", "selection")


def check_disjoint(seeds):
    for k in ("dev", "selection", "eval"):
        if k not in seeds:
            raise ValueError(f"seed split `{k}` must be declared")
    rs = sorted((tuple(v), k) for k, v in seeds.items())
    for ((lo1, hi1), a), ((lo2, hi2), b) in zip(rs, rs[1:]):
        if hi1 > lo2:
            raise ValueError(f"seed splits {a} [{lo1},{hi1}) and {b} [{lo2},{hi2}) overlap")


def pairs(cfg, split, start, per_task, tasks=None, allow_eval=False):
    """(task, seed) list for one block of `per_task` seeds starting at offset `start` inside the split.  Seeds are the
    outer loop, so any prefix of the list (e.g. the interim half) is balanced across tasks."""
    if split not in LOOP_SPLITS + ("eval",) or (split == "eval" and not allow_eval):
        raise PermissionError(f"`{split}` seeds never enter the improvement loop")
    check_disjoint(cfg["seeds"])
    lo, hi = cfg["seeds"][split]
    if lo + start + per_task > hi:
        raise ValueError(f"split {split} exhausted (offset {start}, size {per_task})")
    return [(t, s) for s in range(lo + start, lo + start + per_task) for t in (tasks or cfg["tasks"])]
