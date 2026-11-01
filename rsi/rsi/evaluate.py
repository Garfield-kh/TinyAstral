"""Run harness versions on (task, seed) pairs through a user-supplied runner.

Runner contract (rules.yaml `runner`), either
  callable: "module:function"     called as function(harness_dir, task, seed, out_dir, teacher, max_calls) -> dict
                                  (module looked up next to the rules file first; `teacher` is the rules.yaml `teacher`
                                  mapping, `max_calls` the per-episode tool-call budget runner.max_calls)
  command:  "python my_runner.py --harness {harness} --task {task} --seed {seed} --out {out}"
                                  must write {out}/result.json; the teacher endpoint and the budget are passed as
                                  TEACHER_BASE_URL, TEACHER_MODEL, TEACHER_TEMPERATURE, MAX_CALLS environment variables
The teacher pi_T is frozen: every arm of every test calls the same endpoint with the same decoding settings and the
same per-episode budget.
Result: {"success": bool, "rejected": bool (seed infeasible for reasons unrelated to the harness -> pair dropped),
         "trace": "path to a text trajectory (tool calls, tool results, model reasoning)"}.
A runner that exceeds `runner.timeout_s` or crashes is recorded as a failure with `"infra": true` in the record,
so a crashed episode can never silently disappear from a paired test.
"""
import concurrent.futures as cf
import importlib
import json
import os
import pathlib
import shlex
import subprocess
import sys

from .config import resolve


def _callable(cfg):
    spec = cfg["runner"].get("callable")
    if not spec:
        return None
    if cfg["_dir"] not in sys.path:
        sys.path.insert(0, cfg["_dir"])
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def run_one(cfg, harness_dir, task, seed, out_dir):
    out = pathlib.Path(out_dir) / f"{task}__{seed}"
    rf = out / "result.json"
    if rf.exists():                                              # resumable
        return json.loads(rf.read_text())
    out.mkdir(parents=True, exist_ok=True)
    fn = _callable(cfg)
    try:
        if fn:
            r = fn(str(harness_dir), task, int(seed), str(out), cfg["teacher"], cfg["runner"].get("max_calls"))
        else:
            cmd = cfg["runner"]["command"].format(harness=shlex.quote(str(harness_dir)), task=shlex.quote(task),
                                                  seed=int(seed), out=shlex.quote(str(out)))
            env = dict(os.environ, TEACHER_BASE_URL=str(cfg["teacher"].get("base_url", "")),
                       TEACHER_MODEL=str(cfg["teacher"]["model"]), TEACHER_TEMPERATURE=str(cfg["teacher"].get("temperature", 0)),
                       MAX_CALLS=str(cfg["runner"].get("max_calls", "")))
            subprocess.run(cmd, shell=True, env=env, timeout=cfg["runner"].get("timeout_s", 1800), check=True,
                           stdout=open(out / "runner.log", "w"), stderr=subprocess.STDOUT)
            r = json.loads(rf.read_text())
    except Exception as e:  # noqa: BLE001  timeout, crash, bad output -> failure, recorded
        r = {"success": False, "rejected": False, "infra": True, "error": repr(e)[:300]}
    r = {"success": bool(r.get("success")), "rejected": bool(r.get("rejected")), "trace": r.get("trace"),
         **{k: v for k, v in r.items() if k in ("infra", "error")}}
    rf.write_text(json.dumps(r))
    return r


def evaluate(cfg, harness_dir, pairs, out_dir):
    """{(task, seed): result} for one harness version."""
    with cf.ThreadPoolExecutor(cfg["budget"]["workers"]) as ex:
        futs = {ex.submit(run_one, cfg, harness_dir, t, s, out_dir): (t, s) for t, s in pairs}
        return {futs[f]: f.result() for f in cf.as_completed(futs)}


def evaluate_paired(cfg, parent_dir, cand_dir, pairs, out_dir):
    """Both arms on the same pairs, interleaved in planned order (alternating which arm goes first) so that both arms
    see the same load and infrastructure conditions, and any prefix of the plan is balanced."""
    jobs = []
    for i, (t, s) in enumerate(pairs):
        arms = [("parent", parent_dir), ("cand", cand_dir)]
        jobs += [(a, d, t, s) for a, d in (arms if i % 2 == 0 else arms[::-1])]
    res = {"parent": {}, "cand": {}}
    with cf.ThreadPoolExecutor(cfg["budget"]["workers"]) as ex:
        futs = {ex.submit(run_one, cfg, d, t, s, pathlib.Path(out_dir) / a): (a, t, s) for a, d, t, s in jobs}
        for f in cf.as_completed(futs):
            a, t, s = futs[f]
            res[a][(t, s)] = f.result()
    return res["parent"], res["cand"]


def read_trace(r):
    p = r.get("trace")
    return pathlib.Path(p).read_text(errors="ignore") if p and pathlib.Path(p).exists() else ""


if __name__ == "__main__":                       # final report on the eval split, only after the loop has finished
    import argparse
    from . import seeds
    from .config import load_rules
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--version", required=True, help="accepted version id (directory under workdir/versions)")
    ap.add_argument("--per-task", type=int, default=10)
    a = ap.parse_args()
    cfg = load_rules(a.config)
    wd = resolve(cfg, cfg["workdir"])
    ps = seeds.pairs(cfg, "eval", 0, a.per_task, tasks=cfg["tasks"] + (cfg.get("heldout_tasks") or []), allow_eval=True)
    r = evaluate(cfg, wd / "versions" / a.version, ps, wd / "eval" / a.version)
    ok = [v for v in r.values() if not v["rejected"]]
    print(f"{a.version}: {sum(v['success'] for v in ok)}/{len(ok)} on eval")
