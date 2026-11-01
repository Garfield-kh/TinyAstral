"""The improvement loop (Method, Stage 1).  Only the harness changes; the teacher and the rules are frozen.

  for round k = 1 .. max_rounds:
     1. read trajectories   run H_k on a fresh dev block; collect failed trajectories
     2. propose edits       the master returns candidate components (diff + target cluster + trigger markers)
     3. test and accept     per component: lint -> sandbox copy -> audit -> direction screen on a small fresh block;
                            surviving components are merged into one candidate H'; H' and H_k run interleaved on the
                            same pre-registered selection pairs with a futility interim; the gate decides; ACCEPT makes
                            H_{k+1} = H'.  Every outcome is written to the ledger.
  python -m rsi.loop --config rules.yaml
"""
import argparse
import hashlib
import itertools
import json
import pathlib
import shutil
import subprocess

from . import gate, patch, seeds
from .config import load_rules, resolve
from .evaluate import evaluate, evaluate_paired, read_trace
from .ledger import Ledger
from .proposer import load_proposer


def harness_sources(d, exts=(".py", ".md", ".txt", ".json", ".yaml")):
    d = pathlib.Path(d)
    return {str(p.relative_to(d)): p.read_text() for p in sorted(d.rglob("*")) if p.is_file() and p.suffix in exts}


def traces(parent, cand):
    t = {("parent",) + k: read_trace(v) for k, v in parent.items()}
    t.update({("cand",) + k: read_trace(v) for k, v in cand.items()})
    return t


def record(cfg, rnd, name, parent, cand):
    return dict(round=rnd, candidate=name, parent=parent, summary=cand.get("summary", ""), proposer=cand.get("proposer"),
                failure_class=cand.get("failure_class"), failure_detail=cand.get("failure_detail"),
                target_tasks=cand.get("target_tasks", []), markers=cand.get("markers", []),
                capabilities=cand.get("capabilities", []), components=cand.get("components"),
                diff_sha256=hashlib.sha256(cand["diff"].encode()).hexdigest(), rules_sha256=cfg["_sha256"],
                accepted_version=None)


def admissible(cfg, cand, parent_dir, cand_dir, diffs):
    """Lint, sandbox application and the independent audit.  Returns a list of problems (empty = admissible)."""
    tg, mk = cand.get("target_tasks", []), cand.get("markers", [])
    probs = patch.lint(cand["diff"], cfg["tasks"] + (cfg.get("heldout_tasks") or []), mk, tg,
                       cfg.get("frozen_paths") or [], cfg.get("memory_dir"))
    probs += [f"unknown target task {t}" for t in tg if t not in cfg["tasks"]] + ([] if tg else ["no target cluster"])
    shutil.rmtree(cand_dir, ignore_errors=True)
    if not probs:
        try:
            patch.apply(parent_dir, cand_dir, diffs)
        except Exception as e:  # noqa: BLE001
            probs.append(f"diff does not apply: {e}")
    if not probs and cfg.get("audit_command"):          # independent code audit: untriggered path == parent, generality
        r = subprocess.run(cfg["audit_command"].format(parent=parent_dir, cand=cand_dir), shell=True,
                           capture_output=True, text=True)
        if r.returncode:
            probs.append("audit failed: " + (r.stdout + r.stderr)[-300:])
    return probs


def run_round(cfg, L, st, rnd, cands, cur, wd, rdir):
    """Screen each component, merge the survivors, run one pre-registered paired test.
    Returns (dir, name, candidate) of the accepted H', or None."""
    G, B = cfg["gate"], cfg["budget"]
    n_tasks, survivors = len(cfg["tasks"]), []

    def afford(per_task):
        if st["episodes"] + 2 * n_tasks * per_task <= B["max_episodes"]:
            return True
        print("[budget] episode budget exhausted", flush=True)
        return False

    for i, c in enumerate(cands):
        name, cdir = f"{st['version']}.r{rnd}c{i}", wd / "candidates" / f"{st['version']}.r{rnd}c{i}"
        base = record(cfg, rnd, name, st["version"], c)
        probs = admissible(cfg, c, cur, cdir, [c["diff"]])
        if probs:
            L.append(dict(base, stage="lint", decision="INADMISSIBLE", reasons=probs))
            print(f"  {name} [{c.get('summary', '')}]: INADMISSIBLE ({'; '.join(probs)})", flush=True)
            continue
        k = G["screen_pairs_per_task"]
        if not k:
            survivors.append((0.0, i, c))
            continue
        if not afford(k):
            return None
        b = st["sel_cursor"]; st["sel_cursor"] += k
        ps = seeds.pairs(cfg, "selection", b, k)
        p, q = evaluate_paired(cfg, cur, cdir, ps, rdir / f"{name}_screen")
        st["episodes"] += 2 * len(ps)
        s = gate.decide(p, q, c["target_tasks"], c["markers"], traces(p, q), G, "screen")
        L.append(dict(base, stage="screen", decision=s["decision"], stats=s, selection_offset=b,
                      reasons=["target upper bound below the screen threshold"] if s["decision"] == "DROP" else []))
        print(f"  {name} [{c.get('summary', '')}]: screen {s['decision']}  target {s['target_gain_pp']:+.1f}pp "
              f"(upper {s['target_upper_pp']:+.1f})", flush=True)
        if s["decision"] == "CONTINUE":
            survivors.append((s["target_gain_pp"], i, c))
    if not survivors:
        return None

    merged, kept = [], []                               # best screen gain first; a component that no longer applies
    for _, i, c in sorted(survivors, key=lambda x: (-x[0], x[1])):     # on top of the others waits for a later round
        probe = wd / "candidates" / "_merge_probe"
        shutil.rmtree(probe, ignore_errors=True)
        try:
            patch.apply(cur, probe, merged + [c["diff"]])
            merged.append(c["diff"]); kept.append((i, c))
        except Exception as e:  # noqa: BLE001
            L.append(dict(record(cfg, rnd, f"{st['version']}.r{rnd}c{i}", st["version"], c), stage="merge",
                          decision="DROP", reasons=[f"does not apply on top of the other survivors: {e}"]))
        shutil.rmtree(probe, ignore_errors=True)
    name = f"{st['version']}.r{rnd}"
    H = dict(summary=" + ".join(c.get("summary", "") for _, c in kept), diff="".join(merged),
             components=[f"{st['version']}.r{rnd}c{i}" for i, _ in kept],
             target_tasks=sorted({t for _, c in kept for t in c["target_tasks"]}),
             markers=sorted({m for _, c in kept for m in c["markers"]}),
             proposer=sorted({str(c.get("proposer")) for _, c in kept}),
             capabilities=[x for _, c in kept for x in c.get("capabilities", [])])
    base, hdir = record(cfg, rnd, name, st["version"], H), wd / "candidates" / name
    probs = admissible(cfg, H, cur, hdir, merged)
    if probs:
        L.append(dict(base, stage="lint", decision="INADMISSIBLE", reasons=probs))
        return None
    k = cfg["selection"]["pairs_per_task"]               # number of pairs fixed in advance
    if not afford(k):
        return None
    b = st["sel_cursor"]; st["sel_cursor"] += k
    ps = seeds.pairs(cfg, "selection", b, k)
    n1 = int(len(ps) * G["interim_fraction"])
    p, q = evaluate_paired(cfg, cur, hdir, ps[:n1], rdir / name)
    st["episodes"] += 2 * n1
    s = gate.decide(p, q, H["target_tasks"], H["markers"], traces(p, q), G, "interim")
    if s["decision"] == "CONTINUE":
        p, q = evaluate_paired(cfg, cur, hdir, ps, rdir / name)
        st["episodes"] += 2 * (len(ps) - n1)
        s = gate.decide(p, q, H["target_tasks"], H["markers"], traces(p, q), G, "final")
    why = s.get("reasons") or (["futility at interim"] if s["decision"] == "STOP" else [])
    L.append(dict(base, stage=s["mode"], decision=s["decision"], stats=s, reasons=why, selection_offset=b))
    print(f"  {name} = {' + '.join(H['components'])}: {s['decision']}  J {s['J_gain_pp']:+.1f}pp (lower "
          f"{s['J_lower_pp']:+.1f}), target net {s['target_net']:+d}, U {s['U_gain_pp']:+.1f}pp"
          + (f"  ({'; '.join(why)})" if why else ""), flush=True)
    return (hdir, name, H) if s["decision"] == "ACCEPT" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_rules(ap.parse_args().config)
    seeds.check_disjoint(cfg["seeds"])
    wd = resolve(cfg, cfg["workdir"])
    vdir, L = wd / "versions", Ledger(wd)
    st = L.state("H0")
    if not (vdir / "H0").exists():
        shutil.copytree(resolve(cfg, cfg["harness"]), vdir / "H0")
    proposer, B = load_proposer(cfg), cfg["budget"]
    while st["round"] < B["max_rounds"]:
        st["round"] += 1
        rnd, cur = st["round"], vdir / st["version"]
        rdir = wd / "rounds" / f"r{rnd}"
        rdir.mkdir(parents=True, exist_ok=True)
        n = cfg["dev"]["pairs_per_task"]
        dev = evaluate(cfg, cur, seeds.pairs(cfg, "dev", (rnd - 1) * n, n), rdir / "dev")
        st["episodes"] += len(dev)
        by = {}
        for (t, s), r in sorted(dev.items()):
            if not r["success"] and not r["rejected"]:
                by.setdefault(t, []).append(dict(task=t, seed=s, trace=read_trace(r)))
        fails = [f for grp in itertools.zip_longest(*by.values()) for f in grp if f]   # round-robin over tasks
        ok = sum(r["success"] for r in dev.values() if not r["rejected"])
        print(f"[round {rnd}] {st['version']}: dev {ok}/{len(dev)}; {len(fails)} failed trajectories", flush=True)
        ctx = dict(round=rnd, version=st["version"], harness=harness_sources(cur), failures=fails[: cfg["dev"]["max_failures_shown"]],
                   ledger=L.summary(), n=B["max_candidates_per_round"], tasks=cfg["tasks"],
                   rules=json.dumps({k: cfg.get(k) for k in ("gate", "budget", "selection", "frozen_paths", "memory_dir")}, indent=1))
        cands = proposer.propose(ctx, rdir)
        if not cands:
            print(f"[round {rnd}] master proposed nothing; stopping", flush=True)
            break
        acc = run_round(cfg, L, st, rnd, cands[: B["max_candidates_per_round"]], cur, wd, rdir)
        if acc:
            hdir, name, H = acc
            new = f"H{sum(1 for p in vdir.iterdir() if p.is_dir())}"
            shutil.copytree(hdir, vdir / new)
            L.append(dict(record(cfg, rnd, name, st["version"], H), stage="promote", decision="ACCEPT",
                          reasons=[], accepted_version=new))
            print(f"  -> {new} = {name}", flush=True)
            st["version"] = new
        L.save_state(st)
    L.save_state(st)
    print(f"final harness H* = {st['version']} after {st['round']} rounds, {st['episodes']} episodes; ledger {L.path}")


if __name__ == "__main__":
    main()
