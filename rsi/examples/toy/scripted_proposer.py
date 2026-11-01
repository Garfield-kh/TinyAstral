"""Scripted stand-in for the master, so the toy loop needs no LLM.  It reads the failed trajectories like a master would
(target cluster = tasks whose failures are mostly top-grasp misses) and proposes three edits in round 1: a neutral one
(logs a re-check after a missed grasp, changes nothing), a harmful one (lifts every object high before placing) and a
useful one (side grasp for tall objects, measured from perception).  Later rounds: nothing left to propose."""
import collections
import difflib


def _diff(old, new, path="harness.py"):
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))


GRASP_OLD = '''    ok = env.grasp(obj, approach="top")
    log.append(f"grasp {obj} approach=top -> {'OK' if ok else 'MISSED'}")
'''
EDITS = [
    ("log a re-check after a missed grasp", ["[recheck]"], "tool_grasp", GRASP_OLD, GRASP_OLD +
     '''    if not ok:
        log.append("[recheck] grasp reported a miss")
'''),
    ("lift every object high before placing", ["[high_lift]"], "tool_place", '''    ok = env.place(obj, target)
''', '''    env.lift(obj, height=0.30)
    log.append("[high_lift] lifted to 0.30 m before placing")
    ok = env.place(obj, target)
'''),
    ("side grasp for tall objects (height from perception)", ["[side_grasp]"], "tool_grasp", GRASP_OLD, '''    approach = "top"
    if env.measure(obj)["height_m"] > 0.18:
        approach = "side"
        log.append("[side_grasp] tall object, approaching from the side")
    ok = env.grasp(obj, approach=approach)
    log.append(f"grasp {obj} approach={approach} -> {'OK' if ok else 'MISSED'}")
'''),
]


class ScriptedProposer:
    def __init__(self, cfg):
        pass

    def propose(self, ctx, round_dir):
        src = ctx["harness"]["harness.py"]
        miss = collections.Counter(f["task"] for f in ctx["failures"] if "approach=top -> MISSED" in f["trace"])
        target = sorted(t for t, n in miss.items() if n >= 5)
        out = []
        for summary, markers, func, old, new in EDITS:
            if old not in src or not target or summary in ctx["ledger"]:
                continue                               # already applied / nothing to fix / already tried
            out.append(dict(summary=summary, proposer="scripted", failure_class="grasp failure", target_tasks=target, markers=markers, diff=_diff(src, src.replace(old, new)),
                            capabilities=[dict(capability=summary, category="tool", failure_class="grasp failure")]))
        return out[: ctx["n"]]
