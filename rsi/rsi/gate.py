"""Pre-registered paired acceptance gate (stdlib only).

Inputs: per-pair outcomes of the parent and the candidate on the SAME (task, seed) pairs, the candidate's target tasks
and trigger markers.  A pair is dropped if either arm marks the seed `rejected`.

  J = pairs on target tasks  +  non-target pairs whose candidate trajectory contains a trigger marker
  U = all other pairs (nothing new fired; should behave like the parent)
  gain = task-equal-weighted mean of (candidate - parent) success, in percentage points
  CIs: task-stratified bootstrap (seeds resampled within each task, tasks equally weighted)

With alpha = 0.025 (the paper's setting):
ACCEPT iff  (i)   one-sided 97.5% (1 - alpha) lower bound of the gain on J > 0
       and  (ii)  net wins on the target cluster > 0
       and  (iii) no veto: NOT (gain on U <= veto_untriggered_pp AND two-sided 95% upper bound on U < 0)
       and  (iv)  no trigger marker appears in any parent trajectory (markers must be unique to the new component)
Interim (at interim_fraction of the planned pairs, in planned order): STOP for futility iff the gain on J <= 0 or the
conditional power to end with a positive lower bound, assuming the remaining pairs gain interim_planned_gain_pp, is
< interim_min_power.  The interim can only stop, never accept, so it does not raise the false-acceptance rate.
Screen (per component, small fresh block): DROP iff the two-sided 95% upper bound of the gain on the target cluster
< screen_min_upper_pp.
"""
import collections
import random
from statistics import NormalDist


def split_sets(parent, cand, targets, markers, traces):
    """parent/cand: {(task, seed): result}; traces: {(arm, task, seed): text}.  Returns J, U, target diffs, marker hits."""
    J, U, tgt = [], [], []
    hits = {"parent": collections.Counter(), "cand": collections.Counter()}
    for k in sorted(set(parent) & set(cand)):
        if parent[k]["rejected"] or cand[k]["rejected"]:
            continue
        for arm in ("parent", "cand"):
            tr = traces.get((arm,) + k, "")
            for m in markers:
                hits[arm][m] += m in tr
        d = int(cand[k]["success"]) - int(parent[k]["success"])
        if k[0] in targets:
            tgt.append(d)
            J.append((k[0], d))
        elif any(m in traces.get(("cand",) + k, "") for m in markers):
            J.append((k[0], d))
        else:
            U.append((k[0], d))
    return J, U, tgt, hits


def bootstrap(rows, reps, seed):
    """rows [(task, diff)] -> (point gain in pp, sorted bootstrap gains in pp)."""
    by = collections.defaultdict(list)
    for t, d in rows:
        by[t].append(d)
    if not by:
        return 0.0, [0.0]
    ks = sorted(by)
    point = 100 * sum(sum(by[k]) / len(by[k]) for k in ks) / len(ks)
    rnd = random.Random(seed)
    bs = sorted(100 * sum(sum(rnd.choice(by[k]) for _ in by[k]) / len(by[k]) for k in ks) / len(ks)
                for _ in range(reps))
    return point, bs


def quantile(bs, q):
    return bs[min(len(bs) - 1, max(0, int(q * len(bs))))]


def conditional_power(gain_pp, bs, n_now, n_final, planned_pp, alpha):
    """P(final one-sided lower bound > 0) if the remaining pairs have true gain planned_pp; spread from the bootstrap."""
    se = (sum((b - gain_pp) ** 2 for b in bs) / len(bs)) ** 0.5 / 100
    sig = max(se * n_now ** 0.5, 1e-9)                       # per-pair SD implied by the clustered bootstrap
    n2 = max(n_final - n_now, 1)
    z = NormalDist().inv_cdf(1 - alpha)
    need = (z * sig * n_final ** 0.5 - n_now * gain_pp / 100) / n2   # mean gain the remaining pairs must reach
    return 1 - NormalDist().cdf((need - planned_pp / 100) / (sig / n2 ** 0.5))


def decide(parent, cand, targets, markers, traces, G, mode="final", n_planned_J=None):
    J, U, tgt, hits = split_sets(parent, cand, set(targets), list(markers), traces)
    reps, s, a = G["bootstrap_reps"], G["bootstrap_seed"], G["alpha"]
    if mode == "screen":
        tg, tb = bootstrap([r for r in J if r[0] in targets], reps, s)
        up = quantile(tb, 1 - a)
        return dict(mode=mode, n_target=len(tgt), target_gain_pp=round(tg, 2), target_upper_pp=round(up, 2),
                    decision="DROP" if up < G["screen_min_upper_pp"] else "CONTINUE")
    jg, jb = bootstrap(J, reps, s)
    ug, ub = bootstrap(U, reps, s)
    out = dict(mode=mode, n_J=len(J), n_U=len(U), J_gain_pp=round(jg, 2), J_lower_pp=round(quantile(jb, a), 2),
               target_net=sum(tgt), target_wins=sum(d > 0 for d in tgt), target_losses=sum(d < 0 for d in tgt),
               U_gain_pp=round(ug, 2), U_upper_pp=round(quantile(ub, 1 - a), 2),
               parent_marker_hits=sum(hits["parent"].values()), cand_marker_hits=dict(hits["cand"]))
    if mode == "interim":
        n_final = max(n_planned_J or round(len(J) / G["interim_fraction"]), len(J) + 1)
        cp = conditional_power(jg, jb, len(J), n_final, G["interim_planned_gain_pp"], a)
        out.update(cond_power=round(cp, 3), decision="STOP" if jg <= 0 or cp < G["interim_min_power"] else "CONTINUE")
        return out
    veto = ug <= G["veto_untriggered_pp"] and out["U_upper_pp"] < 0
    reasons = [r for r, bad in (("J lower bound <= 0", out["J_lower_pp"] <= 0), ("target net <= 0", out["target_net"] <= 0),
                                ("untriggered regression veto", veto),
                                ("marker in parent arm", out["parent_marker_hits"] > 0)) if bad]
    out.update(veto=veto, reasons=reasons, decision="ACCEPT" if not reasons else "REJECT")
    return out
