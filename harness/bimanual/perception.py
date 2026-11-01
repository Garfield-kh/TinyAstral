"""Camera-only localisation (v2): same SAM3 + native RGB-D multi-view fusion as perception_core, but also returns the
fused object point cloud (world frame, metres, subsampled) so tools can solve grasps / placements from geometry.
No simulator object state is read."""
from __future__ import annotations

import sys

import numpy as np

import perception_core as F  # noqa: E402
from camera_capture import capture  # noqa: E402

MAX_PTS = 4000


def _near(a, b, tol=0.012):
    """two candidate point sets touch (camera-only test, subsampled)."""
    if np.any(np.minimum(a["hi"], b["hi"]) - np.maximum(a["lo"], b["lo"]) < -tol):
        return False
    from scipy.spatial import cKDTree
    pa = a["pts"][:: max(1, len(a["pts"]) // 400)]
    pb = b["pts"][:: max(1, len(b["pts"]) // 400)]
    d, _ = cKDTree(pa).query(pb, k=1)
    return float(np.min(d)) < tol and float(np.mean(d < 3 * tol)) > 0.05


def cluster(C, rad=0.05):
    """multi-view consensus: candidates of one prompt are merged when their centres are close or their 3-D point sets
    touch (a large object seen whole by the head camera and partly by a wrist camera is one object)."""
    C = [c for c in C if c["score"] >= 0.1]
    for c in C:
        c["lo"], c["hi"] = c["pts"].min(0), c["pts"].max(0)
    par = list(range(len(C)))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            if f(i) == f(j):
                continue
            if np.linalg.norm(C[i]["c"][:2] - C[j]["c"][:2]) < rad or _near(C[i], C[j]):
                par[f(i)] = f(j)
    groups = {}
    for i, c in enumerate(C):
        groups.setdefault(f(i), []).append(c)
    out = []
    for ms in groups.values():
        best = {}
        for m in ms:
            if m["view"] not in best or m["score"] > best[m["view"]]["score"]:
                best[m["view"]] = m
        keep = list(best.values())
        # same-view duplicates beyond the best mask still add points if they overlap the object
        out.append({"c": np.median(np.concatenate([m["pts"] for m in keep]), 0), "m": keep, "views": set(best),
                    "support": sum(m["score"] for m in keep)})
    return out


SYN = {"skillet": "frying pan", "kitchenpot": "cooking pot", "pad": "mat", "seal": "stamp", "phonestand": "phone holder",
       "toycar": "toy", "alarm_clock": "clock", "tabletrashbin": "trash can", "pillbottle": "bottle", "basket": "basket box",
       "coaster": "round coaster", "roller": "rolling pin"}


def _pool(views, segs, groups, prompt_of_group, table_z, xlim, priors, pos_prior, flat):
    pool = []
    for b, ns in groups.items():
        sizes = [np.sort(np.asarray(priors[n]["size"], float)) for n in ns if priors and n in priors]
        prior = {"size": np.median(sizes, axis=0)} if sizes else None
        pp = [np.asarray(pos_prior[n], float) for n in ns if pos_prior and n in pos_prior]
        cands = []
        for mh, w in ((0.002, 1.0), (-0.002, 0.9)):    # also keep near-table points: mats / pads / paper are flat
            for g in cluster(F._cands(views, segs, prompt_of_group(b), table_z, xlim, min_height=mh)):
                g["flat_w"] = w
                cands.append(g)
        for g in cands:
            if g["flat_w"] < 1.0:
                lo_, hi_ = F._box(g, table_z)
                if hi_[2] - table_z < 0.012:      # genuinely flat region (mat, pad, paper): no penalty
                    g["flat_w"] = 1.0
            g["shape_weight"] = g["flat_w"] * (F._shape_weight(g, prior, table_z) if g["flat_w"] == 1.0 else 1.0)
            bonus = 1.0
            if pp:
                d = min(float(np.linalg.norm(g["c"][:2] - p[:2])) for p in pp)
                bonus = 1.0 + 2.0 * np.exp(-d / 0.04)
            pool.append((g["support"] * (1 + 0.25 * (len(g["views"]) - 1)) * g["shape_weight"] * bonus, b, g))
    return pool


def _choose(pool, groups, table_z, taken):
    pool.sort(key=lambda x: -x[0])
    chosen = {b: [] for b in groups}
    for sup, b, g in pool:
        if len(chosen[b]) >= len(groups[b]):
            continue
        if any(F._same_object(g, previous, table_z) for previous in taken):
            continue
        chosen[b].append(g)
        taken.append(g)
    return chosen


PROMPT = {"seal": "stamp"}       # vocabulary: words SAM3 reads differently from the asset name


def prompt_of(n):
    return PROMPT.get(F.base_of(n), F.prompt_of(n))


def _robot_points(rd):
    """proprioception: robot link origins plus midpoints of consecutive links (arm body / gripper occupancy)."""
    P = []
    try:
        r = rd.env.robot
        ents = {id(e): e for e in (r.left_entity, r.right_entity)}
        for e in ents.values():
            prev = None
            for l in e.get_links():
                p = np.asarray(l.get_entity_pose().p, float)
                P.append(p)
                if prev is not None and np.linalg.norm(p - prev) < 0.4:
                    for f in (0.25, 0.5, 0.75):
                        P.append(prev + f * (p - prev))
                prev = p
        for f in (r.get_left_tcp_pose, r.get_right_tcp_pose):
            P.append(np.asarray(f()[:3], float))
    except Exception:  # noqa: BLE001
        return None
    return np.asarray(P) if P else None


def _drop_robot(views, rp, rad=0.06):
    if rp is None:
        return views
    from scipy.spatial import cKDTree
    tree = cKDTree(rp)
    out = []
    for v in views:
        v = dict(v)
        valid = v["valid"].copy()
        idx = np.flatnonzero(valid.ravel())
        if len(idx):
            d, _ = tree.query(v["xyz"].reshape(-1, 3)[idx], k=1)
            vv = valid.ravel()
            vv[idx[d < rad]] = False
            v["valid"] = vv.reshape(valid.shape)
        out.append(v)
    return out


def perceive(rd, names, table_z, priors=None, flat_names=None, prompt_overrides=None, xlim=0.75, pos_prior=None, drop_robot=True):
    """pos_prior {name: world xy (m)}: temporal continuity -- clusters near the last known position of a name are
    preferred (the scene only changes where the robot acted). Missing names are retried as flat surfaces and with a
    generic synonym."""
    views = capture(rd)
    segs_views = views
    if drop_robot:
        views = _drop_robot(views, _robot_points(rd))
    flat_names = set(flat_names or [])
    prompt_overrides = prompt_overrides or {}
    groups = {}
    for n in names:
        key = (F.base_of(n), prompt_overrides.get(n, prompt_of(n)), n in flat_names)
        groups.setdefault(key, []).append(n)
    prompts = sorted({key[1] for key in groups})
    segs = [F.segment(v["rgb"], prompts) for v in views]
    taken = []
    chosen = _choose(_pool(views, segs, groups, lambda b: b[1], table_z, xlim, priors, pos_prior, False), groups, table_z, taken)
    missing = {b: ns for b, ns in groups.items() if not chosen[b] and b[0] in SYN}
    if missing:      # synonym retry
        segs2 = [F.segment(v["rgb"], sorted({SYN[b[0]] for b in missing})) for v in views]
        ch3 = _choose(_pool(views, segs2, missing, lambda b: SYN[b[0]], table_z, xlim, priors, pos_prior, False), missing, table_z, taken)
        for b in missing:
            chosen[b] = ch3[b]
    out = {}
    rng = np.random.default_rng(0)
    for b, ns in groups.items():
        keep = sorted(chosen[b], key=lambda g: g["c"][0])
        pairs = list(zip(sorted(ns), keep))
        if len(ns) > 1 and keep and pos_prior and all(n in pos_prior for n in ns):     # keep identities of duplicates
            pairs, left = [], list(keep)
            for n in sorted(ns, key=lambda n: min(np.linalg.norm(g["c"][:2] - np.asarray(pos_prior[n])[:2]) for g in keep)):
                if not left:
                    break
                j = min(range(len(left)), key=lambda j: np.linalg.norm(left[j]["c"][:2] - np.asarray(pos_prior[n])[:2]))
                g = left.pop(j)
                pairs.append((n, g))
        for n, g in pairs:
            lo, hi = F._box(g, table_z)
            pts = np.concatenate([m["pts"] for m in g["m"]])
            if len(pts) > MAX_PTS:
                pts = pts[rng.choice(len(pts), MAX_PTS, replace=False)]
            lo_c = (lo - [0, 0, table_z]) * 100
            hi_c = (hi - [0, 0, table_z]) * 100
            out[n] = {"center": (lo_c + hi_c) / 2, "size": hi_c - lo_c, "lo": lo_c, "hi": hi_c, "pts": pts,
                      "views": sorted(g["views"])}
    return out
