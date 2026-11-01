"""Generic geometric solvers for the v2 tool layer (no task names, no asset annotations, no simulator object state).

Input is a camera point cloud (world frame, metres) of one object as produced by perception.py.
  grasp_specs(pts, tz, arm, part)  -> ordered list of grasp specs {p, m, why}: fingertip point p and closing axis m (unit xy)
  candidates(spec, arm)            -> ordered (approach a, closing c) hand-axis pairs (top-down first, then tilted)
  body_center / top_center / floor_z -> placement references
Gripper convention (RoboTwin aloha): hand x axis = approach direction, hand y axis = finger closing axis."""
from __future__ import annotations

import numpy as np
from transforms3d.quaternions import mat2quat

MAX_W = 0.085          # widest cross-section the open gripper can enclose comfortably (m)
ARM_BASE = {"left": np.array([-0.30, -0.35]), "right": np.array([0.30, -0.35])}


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def yaw_quat(th):
    return np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])


def hand_quat(a, c):
    a = unit(a)
    c = unit(np.asarray(c, float) - a * float(np.dot(c, a)))
    R = np.column_stack([a, c, np.cross(a, c)])
    return np.asarray(mat2quat(R), float)


def pca2(xy):
    """centre, major axis u, minor axis m of a footprint from the minimum-area bounding rectangle (robust for squares)."""
    xy = np.asarray(xy, float)
    c = np.median(xy, 0)
    if len(xy) < 3:
        return c, np.array([1.0, 0]), np.array([0, 1.0])
    X = xy - c
    best = None
    for th in np.radians(np.arange(0, 90, 2.0)):
        e1 = np.array([np.cos(th), np.sin(th)])
        e2 = np.array([-e1[1], e1[0]])
        a, b = X @ e1, X @ e2
        l1 = np.percentile(a, 98) - np.percentile(a, 2)
        l2 = np.percentile(b, 98) - np.percentile(b, 2)
        if best is None or l1 * l2 < best[0]:
            best = (l1 * l2, e1, e2, l1, l2)
    _, e1, e2, l1, l2 = best
    u = e1 if l1 >= l2 else e2
    if u[0] < 0 or (abs(u[0]) < 1e-6 and u[1] < 0):
        u = -u
    c = c + e1 * (np.percentile(X @ e1, 98) + np.percentile(X @ e1, 2)) / 2 + e2 * (np.percentile(X @ e2, 98) + np.percentile(X @ e2, 2)) / 2
    return c, u, np.array([-u[1], u[0]])


def top_axes(pts):
    """footprint axes from the upper surface points (side points seen obliquely blur the outline)."""
    z = pts[:, 2]
    top = np.percentile(z, 97)
    s = z >= top - max(0.01, 0.3 * (top - np.percentile(z, 3)))
    return pca2(pts[s if s.sum() >= 20 else slice(None), :2])


def extent(v):
    return float(np.percentile(v, 97) - np.percentile(v, 3)) if len(v) else 0.0


def profile(pts):
    """width profile along the major axis: bins of 1 cm -> (t_centre, width, count)."""
    xy = pts[:, :2]
    c, u, m = pca2(xy)
    t, w = (xy - c) @ u, (xy - c) @ m
    lo, hi = np.percentile(t, 1), np.percentile(t, 99)
    edges = np.arange(lo, hi + 0.01, 0.01)
    prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (t >= a) & (t < b)
        if s.sum() >= 4:
            prof.append(((a + b) / 2, float(w[s].max() - w[s].min()), int(s.sum())))
    return c, u, m, t, w, prof


def handles(pts):
    """thin protrusions at the ends of the major axis (handles of pots, pans, mugs ...). Returns list of
    {end: +1/-1, mask, length}."""
    c, u, m, t, w, prof = profile(pts)
    if len(prof) < 6:
        return [], (c, u, m, t, w)
    widths = np.array([p[1] for p in prof])
    body = float(np.percentile(widths, 80))
    out = []
    for end in (1, -1):
        seq = prof if end == -1 else prof[::-1]
        L, tin = 0, None
        for tc, wd, n in seq:
            if wd < 0.55 * body:
                L += 1
                tin = tc
            else:
                break
        if 2 <= L <= len(prof) // 2 and body > 0.05:
            mask = (t >= tin - 0.005) if end == 1 else (t <= tin + 0.005)
            out.append({"end": end, "mask": mask, "length": L / 100.0})
    return out, (c, u, m, t, w)


def body_mask(pts):
    hs, (c, u, m, t, w) = handles(pts)
    keep = np.ones(len(pts), bool)
    for h in hs:
        keep &= ~h["mask"]
    return keep if keep.sum() > 20 else np.ones(len(pts), bool)


def body_center(pts):
    k = body_mask(pts)
    xy = pts[k, :2]
    return (np.percentile(xy, 3, 0) + np.percentile(xy, 97, 0)) / 2


def top_center(pts):
    z = pts[:, 2]
    top = np.percentile(z, 97)
    s = z >= top - 0.01
    return np.median(pts[s, :2], 0), float(top)


def floor_z(pts, xy, r=0.035):
    d = np.linalg.norm(pts[:, :2] - xy, axis=1)
    s = d < r
    if s.sum() < 10:
        return None
    return float(np.percentile(pts[s, 2], 10))


def ring_center(xy):
    """least-squares circle through rim points; None unless they really form a ring (hollow top)."""
    xy = np.asarray(xy, float)
    if len(xy) < 20:
        return None
    A = np.c_[xy, np.ones(len(xy))]
    b = -(xy ** 2).sum(1)
    try:
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    except Exception:  # noqa: BLE001
        return None
    c = np.array([-D / 2, -E / 2])
    r2 = c @ c - F
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    d = np.linalg.norm(xy - c, axis=1)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    cover = len(np.unique(np.floor((ang + np.pi) / (np.pi / 6))))
    if 0.015 < r < 0.07 and np.mean(np.abs(d - r)) < 0.25 * r and np.mean(d < 0.5 * r) < 0.1 and cover >= 5:
        return c
    return None


def _spec(region, tz, m, why, depth_max=0.03, top_face=False):
    z = region[:, 2]
    top, bot = np.percentile(z, 95), np.percentile(z, 5)
    if bot - tz < 0.02:
        bot = tz
    d = min(depth_max, max(0.008, 0.5 * (top - bot)))
    xy = np.median(region[:, :2], 0)
    if top_face:                 # the upper face is seen whole from above: its centre is the best estimate of the body axis
        s = z >= top - 0.01
        if s.sum() >= 15:
            xy = np.median(region[s, :2], 0)
            cc = ring_center(region[s, :2])
            if cc is not None:       # open top (cup, can, bowl): the rim is a ring, use its circle centre
                xy = cc
    p = np.r_[xy, max(top - d, tz + 0.006)]
    return {"p": p, "m": unit(m), "why": why, "top": float(top)}


def _end_region(pts, u, t, end, frac=0.22, cap=0.05):
    L = extent(t)
    seg = min(cap, frac * L)
    if end == 1:
        return t >= np.percentile(t, 97) - seg
    return t <= np.percentile(t, 3) + seg


def arch(pts, tz):
    """carrying handle arching over the top of a container: a thin band of points well above the rim."""
    z = pts[:, 2]
    rim = np.percentile(z, 70)
    top = np.percentile(z, 99)
    if top - rim < 0.03:
        return None
    sel = z > rim + 0.5 * (top - rim)
    if sel.sum() < 15:
        return None
    reg = pts[sel]
    c, u, m = pca2(reg[:, :2])
    w = extent((reg[:, :2] - c) @ m)
    if w > 0.05:
        return None
    s = reg[:, 2] >= np.percentile(reg[:, 2], 70)
    p = np.r_[np.median(reg[s, :2], 0), np.percentile(reg[:, 2], 90) - 0.008]
    return {"p": p, "m": m, "why": "handle arching over the top", "top": float(top)}


def _edge(pts, tz, arm, c):
    d = unit(ARM_BASE[arm] - c)
    xy = pts[:, :2] - c
    s = xy @ d
    q = xy @ np.array([-d[1], d[0]])
    sel = (s >= np.percentile(s, 98) - 0.015) & (np.abs(q) < 0.025)
    if sel.sum() < 8:
        sel = s >= np.percentile(s, 95)
    return _spec(pts[sel], tz, d, "rim/edge nearest the arm", depth_max=0.02)


def grasp_specs(pts, tz, arm, part=None):
    pts = np.asarray(pts, float)
    hs, (c, u, m, t, w) = handles(pts)
    Lu, Lm = extent(t), extent(w)
    part = (str(part).lower().strip() if part else "")
    specs = []
    side = -1 if arm == "left" else 1

    def end_spec(end, why):
        sel = _end_region(pts, u, t, end)
        reg = pts[sel]
        _, u2, m2 = pca2(reg[:, :2])
        mm = m if extent((reg[:, :2] - reg[:, :2].mean(0)) @ m) <= MAX_W else m2
        return _spec(reg, tz, mm, why)

    if part in ("left end", "left", "right end", "right", "front end", "back end", "front", "back"):
        want = {"left": np.array([-1.0, 0]), "right": np.array([1.0, 0]), "front": np.array([0, -1.0]), "back": np.array([0, 1.0])}[part.split()[0]]
        end = 1 if float(u @ want) >= 0 else -1
        specs.append(end_spec(end, f"{part} of the major axis"))
    elif part in ("handle", "handles") and not hs and arch(pts, tz) is not None:
        specs.append(arch(pts, tz))
    elif part in ("handle", "handles"):
        if hs:
            hs = sorted(hs, key=lambda h: -side * h["end"] * float(np.sign(u[0]) or 1))
            bc = body_center(pts)
            for h in hs:
                reg = pts[h["mask"]]
                sp = _spec(reg, tz, m, "thin protrusion (handle)")
                sp["out"] = unit(sp["p"][:2] - bc)
                specs.append(sp)
        else:
            end = 1 if side * u[0] >= 0 else -1
            specs.append(end_spec(end, "no thin protrusion found; end nearest the arm"))
    elif part in ("rim", "edge", "lip", "wall"):
        specs.append(_edge(pts, tz, arm, c))
    else:
        if Lm <= MAX_W:
            ct, ut, mt = top_axes(pts)
            specs.append(_spec(pts, tz, mt, "body centre across the narrow side", top_face=True))
            if Lu <= MAX_W:
                specs.append(_spec(pts, tz, ut, "body centre across the long side", top_face=True))
        elif arch(pts, tz) is not None:
            specs.append(arch(pts, tz))
            specs.append(_edge(pts, tz, arm, c))
        elif hs:
            bc = body_center(pts)
            for h in sorted(hs, key=lambda h: -side * h["end"] * float(np.sign(u[0]) or 1)):
                sp = _spec(pts[h["mask"]], tz, m, "object too wide; thin protrusion (handle)")
                sp["out"] = unit(sp["p"][:2] - bc)
                specs.append(sp)
            specs.append(_edge(pts, tz, arm, c))
        else:
            specs.append(_edge(pts, tz, arm, c))
            end = 1 if side * u[0] >= 0 else -1
            specs.append(end_spec(end, "object too wide; end nearest the arm"))
    for s in specs:
        s["shape"] = {"long_cm": round(Lu * 100, 1), "narrow_cm": round(Lm * 100, 1), "handles": len(hs)}
    return specs


def candidates(spec, arm, tilts=(0.0, 25.0, 45.0)):
    """ordered (approach, closing) pairs for a spec. Tilted approaches lean along the reach direction of the arm."""
    p, m = spec["p"], np.r_[spec["m"], 0.0]
    r = unit(np.r_[p[:2] - ARM_BASE[arm], 0.0])
    out = []
    for al in tilts:
        a_ = np.radians(al)
        a = unit(np.r_[r[:2] * np.sin(a_), -np.cos(a_)])
        c = m - a * float(m @ a)
        if np.linalg.norm(c) < 0.5:
            continue
        for sgn in (1, -1):
            out.append((al, a, unit(c) * sgn))
    if spec.get("out") is not None:                 # handle: approach from outside, fingers pinch it from above / below
        o = np.r_[spec["out"], 0.0]
        for be in (30.0, 50.0, 15.0):
            b = np.radians(be)
            a = unit(-o * np.cos(b) + np.array([0, 0, -np.sin(b)]))
            z = np.array([0, 0, 1.0])
            c = z - a * float(z @ a)
            for sgn in (1, -1):
                out.append((100 + be, a, unit(c) * sgn))
    return out


def door_geometry(pts, tz):
    """front door of a box-like appliance from its point cloud (robot looks along +y; the front face is the min-y face).
    Handle = points sticking out in front of the face (else the face region 80% across from the hinge side);
    hinge = the vertical front edge on the side away from the handle. Returns dict or None."""
    pts = np.asarray(pts, float)
    y = pts[:, 1]
    zt = np.percentile(pts[:, 2], 97)
    x0, x1 = np.percentile(pts[:, 0], 2), np.percentile(pts[:, 0], 98)
    face = float(np.percentile(y, 12))
    pro = pts[(y < face - 0.006) & (pts[:, 2] > tz + 0.03)]
    if len(pro) >= 15:
        hx = float(np.median(pro[:, 0]))
        hz = float(np.percentile(pro[:, 2], 80))
        hy = float(np.median(pro[:, 1]))
        found = True
    else:
        hx, hz, hy, found = x0 + 0.75 * (x1 - x0), zt - 0.03, face, False
    hinge_x = x0 if hx > (x0 + x1) / 2 else x1
    return {"handle": np.array([hx, hy, hz]), "hinge": np.array([hinge_x, face]), "face_y": face, "found": found,
            "top": float(zt), "width": float(x1 - x0)}
