"""Object-centred crops for the checker. The harness cameras are fixed relative to the table top (RoboDawn adds the table
height bias to the camera position), so a table-relative point (x, y in cm world, z in cm above the table) projects the same
way in every episode."""
import math
from pathlib import Path

import numpy as np
from PIL import Image

import paths as _paths  # noqa: E402
OUT = _paths.OUT / "checker_exp1"
TABLE = 0.74
CAMS = {"agent": {"pos": [0.0, -0.36, 1.50], "fwd": [0.0, 0.42, -0.91], "left": [-1.0, 0.0, 0.0], "fovy": 60.0, "W": 640, "H": 480},
        "top": {"pos": [0.0, -0.10, 1.65], "fwd": [0.0, 0.0, -1.0], "left": [-1.0, 0.0, 0.0], "fovy": 50.0, "W": 640, "H": 480}}


def project(cam, pts_cm):
    c = CAMS[cam]
    f = np.asarray(c["fwd"], float)
    f /= np.linalg.norm(f)
    l = np.asarray(c["left"], float)
    l /= np.linalg.norm(l)
    u = np.cross(f, l)
    pos = np.asarray(c["pos"], float) - [0, 0, TABLE]          # relative to the table top
    P = np.asarray(pts_cm, float) / 100.0
    d = P - pos
    xc, yc, zc = -(d @ l), d @ u, d @ f
    fy = (c["H"] / 2) / math.tan(math.radians(c["fovy"]) / 2)
    return np.stack([c["W"] / 2 + fy * xc / zc, c["H"] / 2 - fy * yc / zc], 1)


def box_px(S, objs, cam):
    pts = []
    for o in objs:
        if o not in S["objects"]:
            continue
        lo, hi = S["objects"][o]["lo"], S["objects"][o]["hi"]
        pts += [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    if not pts:
        return None
    uv = project(cam, pts)
    return uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()


def crop(S, objs, cam, out_rel, pad=0.6, min_side=110, size=256):
    """crop around the union of the objects (and nothing else); returns the relative path or None."""
    b = box_px(S, objs, cam)
    if b is None:
        return None
    c = CAMS[cam]
    x0, y0, x1, y1 = b
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    side = min(max(side, min_side), min(c["W"], c["H"]))
    x0 = int(np.clip(cx - side / 2, 0, c["W"] - side))
    y0 = int(np.clip(cy - side / 2, 0, c["H"] - side))
    img = Image.open(OUT / S["frames"][cam]).crop((x0, y0, x0 + int(side), y0 + int(side))).resize((size, size))
    p = OUT / out_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p, quality=88)
    return out_rel
