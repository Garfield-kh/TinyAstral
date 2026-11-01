"""Camera-only object localisation for the V+ harness (replaces the oracle OBJECTS block).

Input : exactly native head, left wrist and right wrist RGB-D cameras: RGB + per-pixel camera-frame
        position (= depth + intrinsics) + camera pose. No simulator object state is read.
Detect: SAM3 text-prompt segmentation (local server sam3_srv.py, weights hf_models/sam3) with the object names.
Lift  : masked pixels -> world points; candidate masks of all views are clustered in 3D (multi-view consensus),
        the best-supported clusters are kept (n of them for numbered duplicates, ordered left->right like the names).
Output: {name: {center, size, lo, hi}} in cm, table frame (z = height above the table) and the same OBJECTS text.
Env   : PERC_SAM3 (default http://127.0.0.1:8130)."""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

import numpy as np
from PIL import Image

_GPU = (os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
SAM3 = os.environ.get("PERC_SAM3", f"http://127.0.0.1:{8130 + int(_GPU) if _GPU.isdigit() else 8130}")
SPLIT = {"toycar": "toy car", "rubikscube": "rubik's cube", "phonestand": "phone stand", "playingcards": "playing cards",
         "tissuebox": "tissue box", "remotecontrol": "remote control", "mousepad": "mouse pad", "pencup": "pen holder",
         "tabletrashbin": "trash bin", "paymentsign": "payment sign", "callbell": "call bell", "kitchenpot": "pot",
         "pillbottle": "pill bottle", "brush": "brush", "plant": "plant", "fries": "fries"}


PHRASE = {"glazed rim porcelain ramekin": "ramekin", "akita black bowl": "black bowl", "alphabet soup": "soup can",
          "tomato sauce": "tomato sauce bottle", "cream cheese": "cream cheese box", "butter": "butter box",
          "flat stove": "stove", "wooden cabinet": "cabinet", "white cabinet": "cabinet", "chocolate pudding": "pudding box",
          "orange juice": "juice carton", "bbq sauce": "sauce bottle", "new salad dressing": "salad dressing bottle",
          "salad dressing": "salad dressing bottle", "wine bottle": "wine bottle", "wine rack": "wine rack"}


def prompt_of(name: str) -> str:
    b = re.sub(r"_\d+$", "", name)
    p = " ".join(SPLIT.get(w, w) for w in b.split("_"))
    return PHRASE.get(p, p)


def base_of(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


# ------------------------------------------------------------------ capture
from camera_capture import capture


# ------------------------------------------------------------------ segmentation
def _b64png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def segment(rgb, prompts, top_k=4, min_score=0.04):
    body = json.dumps({"image_b64": _b64png(rgb), "prompts": prompts, "top_k": top_k, "min_score": min_score}).encode()
    req = urllib.request.Request(SAM3 + "/seg", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())["results"]
    out = {}
    for p, lst in r.items():
        out[p] = [{"score": m["score"], "mask": np.array(Image.open(io.BytesIO(base64.b64decode(m["mask"])))) > 127}
                  for m in lst]
    return out


def _erode(m, k=2):
    from scipy.ndimage import binary_erosion
    e = binary_erosion(m, iterations=k)
    return e if e.sum() >= 15 else m


# ------------------------------------------------------------------ fusion
def _cands(views, segs, prompt, table_z, xlim=0.75, min_height=0.002):
    C = []
    for v, s in zip(views, segs):
        for m in s.get(prompt, []):
            mk = _erode(m["mask"]) & v["valid"]
            pts = v["xyz"][mk]
            pts = pts[(pts[:, 2] > table_z + min_height) & (pts[:, 2] < table_z + 0.40) & ((np.abs(pts[:, 0]) < xlim) if xlim else True)]   # table workspace only (no robot body)
            if len(pts) < 15:
                continue
            med = np.median(pts, 0)
            d = np.linalg.norm(pts - med, axis=1)
            pts = pts[d < max(0.03, np.percentile(d, 90) * 1.5)]
            if len(pts) < 15:
                continue
            C.append({"view": v["name"], "score": m["score"], "pts": pts, "c": np.median(pts, 0)})
    return C


def _cluster(C, rad=0.05):
    cl = []
    for c in sorted(C, key=lambda c: -c["score"]):
        for g in cl:
            if np.linalg.norm(g["c"][:2] - c["c"][:2]) < rad and c["view"] not in g["views"]:
                g["m"].append(c); g["views"].add(c["view"])
                break
        else:
            cl.append({"c": c["c"], "m": [c], "views": {c["view"]}})
    for g in cl:
        g["support"] = sum(m["score"] for m in g["m"])
    return cl


def _box(g, table_z):
    pts = np.concatenate([m["pts"] for m in g["m"]])
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    lo[2] = max(table_z, min(lo[2], np.percentile(pts[:, 2], 1)))
    if lo[2] - table_z < 0.015:           # resting on the table: the hidden bottom is the table
        lo[2] = table_z
    return lo, hi



def _same_object(g, previous, table_z):
    """Suppress competing labels only for strongly overlapping 3-D boxes.

    Close visible-surface medians alone do not imply the same physical object:
    two stacked objects can be within the old 5cm vertical threshold.
    IoU also avoids collapsing a small object inside a larger container.
    """
    lo, hi = _box(g, table_z)
    other_lo, other_hi = _box(previous, table_z)
    extent = np.maximum(hi - lo, 0.002)
    other_extent = np.maximum(other_hi - other_lo, 0.002)
    center, other_center = (lo + hi) / 2, (other_lo + other_hi) / 2
    lo, hi = center - extent / 2, center + extent / 2
    other_lo, other_hi = other_center - other_extent / 2, other_center + other_extent / 2
    intersection = float(np.prod(np.maximum(0, np.minimum(hi, other_hi) - np.maximum(lo, other_lo))))
    union = float(np.prod(extent) + np.prod(other_extent) - intersection)
    return intersection / max(union, 1e-12) > 0.6

def _shape_weight(g, prior, table_z):
    """Downweight candidates much thinner than the first camera observation.

    Uses only measured RGB-D boxes, never simulator object dimensions.
    Sorting dimensions reduces orientation dependence. Planar/invalid initial
    observations cannot provide a useful shape prior and are ignored.
    """
    if prior is None:
        return 1.0
    initial = np.sort(np.asarray(prior["size"], float) / 100)
    if initial.shape != (3,) or not np.isfinite(initial).all() or np.any(initial <= 0.002):
        return 1.0
    lo, hi = _box(g, table_z)
    measured = np.sort(np.maximum(hi - lo, 0.002))
    return float(np.sqrt(np.clip(np.min(measured / initial), 0.0, 1.0)))


def perceive(rd, names, table_z, views=None, return_debug=False, xlim=0.75, priors=None, flat_names=None, prompt_overrides=None):
    views = views or capture(rd)
    flat_names = set(flat_names or [])
    prompt_overrides = prompt_overrides or {}
    groups = {}
    for n in names:
        # Distinct descriptions and surface modes must not share candidate pools.
        key = (base_of(n), prompt_overrides.get(n, prompt_of(n)), n in flat_names)
        groups.setdefault(key, []).append(n)
    prompts = sorted({key[1] for key in groups})
    segs = [segment(v["rgb"], prompts) for v in views]
    res, dbg = {}, {}
    pool = []                                   # (support, base, cluster)
    for b, ns in groups.items():
        sizes = [np.sort(np.asarray(priors[n]["size"], float)) for n in ns if priors and n in priors]
        prior = {"size": np.median(sizes, axis=0)} if sizes else None
        min_height = -0.002 if b[2] else 0.002
        for g in _cluster(_cands(views, segs, b[1], table_z, xlim, min_height=min_height)):
            g["shape_weight"] = _shape_weight(g, prior, table_z)
            pool.append((g["support"] * (1 + 0.25 * (len(g["views"]) - 1)) * g["shape_weight"], b, g))
    pool.sort(key=lambda x: -x[0])
    taken, chosen = [], {b: [] for b in groups}
    for sup, b, g in pool:                      # one physical object -> one name (best-supported claim wins)
        if len(chosen[b]) >= len(groups[b]):
            continue
        if any(_same_object(g, previous, table_z) for previous in taken):
            continue
        chosen[b].append(g)
        taken.append(g)
    for b, ns in groups.items():
        keep = sorted(chosen[b], key=lambda g: g["c"][0])
        for n, g in zip(sorted(ns), keep):
            lo, hi = _box(g, table_z)
            res[n] = {"lo": lo, "hi": hi}
            dbg[n] = {"views": sorted(g["views"]), "support": round(g["support"], 2), "shape_weight": round(g["shape_weight"], 3)}
    out = {}
    for n, o in res.items():
        lo = (o["lo"] - [0, 0, table_z]) * 100
        hi = (o["hi"] - [0, 0, table_z]) * 100
        out[n] = {"center": [round(float(x), 1) for x in (lo + hi) / 2], "size": [round(float(x), 1) for x in hi - lo],
                  "lo": [round(float(x), 1) for x in lo], "hi": [round(float(x), 1) for x in hi]}
    return (out, dbg) if return_debug else out


def objects_text(objs, names=None):
    lines = []
    for n in (names or objs):
        o = objs.get(n)
        if o is None:
            lines.append(f"- {n}: not found by the cameras")
            continue
        lines.append(f"- {n}: centre ({o['center'][0]:.1f}, {o['center'][1]:.1f}, {o['center'][2]:.1f}), size "
                     f"({o['size'][0]:.0f}, {o['size'][1]:.0f}, {o['size'][2]:.0f}), top z {o['hi'][2]:.1f}")
    return "OBJECTS (estimated from the RGB-D cameras in cm, may be off by 1-3 cm; z = height above the table):\n" + "\n".join(lines)
