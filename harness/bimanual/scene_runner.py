#!/usr/bin/env python
"""harness gate 1, per task: for fixed scene seeds (first N seeds >= seed0 whose scene can be built; never filtered by
outcome) run
  (1) the scripted expert with its semantic API instrumented: every grasp_actor / place_actor / move_by_displacement /
      move_to_pose / open_gripper / close_gripper / back_to_origin call is logged with its semantic arguments (objects by
      name, contact / functional point ids, targets also as a relation to the nearest object), and every executed
      T.move group with pre/post state (objects, joints, grippers, contacts / held), a restorable sim state and
      post-execution images  -> expert_ok
  (2) the compiled teacher: the parsed semantic calls executed through the shared tools (tools.py) on a fresh scene of
      the same seed, without running the task script -> teacher_ok, per-call codes, first failure.
Usage: scene_runner.py --task T --n 13 --seed0 21000 [--images]"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))              # harness/: paths, tools, geom
import paths as _paths  # noqa: E402
_paths.use("bimanual")
sys.path.insert(0, str(Path(__file__).resolve().parent / "scene"))
import cot_l2 as M  # noqa: E402  (sim setup helpers)

sys.path.insert(0, str(HERE))
import geom as Gm  # noqa: E402
from adapter_robotwin import RoboTwinAdapter, act2dict  # noqa: E402
from tools import Tools, resolve  # noqa: E402
import sapien  # noqa: E402
from envs.utils.actor_utils import Actor  # noqa: E402

OUT = Path(os.environ.get("TINYASTRAL_OUT", "runs")) / "scenes"
SEM = ("grasp_actor", "place_actor", "move_by_displacement", "move_to_pose", "open_gripper", "close_gripper", "back_to_origin")
REL_MAX = 0.10


def _aabb_dist(p, lo, hi):
    p, lo, hi = map(np.asarray, (p, lo, hi))
    return float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi))))


def relation(A, pose7, point, exclude=()):
    best, bd = None, REL_MAX
    for n in A.objects():
        if n in exclude:
            continue
        lo, hi = A.aabb(n)
        d = _aabb_dist(point, lo, hi)
        if d < bd:
            best, bd = n, d
    return {"ref": best, "rel": (Gm.rel(A.obj_pose(best), pose7) if best else np.asarray(pose7, float)).tolist()}


def relation_snap(S, pose7, point, exclude=()):
    """relation of a target pose to the nearest object in a recorded snapshot; objects held by a hand are excluded
    (they move with the hand, so they cannot anchor a hand / place target)."""
    exclude = set(exclude) | set(S["held"].values())
    best, bd = None, REL_MAX
    for n, o in S["objects"].items():
        if n in exclude:
            continue
        d = _aabb_dist(point, o["lo"], o["hi"])
        if d < bd:
            best, bd = n, d
    return {"ref": best, "rel": (Gm.rel(S["objects"][best]["pose"], pose7) if best else np.asarray(pose7, float)).tolist()}


def relink(A, g, call, item):
    """recompute the relations of a logged group from its pre-execution snapshot."""
    S = g["pre"]
    if call is not None and call["api"] in ("place_actor", "move_to_pose"):
        call = dict(call)
        tp = np.asarray(call["args"]["target_pose"], float)
        tp7 = tp if tp.size == 7 else np.concatenate([tp[:3], [1, 0, 0, 0]])
        if call["api"] == "place_actor":
            call["target_rel"] = relation_snap(S, tp7, tp7[:3], exclude=(call["args"]["actor"]["obj"],))
        else:
            call["target_rel"] = relation_snap(S, tp7, A.tcp_point(tp7))
    if call is None:
        item = dict(item)
        acts = []
        for d in item["actions"]:
            d = dict(d)
            if d["type"] == "move" and any(d["pose"]):
                d["rel"] = relation_snap(S, d["pose"], A.tcp_point(d["pose"]))
            acts.append(d)
        item["actions"] = acts
    return call, item


def norm(A, v):
    if isinstance(v, Actor):
        return {"obj": A.name_of(v)}
    if isinstance(v, sapien.Pose):
        return list(map(float, v.p)) + list(map(float, v.q))
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [norm(A, x) for x in v]
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if hasattr(v, "arm") or type(v).__name__ == "ArmTag":
        return str(v)
    return repr(v)


# ====================================================================== (1) instrumented expert
def run_expert(sim, seed, sdir, images):
    T = M.setup(sim, seed)
    A = RoboTwinAdapter(sim)
    calls, groups, reg = [], [], {}
    ever = [False]
    orig = {n: getattr(T, n) for n in SEM}
    orig_move = T.move

    def wrap(name):
        f = orig[name]
        sig = inspect.signature(f)

        def w(*a, **kw):
            r = f(*a, **kw)
            try:
                b = sig.bind(*a, **kw)
                b.apply_defaults()
                args = {k: norm(A, v) for k, v in b.arguments.items() if k != "args"}
                args.update({k: norm(A, v) for k, v in (b.arguments.get("args") or {}).items()})
            except Exception as exc:  # noqa: BLE001
                args = {"_bind_error": str(exc)}
            c = {"id": len(calls), "api": name, "args": args}
            if name == "place_actor" and args.get("target_pose") is not None:
                tp = np.asarray(args["target_pose"], float)
                tp7 = tp if tp.size == 7 else np.concatenate([tp[:3], [1, 0, 0, 0]])
                c["target_rel"] = relation(A, tp7, tp7[:3], exclude=(args["actor"]["obj"],))
                c["target_is_position"] = tp.size != 7
            if name == "move_to_pose":
                tp7 = np.asarray(args["target_pose"], float)
                c["target_rel"] = relation(A, tp7, A.tcp_point(tp7))
            try:
                c["planned"] = [act2dict(x) for x in r[1]] if r and r[0] is not None else []
            except Exception:  # noqa: BLE001
                c["planned"] = []
            calls.append(c)
            reg[id(r)] = (r, c["id"])
            return r
        return w

    for n in SEM:
        setattr(T, n, wrap(n))

    def mv(a1, a2=None, save_freq=-1):
        items = []
        for x in (a1, a2):
            if x is None or x[0] is None:
                continue
            cid = reg[id(x)][1] if id(x) in reg and reg[id(x)][0] is x else None
            acts = [act2dict(a) for a in x[1]]
            if cid is None:          # raw Action list built by the script: moves become relations to the nearest object
                for d in acts:
                    if d["type"] == "move" and any(d["pose"]):
                        d["rel"] = relation(A, d["pose"], A.tcp_point(d["pose"]))
            it = {"arm": str(x[0]), "call": cid, "actions": acts}
            if cid is not None:      # the script edited the returned actions (e.g. raised a pre-grasp pose): keep the offsets
                pl = calls[cid].get("planned") or []
                if len(pl) == len(acts):
                    adj = [(np.asarray(e["pose"][:3]) - np.asarray(p["pose"][:3])).tolist() if e["type"] == "move" and p["type"] == "move"
                           else [0.0, 0.0, 0.0] for e, p in zip(acts, pl)]
                    if any(abs(v) > 1e-4 for d in adj for v in d):
                        it["adjust"] = adj
            items.append(it)
        g = {"k": len(groups), "items": items, "pre": A.snapshot(), "pre_state": A.save_state()}
        r = orig_move(a1, a2, save_freq=save_freq) if a2 is not None else orig_move(a1, save_freq=save_freq)
        g["post"] = A.snapshot()
        g["success_now"] = A.success()
        ever[0] = ever[0] or g["success_now"]
        g["plan_success"] = bool(T.plan_success)
        if images:
            ims = A.images()
            for cam, im in (ims or {}).items():
                try:
                    p = sdir / "images" / f"g{len(groups):03d}_{cam}.jpg"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    (im if hasattr(im, "save") else __import__("PIL.Image").Image.fromarray(np.asarray(im))).save(p, quality=80)
                except Exception:  # noqa: BLE001
                    pass
        groups.append(g)
        return r

    T.move = mv
    err = None
    info = {}
    try:
        ret = T.play_once()
        info = ret.get("info", {}) if isinstance(ret, dict) else {}
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        for n in SEM:
            if n in vars(T):
                delattr(T, n)
        if "move" in vars(T):
            del T.move
    final_ok = A.success()
    ok = final_ok or ever[0]
    return {"info": info, "expert_ok": ok, "expert_final_ok": final_ok, "expert_error": err, "calls": calls, "groups": groups,
            "final": A.snapshot(), "objects": A.objects()}


# ====================================================================== (2) compiled teacher
def plan_for(tools, call, item):
    p = _plan_for(tools, call, item)
    adj = item.get("adjust")
    if adj and p.get("actions") and len(p["actions"]) == len(adj):
        acts = []
        for d, o in zip(p["actions"], adj):
            d = dict(d)
            if d["type"] == "move":
                d["pose"] = (np.asarray(d["pose"], float) + np.concatenate([o, [0, 0, 0, 0]])).tolist()
            acts.append(d)
        p["actions"] = acts
    return p


def _plan_for(tools, call, item):
    arm = item["arm"]
    if call is None:
        acts = [({"type": "move", "pose": resolve(tools.A, d["rel"]).tolist(), **({"constraint": d["constraint"]} if d.get("constraint") else {})}
                 if d.get("rel") else d) for d in item["actions"]]
        return {"arm": arm, "tool": "raw_rel", "actions": acts}
    a, api = call["args"], call["api"]
    if api == "grasp_actor":
        return tools.grasp(arm, a["actor"]["obj"], contact=a.get("contact_point_id"), pre_dist=a.get("pre_grasp_dis", 0.1),
                           depth=a.get("grasp_dis", 0), grip=a.get("gripper_pos", 0.0))
    if api == "place_actor":
        known = {"self", "actor", "arm_tag", "target_pose", "functional_point_id", "pre_dis", "dis", "is_open"}
        extra = {k: v for k, v in a.items() if k not in known and not k.startswith("_")}
        tgt = call["target_rel"]
        if call.get("target_is_position"):
            tp = np.asarray(tools.A.obj_pose(tgt["ref"]) if tgt["ref"] else [0, 0, 0, 1, 0, 0, 0], float)
            pos = Gm.compose(tp, tgt["rel"])[:3] if tgt["ref"] else np.asarray(tgt["rel"][:3])
            return _place_pos(tools, arm, a, pos, extra)
        return tools.place(arm, a["actor"]["obj"], tgt, a.get("functional_point_id"), a.get("pre_dis", 0.1), a.get("dis", 0.02),
                           a.get("is_open", True), extra)
    if api == "move_by_displacement":
        return tools.move_delta(arm, a.get("x", 0.0), a.get("y", 0.0), a.get("z", 0.0), a.get("quat"), a.get("move_axis", "world"))
    if api == "move_to_pose":
        return tools.move_pose(arm, call["target_rel"])
    if api in ("open_gripper", "close_gripper"):
        return tools.gripper(arm, a.get("pos", 1.0 if api == "open_gripper" else 0.0))
    if api == "back_to_origin":
        return tools.home(arm)
    return {"arm": arm, "tool": "raw", "actions": item["actions"]}


def _place_pos(tools, arm, a, pos, extra):
    """place_actor called with a 3-d target position (orientation free)."""
    if tools.cancelled.get(arm) == a["actor"]["obj"]:
        return {"arm": arm, "tool": "place", "obj": a["actor"]["obj"], "skip": "NOT_HELD"}
    acts = tools.A.place_plan(a["actor"]["obj"], arm, list(pos), a.get("functional_point_id"), a.get("pre_dis", 0.1),
                              a.get("dis", 0.02), a.get("is_open", True), extra)
    if acts is None:
        return {"arm": arm, "tool": "place", "obj": a["actor"]["obj"], "skip": "PLAN_FAIL"}
    return {"arm": arm, "tool": "place", "obj": a["actor"]["obj"], "actions": acts, "release": a["actor"]["obj"] if a.get("is_open", True) else None}


def run_teacher(sim, seed, ex):
    T = M.setup(sim, seed)
    A = RoboTwinAdapter(sim)
    tools = Tools(A)
    calls = {c["id"]: c for c in ex["calls"]}
    steps, first_fail, ever = [], None, False
    n_raw = 0
    for g in ex["groups"]:
        plans = []
        for it in g["items"]:
            c = calls.get(it["call"]) if it["call"] is not None else None
            n_raw += c is None
            c, it = relink(A, g, c, it)
            try:
                plans.append(plan_for(tools, c, it))
            except Exception as exc:  # noqa: BLE001
                plans.append({"arm": it["arm"], "tool": (c or {}).get("api", "raw"), "skip": f"PLAN_ERROR {type(exc).__name__}: {exc}"[:200]})
        r = tools.execute(plans)
        r["k"] = g["k"]
        r["apis"] = [(calls.get(it["call"]) or {}).get("api", "raw") for it in g["items"]]
        r["success_now"] = A.success()
        ever = ever or r["success_now"]
        steps.append(r)
        bad = {a: v["code"] for a, v in r["arms"].items() if not v["ok"]}
        if bad and first_fail is None:
            first_fail = {"k": g["k"], "codes": bad, "apis": r["apis"]}
    final_ok = A.success()
    return {"teacher_ok": ever or final_ok, "teacher_final_ok": final_ok, "first_fail": first_fail, "n_raw_items": n_raw, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--seed0", type=int, default=21000)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--teacher_only", default="", help="re-run only the teacher from scenes/<id>/log.json into this dir")
    a = ap.parse_args()
    M.VP.A = Namespace(guard=False, final_check=True, task_checker="v2", wall=720)
    (M.OUT / "_tmp").mkdir(parents=True, exist_ok=True)
    sim = M.SimEnv(a.task, seed_dir=str(M.OUT / "_seedcache"), task_config="demo_randomized", seed_base_index=-1)
    done, seed, invalid = 0, a.seed0, []
    while done < a.n and seed < a.seed0 + 200:
        sdir = OUT / f"{a.task}__s{seed}"
        if (sdir / "result.json").exists() and not a.teacher_only:
            r = json.loads((sdir / "result.json").read_text())
            done += r.get("scene_valid", True)
            seed += 1
            continue
        if a.teacher_only and not (sdir / "result.json").exists():
            break
        t0 = time.time()
        if a.teacher_only:
            lp = sdir / "log.json"
            od = OUT.parent / a.teacher_only / sdir.name
            if not lp.exists() or (od / "result.json").exists():
                seed += 1
                continue
            od.mkdir(parents=True, exist_ok=True)
            ex = json.loads(lp.read_text())
            try:
                te = run_teacher(sim, seed, ex)
            except Exception as exc:  # noqa: BLE001
                te = {"teacher_ok": False, "first_fail": {"error": f"{type(exc).__name__}: {exc}"}, "tb": traceback.format_exc()[-1200:]}
            try:
                sim.rd.close_episode()
            except Exception:  # noqa: BLE001
                pass
            res = dict(r0 := json.loads((sdir / "result.json").read_text()))
            res.update(teacher_ok=te["teacher_ok"], first_fail=te.get("first_fail"), n_raw_items=te.get("n_raw_items"),
                       seconds_teacher=round(time.time() - t0, 1))
            (od / "teacher.json").write_text(json.dumps(te, default=str))
            (od / "result.json").write_text(json.dumps(res, default=str))
            print(a.task, seed, {k: res[k] for k in ("expert_ok", "teacher_ok", "first_fail")}, flush=True)
            seed += 1
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        try:
            ex = run_expert(sim, seed, sdir, a.images)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            try:
                sim.rd.close_episode()
            except Exception:  # noqa: BLE001
                pass
            valid = "UnStable" not in msg and "unstable" not in msg
            (sdir / "result.json").write_text(json.dumps({"task": a.task, "seed": seed, "scene_valid": valid, "error": msg,
                                                          "tb": traceback.format_exc()[-800:]}))
            print(a.task, seed, "scene error", msg[:120], flush=True)
            done += valid
            seed += 1
            continue
        try:
            sim.rd.close_episode()
        except Exception:  # noqa: BLE001
            pass
        (sdir / "log.json").write_text(json.dumps(ex, default=str))
        try:
            te = run_teacher(sim, seed, ex)
        except Exception as exc:  # noqa: BLE001
            te = {"teacher_ok": False, "first_fail": {"error": f"{type(exc).__name__}: {exc}"}, "tb": traceback.format_exc()[-1200:]}
        try:
            sim.rd.close_episode()
        except Exception:  # noqa: BLE001
            pass
        res = {"task": a.task, "seed": seed, "scene_valid": True, "expert_ok": ex["expert_ok"], "expert_error": ex["expert_error"],
               "teacher_ok": te["teacher_ok"], "first_fail": te.get("first_fail"), "n_calls": len(ex["calls"]),
               "n_groups": len(ex["groups"]), "n_raw_items": te.get("n_raw_items"), "seconds": round(time.time() - t0, 1)}
        (sdir / "teacher.json").write_text(json.dumps(te, default=str))
        (sdir / "result.json").write_text(json.dumps(res, default=str))
        print(a.task, seed, {k: res[k] for k in ("expert_ok", "teacher_ok", "first_fail", "n_raw_items", "seconds")}, flush=True)
        done += 1
        seed += 1
    sim.rd.close()


if __name__ == "__main__":
    main()
