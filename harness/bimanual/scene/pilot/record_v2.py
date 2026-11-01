#!/usr/bin/env python
"""v2 oracle-expert recorder (tools v2: staged grasp/place, semantic place targets, visibility for the checker).

Differences to v1 (record.py, whose hooks are reused):
  * every snapshot also stores, per camera, the number of segmentation pixels of each task object (visibility);
  * if the expert's last place has no retreat, the tool's default retreat (5 cm back along the arm) is executed;
  * perturbation = grasp-point offset: the expert's grasp poses are shifted 5.5-7 cm sideways (perpendicular to approach
    and finger axis) before execution; a missed grasp triggers the recovery open -> back off -> locate -> grasp;
  * translation into tools v2 (grasp absorbs pre-shape and the lift, place absorbs the retreat, a retreat move followed by
    back_to_origin becomes home), subgoal grouping and checker labels with an "unknown" level.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import traceback
from pathlib import Path

import numpy as np

import record as R1  # noqa: E402  (sets up RoboTwin paths)
from record import Recorder, wrist_to_tcp, r1, grasp_part, parts_of, find_target, rename_thin  # noqa: E402
from record import SimEnv, UnStableError, generate_episode_descriptions, ArmTag  # noqa: E402
import common_v2 as C  # noqa: E402

log = logging.getLogger("record_v2")
CAM_KEYS = {"agent": "agent_camera", "top": "top_camera"}
OFFSET_AXIS = 2          # grasp-frame axis along which a perturbed grasp is shifted (set by --offset_axis)


class RecorderV2(Recorder):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.failed_grasps = 0
        self.offset_axis = OFFSET_AXIS

    # ------------------------------------------------------------------ snapshot + visibility
    def snap(self, label):
        idx = super().snap(label)
        st = self.states[idx]
        cams = {}
        for k, name in CAM_KEYS.items():
            cam = self.sim.rd._extra_cameras.get(name)
            if cam is not None:
                cams[k] = cam
        try:
            cams["left"] = self.T.cameras.left_camera
            cams["right"] = self.T.cameras.right_camera
        except Exception:  # noqa: BLE001
            pass
        e2n = self.sim.entity2name
        vis = {}
        for k, cam in cams.items():
            try:
                seg = cam.get_picture("Segmentation")[..., 1]
                ids, cnt = np.unique(seg, return_counts=True)
                d = {}
                for i, c in zip(ids.tolist(), cnt.tolist()):
                    n = e2n.get(int(i))
                    if n:
                        d[n] = d.get(n, 0) + int(c)
                vis[k] = d
            except Exception as exc:  # noqa: BLE001
                log.debug("segmentation %s failed: %s", k, exc)
        st["visible"] = vis
        if idx == 0 and "agent" in cams:             # measured colour of each task object (for instruction checks)
            try:
                seg = cams["agent"].get_picture("Segmentation")[..., 1]
                rgb = np.clip(cams["agent"].get_picture("Color")[..., :3], 0, 1)
                cols = {}
                for i in np.unique(seg).tolist():
                    n = e2n.get(int(i))
                    if n:
                        px = rgb[seg == i]
                        if len(px) >= 30:
                            cols[n] = {"rgb": [round(float(x), 3) for x in np.median(px, 0)], "pixels": int(len(px))}
                for n, d in cols.items():
                    d["name"] = color_name(d["rgb"])
                st["colors"] = cols
            except Exception as exc:  # noqa: BLE001
                log.warning("colour measurement failed: %s", exc)
        return idx

    # ------------------------------------------------------------------ one expert call (v2 perturbation)
    def exec_one(self, it, recovery=False):
        arm = str(it[0])
        acts = list(it[1])
        m = self.meta.get(id(it[1]), {"api": "unknown", "args": {}})
        api, args = m["api"], m["args"]
        if not self.states:
            self.snap("start")
        ev = {"api": api, "arm": arm, "before": len(self.states) - 1, "mid": None, "recovery": recovery}
        sim, tz = self.sim, self.sim.table_z
        if api == "grasp_actor":
            ev["object"] = self.name_of.get(id(args.get("actor")), "?")
            moves = [a for a in acts if a.action == "move"]
            g = np.asarray(moves[-1].target_pose, float)
            ev["grasp_tcp"] = r1(np.r_[wrist_to_tcp(g)[:2] * 100, (wrist_to_tcp(g)[2] - tz) * 100])
            ev["contact_point_id"] = args.get("contact_point_id")
            ev["grasp_args"] = {k: args.get(k) for k in ("pre_grasp_dis", "grasp_dis", "gripper_pos", "contact_point_id")}
            if self.perturb and not self.perturbed and not recovery:
                import transforms3d as t3d
                Rm = t3d.quaternions.quat2mat(g[3:])
                side = Rm[:, self.offset_axis].copy()
                side[2] = 0.0
                if np.linalg.norm(side) < 0.3:          # side axis nearly vertical: use horizontal normal of approach
                    side = np.cross(Rm[:, 0], [0, 0, 1.0])
                side /= np.linalg.norm(side)
                dist = self.rng.uniform(0.055, 0.07) * (1 if self.rng.rand() < 0.5 else -1)
                d = side * dist
                for a in moves:
                    tp = np.asarray(a.target_pose, float).copy()
                    tp[:3] += d
                    a.target_pose = tp
                self.perturbed = True
                ev["perturbation"] = {"type": "grasp_offset", "offset_cm": r1(d * 100)}
        elif api == "place_actor":
            ev["object"] = self.name_of.get(id(args.get("actor")), "?")
            tp = args.get("target_pose")
            tp = np.asarray(tp.p if hasattr(tp, "p") else tp, float)[:3]
            ev["target_point"] = r1([tp[0] * 100, tp[1] * 100, (tp[2] - tz) * 100])
            ev["target_point_world_z"] = round(float(tp[2]) * 100, 1)
            ev["release"] = bool(args.get("is_open", True))
        elif api in ("move_by_displacement", "move_to_pose"):
            cur = self._arm_wrist(arm)
            tgt = np.asarray(acts[0].target_pose, float)
            d = (wrist_to_tcp(tgt) - wrist_to_tcp(cur)) * 100
            ev["delta"] = [int(round(x)) for x in d]
            ev["target_tcp"] = r1(np.r_[wrist_to_tcp(tgt)[:2] * 100, (wrist_to_tcp(tgt)[2] - tz) * 100])
            ev["move_axis"] = args.get("move_axis", "world")
        elif api in ("close_gripper", "open_gripper"):
            ev["pos"] = float(args.get("pos", 0.0 if api == "close_gripper" else 1.0))

        n_moves = sum(1 for a in acts if a.action == "move")
        if len(acts) >= 2 and n_moves >= 1:
            self.orig_move((it[0], acts[:1]))
            ev["mid"] = self.snap(f"{api}_mid")
            if self.T.plan_success:
                if api in ("grasp_actor", "place_actor") and len(acts) >= 3:
                    self.orig_move((it[0], acts[1:-1]))      # reach the grasp / place pose
                    ev["reached"] = self.snap(f"{api}_reached")
                    if self.T.plan_success:
                        self.orig_move((it[0], acts[-1:]))   # close / open
                else:
                    self.orig_move((it[0], acts[1:]))
        else:
            self.orig_move((it[0], acts))
        ev["plan_success"] = bool(self.T.plan_success)
        ev["after"] = self.snap(api)
        self.events.append(ev)

        if api == "grasp_actor" and self.states[-1]["held"].get(arm) != ev["object"]:
            self.failed_grasps += 1
            if ev.get("perturbation") and self.T.plan_success:
                g0 = self.states[ev["before"]]["arms"][arm]["gripper_cmd"]
                self.exec_one(self.T.open_gripper(ArmTag(arm), pos=max(g0, 0.6) if g0 < 0.99 else 1.0), recovery=True)
                self.exec_one(self.T.move_by_displacement(ArmTag(arm), z=0.05), recovery=True)
                self.events.append({"api": "locate", "arm": arm, "object": ev["object"], "before": len(self.states) - 1,
                                    "after": len(self.states) - 1, "mid": None, "recovery": True, "plan_success": True})
                ga = ev["grasp_args"]
                self.exec_one(self.T.grasp_actor(sim.names[ev["object"]], ArmTag(arm),
                                                 pre_grasp_dis=ga.get("pre_grasp_dis") or 0.1,
                                                 grasp_dis=ga.get("grasp_dis") or 0,
                                                 gripper_pos=ga.get("gripper_pos") or 0.0,
                                                 contact_point_id=ga.get("contact_point_id")), recovery=True)


def color_name(rgb):
    import colorsys
    r, g, b = rgb
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    if sat < 0.25 or max(rgb) - min(rgb) < 0.12:
        return "white" if l > 0.8 else ("black" if l < 0.15 else "gray")
    hd = h * 360
    for lim, name in ((15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (195, "cyan"), (255, "blue"),
                      (290, "purple"), (340, "pink"), (361, "red")):
        if hd < lim:
            return name
    return "red"


# ============================================================================ translation v2
def fb(S, ok, stage, note, extra=None):
    d = {"ok": bool(ok), "stage": stage, "note": note,
         "holding": {"left": S["held"]["left"], "right": S["held"]["right"]},
         "tcp": {"left": S["arms"]["left"]["tcp"], "right": S["arms"]["right"]["tcp"]}}
    if extra:
        d.update(extra)
    return d


def locate_call(S, idx, obj, recovery=False):
    note, res = R1.locate_result(S[idx], obj)
    return {"tool": "locate", "args": {"object": obj}, "before": idx, "after": idx, "mid": None, "expert_api": [],
            "synthetic": True, "recovery": recovery, "feedback": fb(S[idx], True, "done", note, {"object": res})}


def semantic_target(B, obj, tp, release):
    tgt = find_target(B, obj, tp)
    if tgt:
        return tgt, "on"
    if not release or tp[2] > 6:
        return "handover_point", "at"
    if abs(tp[0]) < 4 and abs(tp[1] + 13) < 4:
        return "table_center", "on"
    return "table_free_spot", "on"


def is_up(e):
    d = e.get("delta") or [0, 0, 0]
    return d[2] >= 3 or e.get("move_axis") == "arm"


def translate(states, events):
    S = states
    evs = list(events)
    calls, located, i = [], set(), 0
    while i < len(evs):
        e = evs[i]
        api, arm = e["api"], e["arm"]
        nxt = evs[i + 1] if i + 1 < len(evs) else None
        base = {"recovery": e.get("recovery", False), "mid": e.get("mid"), "perturbation": e.get("perturbation")}
        if api == "close_gripper" and 0 < e.get("pos", 0) < 1 and nxt and nxt["api"] == "grasp_actor" and nxt["arm"] == arm:
            e2 = dict(nxt)
            e2["before"] = e["before"]
            e2["preshape"] = e["pos"]
            evs[i + 1] = e2
            i += 1
            continue
        if api == "grasp_actor":
            obj = e["object"]
            if obj not in located:
                calls.append(locate_call(S, e["before"], obj, e.get("recovery", False)))
                located.add(obj)
            closed = S[e["after"]]
            held = closed["held"].get(arm) == obj
            lift = None
            if held and nxt and nxt["api"] == "move_by_displacement" and nxt["arm"] == arm and is_up(nxt):
                lift = nxt
            args = {"arm": arm, "object": obj}
            part = grasp_part(S[e["before"]], obj, e["grasp_tcp"])
            if part:
                args["part"] = part
            after = S[lift["after"]] if lift else closed
            lifted = after["objects"][obj]["lo"][2] - S[e["before"]]["objects"][obj]["lo"][2]
            if held and not lift:
                args["lift_cm"] = 0
            reached = e.get("reached") is not None and e.get("plan_success", True)
            if not held:
                touching = [o for a, o in closed["contacts"] if a == arm]
                stage = "closed" if reached else "approach"
                note = (f"approached and closed, but only one finger touches {obj}: not held; lift skipped" if obj in touching
                        else f"approached and closed on nothing: {obj} is not between the fingers; lift skipped")
                ok = False
            elif lift:
                stage, ok = "lifted", True
                note = f"grasped {obj} with the {arm} gripper and lifted it {lifted:.0f} cm"
            else:
                stage, ok = "closed", True
                note = f"grasped {obj} with the {arm} gripper (no lift requested)"
            c = dict(base, tool="grasp", args=args, before=e["before"], after=(lift or e)["after"],
                     closed=e["after"], reached=e.get("reached"),
                     expert_api=(["close_gripper"] if e.get("preshape") else []) + ["grasp_actor"] + (["move_by_displacement"] if lift else []),
                     feedback=fb(after, ok, stage, note, {"lifted_cm": round(max(lifted, 0.0), 1)}))
            calls.append(c)
            i += 2 if lift else 1
            continue
        if api == "place_actor":
            obj = e["object"]
            B = S[e["before"]]
            tp = e["target_point"]
            tgt, rel = semantic_target(B, obj, tp, e["release"])
            retreat = None
            if e["release"] and nxt and nxt["api"] == "move_by_displacement" and nxt["arm"] == arm:
                retreat = nxt
            args = {"arm": arm, "object": obj, "target": tgt}
            if rel != "on":
                args["relation"] = rel
            if not e["release"]:
                args["release"] = False
            if not retreat:
                args["retreat"] = False
            A = S[(retreat or e)["after"]]
            rel_expr = f"on({obj}, {tgt})" if tgt not in ("handover_point", "table_center", "table_free_spot") else (
                f"on({obj}, table)" if tgt != "handover_point" else f"holding({arm}, {obj})")
            if tgt == "table_center":
                rel_expr2 = f"at({obj}, 0, -13)"
            else:
                rel_expr2 = None
            rv = C.evaluate(rel_expr, S[e["after"]], S[0])[0] and (rel_expr2 is None or C.evaluate(rel_expr2, S[e["after"]], S[0])[0])
            if e["release"]:
                ok = bool(rv and A["arms"][arm]["gripper_cmd"] > 0.8)
                stage = "retreated" if retreat else "released"
                where = tgt if tgt not in ("table_center", "table_free_spot") else tgt.replace("_", " ")
                note = (f"put {obj} on {where}, released it" + (" and backed off" if retreat else "")) if ok else \
                       f"released {obj} but it is not resting on {where}"
            else:
                ok = A["held"].get(arm) == obj
                stage = "reached"
                note = f"{obj} is at the {tgt.replace('_', ' ')} and still held" if ok else f"{obj} was dropped on the way"
            extra = {"relation": f"{rel_expr if tgt != 'handover_point' else 'at(' + obj + ', handover_point)'} = {str(bool(rv)).lower()}"}
            c = dict(base, tool="place", args=args, before=e["before"], after=(retreat or e)["after"], reached=e.get("reached"),
                     released=e["after"] if e["release"] else None,
                     expert_api=["place_actor"] + (["move_by_displacement"] if retreat else []),
                     target_point=tp, feedback=fb(A, ok, stage, note, extra))
            calls.append(c)
            i += 2 if retreat else 1
            continue
        if api in ("move_by_displacement", "move_to_pose"):
            if nxt and nxt["api"] == "back_to_origin" and nxt["arm"] == arm and not e.get("recovery"):
                A = S[nxt["after"]]
                calls.append(dict(base, tool="home", args={"arm": arm}, before=e["before"], after=nxt["after"],
                                  expert_api=["move_by_displacement", "back_to_origin"],
                                  feedback=fb(A, nxt.get("plan_success", True), "done", f"{arm} arm backed off and is at its start pose")))
                i += 2
                continue
            d = e["delta"]
            A = S[e["after"]]
            err = float(np.linalg.norm(np.asarray(A["arms"][arm]["tcp"]) - np.asarray(e["target_tcp"])))
            ok = err < 1.5 and e.get("plan_success", True)
            h = A["held"].get(arm)
            note = (f"moved the {arm} fingertips by ({d[0]:+d}, {d[1]:+d}, {d[2]:+d}) cm" if ok else
                    f"stopped {err:.1f} cm short of the target") + (f"; holding {h}" if h else "")
            calls.append(dict(base, tool="move", args={"arm": arm, "dx": d[0], "dy": d[1], "dz": d[2]}, before=e["before"],
                              after=e["after"], expert_api=[api], feedback=fb(A, ok, "done", note)))
            i += 1
            continue
        if api == "back_to_origin":
            A = S[e["after"]]
            calls.append(dict(base, tool="home", args={"arm": arm}, before=e["before"], after=e["after"],
                              expert_api=[api], feedback=fb(A, e.get("plan_success", True), "done", f"{arm} arm is at its start pose")))
            i += 1
            continue
        if api in ("close_gripper", "open_gripper"):
            B, A = S[e["before"]], S[e["after"]]
            action = "open" if api == "open_gripper" else "close"
            if action == "open":
                lost = B["held"].get(arm)
                note = f"{arm} gripper opened" + (f"; let go of {lost}" if lost else "")
            else:
                h = A["held"].get(arm)
                note = f"{arm} gripper closed" + (f" on {h}" if h else ": nothing held")
            calls.append(dict(base, tool="gripper", args={"arm": arm, "action": action}, before=e["before"], after=e["after"],
                              expert_api=[api], feedback=fb(A, True, "done", note)))
            i += 1
            continue
        if api == "locate":
            if e["object"] in located and not e.get("recovery"):
                i += 1
                continue
            calls.append(locate_call(S, e["before"], e["object"], e.get("recovery", False)))
            located.add(e["object"])
            i += 1
            continue
        calls.append(dict(base, tool="unknown", args={"api": api}, before=e["before"], after=e["after"], expert_api=[api],
                          feedback=fb(S[e["after"]], False, "error", "unknown")))
        i += 1
    return calls


def group(calls, S):
    KEY = ("grasp", "place", "gripper")
    groups, cur = [], []

    def has_key(g):
        return any(c["tool"] in KEY for c in g)
    for c in calls:
        if c.get("recovery"):
            cur.append(c)
        elif c["tool"] == "home" and cur and cur[-1]["tool"] == "gripper" and cur[-1]["args"]["arm"] == c["args"]["arm"]:
            cur.append(c)                       # the releasing arm backs off: part of the hand-over
        elif c["tool"] in ("locate", "home"):
            if has_key(cur):
                groups.append(cur)
                cur = []
            cur.append(c)
        elif c["tool"] in KEY:
            if has_key(cur):
                groups.append(cur)
                cur = []
            cur.append(c)
        else:
            cur.append(c)
    if cur:
        groups.append(cur)
    merged = []
    for g in groups:
        if merged:
            pk = [c for c in merged[-1] if c["tool"] in KEY]
            gk = [c for c in g if c["tool"] in KEY]
            if pk and gk and pk[-1]["tool"] == "grasp" and gk[0]["tool"] == "gripper" and gk[0]["args"]["action"] == "open" \
                    and S[pk[-1]["before"]]["held"].get(gk[0]["args"]["arm"]) == pk[-1]["args"]["object"]:
                merged[-1] = merged[-1] + g
                continue
        merged.append(g)
    return [describe(g, S) for g in merged]


def describe(g, S):
    keys = [c for c in g if c["tool"] in ("grasp", "place", "gripper")]
    arms = sorted({c["args"]["arm"] for c in g if "arm" in c["args"]})
    arm = arms[0] if len(arms) == 1 else "both"
    k = keys[-1] if keys else g[-1]
    if len(keys) >= 2 and keys[-1]["tool"] == "gripper" and keys[0]["tool"] == "grasp" and keys[0]["args"]["arm"] != keys[-1]["args"]["arm"]:
        obj, b, a = keys[0]["args"]["object"], keys[0]["args"]["arm"], keys[-1]["args"]["arm"]
        goal, crit, kind = f"hand the {obj} over from the {a} arm to the {b} arm", [f"holding({b}, {obj})", f"not holding({a}, {obj})"], "handover"
    elif k["tool"] == "grasp":
        obj, a = k["args"]["object"], k["args"]["arm"]
        if k["args"].get("lift_cm") == 0:
            goal, crit = f"grasp the {obj} with the {a} arm", [f"holding({a}, {obj})"]
        else:
            goal, crit = f"pick up the {obj} with the {a} arm", [f"holding({a}, {obj})", f"lifted({obj}, 5cm)"]
        kind = "pick"
    elif k["tool"] == "place":
        obj, a, tgt = k["args"]["object"], k["args"]["arm"], k["args"]["target"]
        if k["args"].get("release") is False:
            goal = f"bring the {obj} with the {a} arm to the hand-over point above the table centre and keep holding it"
            crit, kind = [f"holding({a}, {obj})", f"at({obj}, 0, 0)"], "carry"
        elif tgt == "table_center":
            goal = f"put the {obj} down at the stacking spot in front of the robot's centre with the {a} arm and release it"
            crit, kind = [f"on({obj}, table)", f"at({obj}, 0, -13)", f"gripper_open({a})"], "place"
        else:
            goal = f"place the {obj} on the {tgt} with the {a} arm and release it"
            crit, kind = [f"on({obj}, {tgt})", f"gripper_open({a})"], "place"
    elif k["tool"] == "gripper":
        a = k["args"]["arm"]
        goal = f"{k['args']['action']} the {a} gripper"
        crit, kind = [f"gripper_open({a})" if k["args"]["action"] == "open" else f"gripper_closed({a})"], "gripper"
    else:
        goal, crit, kind = "move the arm", [], "move"
    return {"goal": goal, "arm": arm, "success": crit, "kind": kind, "calls": g, "start": g[0]["before"], "end": g[-1]["after"]}


def pick_mid(sg):
    bad = (sg["end"], sg["start"])
    cands = []
    for c in sg["calls"]:
        for key in ("mid", "reached", "closed"):
            if c.get(key) is not None and c[key] not in bad:
                cands.append(c[key])
    if not cands:
        cands = [c["after"] for c in sg["calls"][:-1] if c["after"] not in bad]
    cands = sorted(set(cands))
    return cands[len(cands) // 2] if cands else None


def checker_cams(arm):
    return ["agent", "top"] + (["left", "right"] if arm == "both" else [arm])


def checker_labels(subs, S):
    out = []
    for k, sg in enumerate(subs):
        for which, idx in (("end", sg["end"]), ("mid", pick_mid(sg))):
            if idx is None:
                continue
            res, lab = C.check_label(sg["success"], S[idx], S[0], checker_cams(sg["arm"]))
            out.append({"subgoal_index": k, "sample": which, "before": sg["start"], "after": idx, "criteria": res, "subgoal": lab})
    return out


def postprocess(states, events):
    for st in states:                       # fingers still touching right after opening do not count as holding
        for arm in ("left", "right"):
            if st["held"].get(arm) and st["arms"][arm]["gripper_cmd"] > 0.8:
                st["held"][arm] = None
    calls = translate(states, events)
    subs = group(calls, states)
    checks = checker_labels(subs, states)
    sub_json = [{k: v for k, v in s.items() if k != "calls"} | {"call_idx": [next(i for i, x in enumerate(calls) if x is c) for c in s["calls"]]}
                for s in subs]
    canon = [all(r["status"] != "unsatisfied" for c in checks if c["subgoal_index"] == k and c["sample"] == "end" for r in c["criteria"])
             for k in range(len(subs))]
    return calls, sub_json, checks, canon


# ============================================================================ main loop
def run(task, seeds, target, out, perturb, worker):
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    sim = SimEnv(task, seed_dir=str(out / "seedcache"), task_config="demo_randomized", seed_base_index=-1)
    rd, T = sim.rd, sim.rd.env
    n_ok = 0
    stats = {"task": task, "worker": worker, "perturb": perturb, "tried": [], "ok": [], "failed": [], "unstable": [],
             "perturb_no_failure": []}
    rng = np.random.RandomState(4321 + worker)
    for seed in seeds:
        if n_ok >= target:
            break
        t0 = time.time()
        stats["tried"].append(seed)
        try:
            T.setup_demo(now_ep_num=0, seed=seed, is_test=True, **rd.args)
        except UnStableError as exc:
            stats["unstable"].append(seed)
            log.info("seed %d unstable: %s", seed, exc)
            try:
                T.close_env()
            except Exception:  # noqa: BLE001
                pass
            continue
        rd._episode_open = True
        sim.T = T
        T.eval_success = False
        T.plan_success = True
        rd._create_agent_camera()
        sim._fingers()
        sim._build_names()
        rename_thin(sim)
        rec = RecorderV2(sim, out, task, seed, perturb, rng)
        rec.install()
        err, info = None, {}
        try:
            rec.snap("start")
            ret = T.play_once()
            info = ret.get("info", {}) if isinstance(ret, dict) else {}
            last = rec.events[-1] if rec.events else None
            if T.plan_success and last and last["api"] == "place_actor" and last.get("release"):
                rec.exec_one(T.move_by_displacement(ArmTag(last["arm"]), z=0.05, move_axis="arm"))   # tool default retreat
                rec.events[-1]["tool_default_retreat"] = True
            ok = bool(T.plan_success and T.check_success())
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"{type(exc).__name__}: {exc}"
            log.warning("seed %d expert raised: %s\n%s", seed, err, traceback.format_exc())
        rec.uninstall()
        try:
            res = generate_episode_descriptions(task, [info], 1)
            cands = [str(x) for x in res[0]["unseen"]]
            instruction = str(np.random.RandomState(seed).choice(res[0]["unseen"]))
        except Exception as exc:  # noqa: BLE001
            cands, instruction = [], task.replace("_", " ")
            log.warning("instruction failed: %s", exc)
        try:
            calls, subs, checks, canon = postprocess(rec.states, rec.events)
        except Exception as exc:  # noqa: BLE001
            log.warning("post-processing failed: %s\n%s", exc, traceback.format_exc())
            calls, subs, checks, canon = [], [], [], []
            err = (err or "") + f" | post: {exc}"
        keep = ok and (not perturb or rec.failed_grasps > 0)
        ep = {"episode_id": rec.ep_id, "version": "v2", "task": task, "seed": seed,
              "seed_note": "RoboTwin collection seed (RoboDawn seed_base_index=-1); eval seeds are >=100000",
              "task_config": "demo_randomized", "instruction": instruction, "instruction_candidates": cands,
              "expert_info": info, "expert_success": ok, "keep": keep, "failed_grasps": rec.failed_grasps,
              "error": err, "perturbed": perturb, "objects": list(sim.names), "states": rec.states, "events": rec.events,
              "calls": calls, "subgoals": subs, "checker": checks, "canonical_criteria_hold_at_end": canon,
              "seconds": round(time.time() - t0, 1)}
        (out / "episodes" / f"{rec.ep_id}.json").write_text(json.dumps(ep, ensure_ascii=False, default=str))
        if ok and perturb and not rec.failed_grasps:
            stats["perturb_no_failure"].append(seed)
        (stats["ok"] if ok else stats["failed"]).append(seed)
        if keep:
            n_ok += 1
        log.info("%s seed %d ok=%s keep=%s failed_grasps=%d subgoals=%d calls=%d %.0fs canon=%s", task, seed, ok, keep,
                 rec.failed_grasps, len(subs), len(calls), time.time() - t0, canon)
        rd.close_episode()
        (out / f"stats_{task}_w{worker}.json").write_text(json.dumps(stats))
    rd.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_seed", type=int, default=5000)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--offset_axis", type=int, default=2)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--out", default="runs/oracle_expert_pilot_v2")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("curobo").setLevel(logging.WARNING)
    OFFSET_AXIS = a.offset_axis
    run(a.task, list(range(a.start, a.max_seed, a.step)), a.n, Path(a.out), a.perturb, a.worker)
