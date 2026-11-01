#!/usr/bin/env python
"""Oracle-expert recorder.

Replays RoboTwin's scripted expert (play_once) with its high-level API wrapped (no RoboTwin source change):
grasp_actor / place_actor / move_by_displacement / close_gripper / open_gripper / back_to_origin / move_to_pose
tag the action lists they return, and Base_Task.move is replaced by a recorder that executes each tagged call on its
own (parallel arm calls are run one after the other, home first), snapshots 4 cameras + simulator truth before, in the
middle (after the first motion of multi-motion calls) and after, and stores the call translated into our generic tool
vocabulary with the tool feedback computed from truth.  Afterwards the calls are grouped into subgoals with canonical
criteria, and checker labels are computed from truth.

Seeds: RoboTwin data-collection seeds 0,1,2,... (RoboDawn seed_base_index=-1); RoboDawn evaluation uses 100000+.
The instruction is sampled exactly like RoboDawn's reset_episode (unseen split, RandomState(seed)).
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OH))
from oh.simenv import SimEnv  # noqa: E402  (imports RoboDawn env -> chdir to RoboTwin root)
from harness.robotwin.env import wrist_to_tcp, annotate_head_image, enhance_image  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402
import sapien  # noqa: E402
from envs.utils import ArmTag  # noqa: E402
from PIL import Image  # noqa: E402

import common  # noqa: E402

log = logging.getLogger("record")
API = ("grasp_actor", "place_actor", "move_by_displacement", "close_gripper", "open_gripper", "back_to_origin",
       "move_to_pose")
CAMS = ("agent", "top", "left", "right")


def r1(v):
    return [round(float(x), 1) for x in v]


class Recorder:
    def __init__(self, sim: SimEnv, out: Path, task: str, seed: int, perturb: bool, rng):
        self.sim, self.T = sim, sim.T
        self.out, self.task, self.seed = out, task, seed
        self.ep_id = f"{task}__s{seed}" + ("__perturb" if perturb else "")
        self.fdir = out / "frames" / self.ep_id
        self.fdir.mkdir(parents=True, exist_ok=True)
        self.perturb, self.perturbed = perturb, False
        self.rng = rng
        self.states: list[dict] = []
        self.events: list[dict] = []
        self.meta: dict = {}
        self.name_of = {id(v): n for n, v in sim.names.items()}

    # ------------------------------------------------------------------ truth + frames
    def snap(self, label: str) -> int:
        sim, T = self.sim, self.T
        idx = len(self.states)
        obs = sim.rd.observe()
        imgs = {}
        try:
            imgs["agent"] = annotate_head_image(sim.rd, obs, camera="agent_camera")
            imgs["top"] = annotate_head_image(sim.rd, obs, camera="top_camera")
        except Exception as exc:  # noqa: BLE001
            log.warning("annotate failed: %s", exc)
        for arm in ("left", "right"):
            k = f"{arm}_camera"
            if k in obs["images"]:
                imgs[arm] = enhance_image(Image.fromarray(obs["images"][k]))
        files = {}
        for k, im in imgs.items():
            fn = f"{idx:03d}_{k}.jpg"
            im.convert("RGB").save(self.fdir / fn, quality=85)
            files[k] = f"frames/{self.ep_id}/{fn}"
        tz = sim.table_z
        objs = {}
        for n in sim.names:
            lo, hi = sim.aabb(n)
            lo = (lo - [0, 0, tz]) * 100
            hi = (hi - [0, 0, tz]) * 100
            q = np.asarray(sim.names[n].get_pose().q)
            import transforms3d as t3d
            yaw = math.degrees(t3d.euler.quat2euler(q, axes="sxyz")[2])
            objs[n] = {"center": r1((lo + hi) / 2), "size": r1(hi - lo), "lo": r1(lo), "hi": r1(hi), "yaw": round(yaw, 1)}
        arms = {}
        for arm in ("left", "right"):
            tcp = sim.tcp(arm)
            arms[arm] = {"tcp": r1([tcp[0] * 100, tcp[1] * 100, (tcp[2] - tz) * 100]),
                         "gripper_cmd": round(sim.gripper_cmd(arm), 2), "opening": round(sim.opening(arm), 2)}
        held = sim.held()
        contacts = sorted({(a, o) for a, o, _ in sim.contacts()})
        st = {"idx": idx, "label": label, "frames": files, "objects": objs, "arms": arms,
              "held": {"left": held.get("left"), "right": held.get("right")}, "contacts": [list(c) for c in contacts],
              "physics_steps": sim.physics_steps}
        self.states.append(st)
        return idx

    # ------------------------------------------------------------------ hooks
    def install(self):
        T = self.T
        self.orig = {}
        for name in API:
            orig = getattr(T, name)
            self.orig[name] = orig
            sig = inspect.signature(orig)

            def wrapped(*a, _orig_=orig, _nm_=name, _sg_=sig, **kw):
                out = _orig_(*a, **kw)
                try:
                    b = _sg_.bind(*a, **kw)
                    b.apply_defaults()
                    args = dict(b.arguments)
                except TypeError:
                    args = {}
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], list):
                    self.meta[id(out[1])] = {"api": _nm_, "args": args, "acts": out[1]}
                return out
            setattr(T, name, wrapped)
        self.orig_move = T.move
        T.move = self.move

    def uninstall(self):
        for name in API:
            try:
                delattr(self.T, name)
            except AttributeError:
                pass
        try:
            delattr(self.T, "move")
        except AttributeError:
            pass

    def move(self, a1, a2=None, save_freq=-1):
        items = [x for x in (a1, a2) if x is not None and x[0] is not None and x[1]]
        items.sort(key=lambda x: 0 if self.meta.get(id(x[1]), {}).get("api") == "back_to_origin" else 1)
        for it in items:
            if self.T.plan_success is False:
                return False
            self.exec_one(it)
        return bool(self.T.plan_success)

    # ------------------------------------------------------------------ one expert call
    def _arm_wrist(self, arm):
        return np.asarray(self.T.get_arm_pose(arm), float)

    def exec_one(self, it, recovery=False):
        arm = str(it[0])
        acts = list(it[1])
        m = self.meta.get(id(it[1]), {"api": "unknown", "args": {}})
        api, args = m["api"], m["args"]
        if not self.states:
            self.snap("start")
        ev = {"api": api, "arm": arm, "before": len(self.states) - 1, "mid": None, "recovery": recovery}
        sim = self.sim
        tz = sim.table_z
        if api == "grasp_actor":
            ev["object"] = self.name_of.get(id(args.get("actor")), "?")
            moves = [a for a in acts if a.action == "move"]
            g = np.asarray(moves[-1].target_pose, float)
            ev["grasp_tcp"] = r1(np.r_[wrist_to_tcp(g)[:2] * 100, (wrist_to_tcp(g)[2] - tz) * 100])
            ev["contact_point_id"] = args.get("contact_point_id")
        elif api == "place_actor":
            ev["object"] = self.name_of.get(id(args.get("actor")), "?")
            tp = args.get("target_pose")
            tp = np.asarray(tp.p if hasattr(tp, "p") else tp, float)[:3]
            ev["target_point"] = r1([tp[0] * 100, tp[1] * 100, (tp[2] - tz) * 100])
            ev["release"] = bool(args.get("is_open", True))
        elif api in ("move_by_displacement", "move_to_pose"):
            cur = self._arm_wrist(arm)
            tgt = np.asarray(acts[0].target_pose, float)
            d = (wrist_to_tcp(tgt) - wrist_to_tcp(cur)) * 100
            ev["delta"] = [int(round(x)) for x in d]
            ev["target_tcp"] = r1(np.r_[wrist_to_tcp(tgt)[:2] * 100, (wrist_to_tcp(tgt)[2] - tz) * 100])
        elif api in ("close_gripper", "open_gripper"):
            ev["pos"] = float(args.get("pos", 0.0 if api == "close_gripper" else 1.0))

        # ---- perturbation: object slides after the grasp pose was computed (stale perception)
        if self.perturb and not self.perturbed and api == "grasp_actor" and not recovery:
            obj = sim.names[ev["object"]]
            p = obj.get_pose()
            ang = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(0.03, 0.05)
            d = np.array([math.cos(ang) * dist, math.sin(ang) * dist, 0.0])
            obj.actor.set_pose(sapien.Pose(np.asarray(p.p) + d, p.q))
            for _ in range(10):
                self.T.scene.step()
            self.perturbed = True
            ev["perturbation"] = {"object": ev["object"], "shift_cm": r1(d * 100)}
            ev["before_unperturbed"] = ev["before"]
            ev["before"] = self.snap("perturbed")

        self.T.plan_success = True if self.T.plan_success is None else self.T.plan_success
        n_moves = sum(1 for a in acts if a.action == "move")
        if len(acts) >= 2 and n_moves >= 1:
            self.orig_move((it[0], acts[:1]))
            ev["mid"] = self.snap(f"{api}_mid")
            if self.T.plan_success:
                self.orig_move((it[0], acts[1:]))
        else:
            self.orig_move((it[0], acts))
        ev["plan_success"] = bool(self.T.plan_success)
        ev["after"] = self.snap(api)
        self.events.append(ev)

        # ---- recovery after a missed (perturbed) grasp: open, back off, locate, grasp again
        if api == "grasp_actor" and not recovery and ev.get("perturbation") and \
                self.states[-1]["held"].get(arm) != ev["object"] and self.T.plan_success:
            g0 = self.states[ev["before"]]["arms"][arm]["gripper_cmd"]
            self.exec_one(self.T.open_gripper(ArmTag(arm), pos=max(g0, 0.6) if g0 < 0.99 else 1.0), recovery=True)
            self.exec_one(self.T.move_by_displacement(ArmTag(arm), z=0.05), recovery=True)
            self.events.append({"api": "locate", "arm": arm, "object": ev["object"], "before": len(self.states) - 1,
                                "after": len(self.states) - 1, "mid": None, "recovery": True, "plan_success": True})
            cp = ev.get("contact_point_id")
            pre = self.meta.get(id(it[1]), {}).get("args", {})
            self.exec_one(self.T.grasp_actor(sim.names[ev["object"]], ArmTag(arm),
                                             pre_grasp_dis=pre.get("pre_grasp_dis", 0.1),
                                             grasp_dis=pre.get("grasp_dis", 0),
                                             gripper_pos=pre.get("gripper_pos", 0.0),
                                             contact_point_id=cp), recovery=True)


# ============================================================================ translation to tool calls
def grasp_part(st, obj, tcp):
    o = st["objects"][obj]
    h = o["hi"][2] - o["lo"][2]
    if h < 12:
        return None
    f = (tcp[2] - o["lo"][2]) / max(h, 1e-6)
    return "upper half" if f >= 0.5 else "lower half"


def parts_of(st, obj):
    o = st["objects"][obj]
    return ["upper half", "lower half"] if o["hi"][2] - o["lo"][2] >= 12 else ["body"]


def find_target(st, obj, tp):
    """Which object (if any) the place target point lies on."""
    best = None
    for n, o in st["objects"].items():
        if n == obj:
            continue
        if o["lo"][0] - 2 <= tp[0] <= o["hi"][0] + 2 and o["lo"][1] - 2 <= tp[1] <= o["hi"][1] + 2 \
                and abs(tp[2] - o["hi"][2]) < 4:
            d = abs(tp[0] - o["center"][0]) + abs(tp[1] - o["center"][1])
            if best is None or d < best[0]:
                best = (d, n)
    return best[1] if best else None


def feedback(S, ok, note, extra=None):
    fb = {"ok": bool(ok), "note": note, "holding": {"left": S["held"]["left"], "right": S["held"]["right"]},
          "tcp": {"left": S["arms"]["left"]["tcp"], "right": S["arms"]["right"]["tcp"]}}
    if extra:
        fb.update(extra)
    return fb


def locate_result(S, obj):
    o = S["objects"][obj]
    res = {"name": obj, "center": [round(x) for x in o["center"]], "size": [round(x) for x in o["size"]],
           "top_z": round(o["hi"][2]), "parts": parts_of(S, obj)}
    note = (f"{obj} at ({res['center'][0]}, {res['center'][1]}), size {res['size'][0]}x{res['size'][1]}x{res['size'][2]} cm, "
            f"top at {res['top_z']} cm")
    return note, res


def translate(rec: Recorder) -> list[dict]:
    """events -> list of tool calls {tool, args, before, after, mid, feedback, ...}"""
    S = rec.states
    calls = []
    evs = rec.events
    i = 0
    located = set()
    while i < len(evs):
        e = evs[i]
        api, arm = e["api"], e["arm"]
        B, A = S[e["before"]], S[e["after"]]
        c = {"before": e["before"], "after": e["after"], "mid": e.get("mid"), "expert_api": [api],
             "recovery": e.get("recovery", False), "plan_success": e.get("plan_success", True)}
        if e.get("perturbation"):
            c["perturbation"] = e["perturbation"]
        # pre-shaping the gripper right before a grasp is part of grasp()
        if api == "close_gripper" and 0 < e.get("pos", 0) < 1 and i + 1 < len(evs) and \
                evs[i + 1]["api"] == "grasp_actor" and evs[i + 1]["arm"] == arm:
            nxt = evs[i + 1]
            if nxt["object"] not in located:
                note, res = locate_result(S[e["before"]], nxt["object"])
                calls.append({"tool": "locate", "args": {"object": nxt["object"]}, "before": e["before"],
                              "after": e["before"], "mid": None, "expert_api": [], "synthetic": True,
                              "feedback": feedback(S[e["before"]], True, note, {"object": res})})
                located.add(nxt["object"])
            i += 1
            e2 = dict(nxt)
            e2["before"] = e["before"]
            e2["preshape"] = e["pos"]
            evs = evs[:i] + [e2] + evs[i + 1:]
            continue
        if api == "grasp_actor":
            obj = e["object"]
            if obj not in located:
                b0 = e.get("before_unperturbed", e["before"])
                note, res = locate_result(S[b0], obj)
                calls.append({"tool": "locate", "args": {"object": obj}, "before": b0, "after": b0,
                              "mid": None, "expert_api": [], "synthetic": True,
                              "feedback": feedback(S[b0], True, note, {"object": res})})
                located.add(obj)
            part = grasp_part(B, obj, e["grasp_tcp"])
            args = {"arm": arm, "object": obj}
            if part:
                args["part"] = part
            held = A["held"].get(arm) == obj
            touching = [o for a, o in A["contacts"] if a == arm]
            if held:
                note = f"grasped {obj} with the {arm} gripper"
            elif obj in touching:
                note = f"fingers closed but only one finger touches {obj}: it is NOT held"
            else:
                note = f"gripper closed on nothing: {obj} is not between the fingers (it may have moved)"
            c.update(tool="grasp", args=args, feedback=feedback(A, held, note))
            if e.get("before_unperturbed") is not None:   # the decision was taken on the pre-shift view
                c["before"] = e["before_unperturbed"]
                c["perturbed_frame"] = e["before"]
            if e.get("preshape"):
                c["expert_api"] = ["close_gripper", "grasp_actor"]
        elif api == "place_actor":
            obj = e["object"]
            tp = e["target_point"]
            tgt = find_target(B, obj, tp)
            args = {"arm": arm, "object": obj}
            if tgt:
                args.update(target=tgt, relation="on")
            elif tp[2] < 4:
                args.update(target="table", relation="on", xy=[round(tp[0]), round(tp[1])])
            else:
                args.update(target="air", relation="at", xyz=[round(tp[0]), round(tp[1]), round(tp[2])])
            if not e["release"]:
                args["release"] = False
            if e["release"]:
                rel = f"on({obj}, {tgt})" if tgt else f"on({obj}, table)"
                v, _ = common.evaluate(rel, A, S[0])
                rel_ok = v and (tgt or common.evaluate(f"at({obj}, {tp[0]}, {tp[1]})", A, S[0])[0])
                ok = bool(rel_ok and A["arms"][arm]["gripper_cmd"] > 0.8)
                where = tgt if tgt else f"the table at ({round(tp[0])}, {round(tp[1])})"
                note = (f"placed {obj} on {where} and opened the gripper" if ok else
                        f"opened the gripper but {obj} is not resting on {where}")
            else:
                ok = A["held"].get(arm) == obj
                note = (f"{obj} brought to ({', '.join(str(round(x)) for x in tp)}) and still held" if ok
                        else f"{obj} was dropped on the way")
            c.update(tool="place", args=args, feedback=feedback(A, ok, note))
        elif api in ("move_by_displacement", "move_to_pose"):
            d = e["delta"]
            args = {"arm": arm, "dx": d[0], "dy": d[1], "dz": d[2]}
            err = float(np.linalg.norm(np.asarray(A["arms"][arm]["tcp"]) - np.asarray(e["target_tcp"])))
            ok = err < 1.5 and e.get("plan_success", True)
            h = A["held"].get(arm)
            note = (f"moved the {arm} fingertips by ({d[0]:+d}, {d[1]:+d}, {d[2]:+d}) cm" if ok else
                    f"stopped {err:.1f} cm short of the target") + (f"; still holding {h}" if h else "")
            c.update(tool="move", args=args, feedback=feedback(A, ok, note))
        elif api == "back_to_origin":
            c.update(tool="home", args={"arm": arm}, feedback=feedback(A, e.get("plan_success", True),
                                                                        f"{arm} arm is back at its start pose"))
        elif api in ("close_gripper", "open_gripper"):
            action = "open" if api == "open_gripper" else "close"
            if action == "open":
                lost = [o for o in [B["held"].get(arm)] if o and A["held"].get(arm) != o]
                note = f"{arm} gripper opened" + (f"; released {lost[0]}" if lost else "")
                ok = True
            else:
                h = A["held"].get(arm)
                note = f"{arm} gripper closed" + (f" on {h}" if h else ": nothing held")
                ok = True
            c.update(tool="gripper", args={"arm": arm, "action": action}, feedback=feedback(A, ok, note))
        elif api == "locate":
            note, res = locate_result(B, e["object"])
            c.update(tool="locate", args={"object": e["object"]}, feedback=feedback(B, True, note, {"object": res}),
                     synthetic=True)
            located.add(e["object"])
        else:
            c.update(tool="unknown", args={"api": api}, feedback=feedback(A, False, "unknown"))
        calls.append(c)
        i += 1
    return calls


# ============================================================================ subgoals
def group_subgoals(calls, S) -> list[dict]:
    KEY = ("grasp", "place", "gripper")
    groups, cur = [], []

    def has_key(g):
        return any(c["tool"] in KEY for c in g)
    for c in calls:
        if c.get("recovery"):                 # recovery steps stay in the subgoal of the failed attempt
            cur.append(c)
        elif c["tool"] in ("locate", "home"):
            if has_key(cur):
                groups.append(cur)
                cur = []
            cur.append(c)
        elif c["tool"] in KEY:
            if has_key(cur) and not c.get("recovery"):
                groups.append(cur)
                cur = []
            cur.append(c)
        else:
            cur.append(c)
    if cur:
        groups.append(cur)
    # a failed grasp stays in the same subgoal as its recovery (already handled by recovery flag above)
    # hand-over: grasp by arm B of an object held by arm A, then A opens -> one subgoal
    merged = []
    for g in groups:
        if merged:
            prev = merged[-1]
            pk = [c for c in prev if c["tool"] in KEY]
            gk = [c for c in g if c["tool"] in KEY]
            if pk and gk and pk[-1]["tool"] == "grasp" and gk[0]["tool"] == "gripper" and gk[0]["args"]["action"] == "open":
                obj = pk[-1]["args"]["object"]
                if S[pk[-1]["before"]]["held"].get(gk[0]["args"]["arm"]) == obj:
                    merged[-1] = prev + g
                    continue
        merged.append(g)
    subs = []
    for g in merged:
        subs.append(describe(g, S))
    return subs


def describe(g, S):
    keys = [c for c in g if c["tool"] in ("grasp", "place", "gripper")]
    arms = sorted({c["args"]["arm"] for c in g if "arm" in c["args"]})
    arm = arms[0] if len(arms) == 1 else "both"
    end = S[g[-1]["after"]]
    k = keys[-1] if keys else g[-1]
    crit = []
    if len(keys) >= 2 and keys[-1]["tool"] == "gripper" and keys[0]["tool"] == "grasp" and keys[0]["args"]["arm"] != keys[-1]["args"]["arm"]:
        obj, b, a = keys[0]["args"]["object"], keys[0]["args"]["arm"], keys[-1]["args"]["arm"]
        goal = f"hand the {obj} over from the {a} arm to the {b} arm"
        crit = [f"holding({b}, {obj})", f"not holding({a}, {obj})"]
        kind = "handover"
    elif k["tool"] == "grasp":
        obj, a = k["args"]["object"], k["args"]["arm"]
        lifted = [c for c in g[g.index(k) + 1:] if c["tool"] == "move" and c["args"]["dz"] >= 5]
        if lifted:
            goal = f"pick up the {obj} with the {a} arm"
            crit = [f"holding({a}, {obj})", f"lifted({obj}, 5cm)"]
        else:
            goal = f"grasp the {obj} with the {a} arm"
            crit = [f"holding({a}, {obj})"]
        kind = "pick"
    elif k["tool"] == "place":
        obj, a, tgt = k["args"]["object"], k["args"]["arm"], k["args"]["target"]
        if k["args"].get("release") is False:
            x, y, z = k["args"]["xyz"]
            goal = f"bring the {obj} with the {a} arm to the hand-over point above the table centre ({x}, {y}) and keep holding it"
            crit = [f"holding({a}, {obj})", f"at({obj}, {x}, {y})"]
            kind = "carry"
        elif tgt == "table":
            x, y = k["args"]["xy"]
            goal = f"put the {obj} down on the table at ({x}, {y}) with the {a} arm and release it"
            crit = [f"on({obj}, table)", f"at({obj}, {x}, {y})", f"gripper_open({a})"]
            kind = "place"
        else:
            goal = f"place the {obj} on the {tgt} with the {a} arm and release it"
            crit = [f"on({obj}, {tgt})", f"gripper_open({a})"]
            kind = "place"
    elif k["tool"] == "gripper":
        a = k["args"]["arm"]
        goal = f"{k['args']['action']} the {a} gripper"
        crit = [f"gripper_open({a})" if k["args"]["action"] == "open" else f"gripper_closed({a})"]
        kind = "gripper"
    else:
        goal, kind = "move the arm", "move"
        crit = []
    return {"goal": goal, "arm": arm, "success": crit, "kind": kind, "calls": g,
            "start": g[0]["before"], "end": g[-1]["after"]}


def checker_labels(subs, S):
    out = []
    for k, sg in enumerate(subs):
        for which, idx in (("end", sg["end"]), ("mid", pick_mid(sg))):
            if idx is None:
                continue
            res = []
            for cexpr in sg["success"]:
                v, ev = common.evaluate(cexpr, S[idx], S[0])
                res.append({"criterion": cexpr, "status": "satisfied" if v else "unsatisfied", "evidence": ev})
            out.append({"subgoal_index": k, "sample": which, "before": sg["start"], "after": idx,
                        "criteria": res,
                        "subgoal": "satisfied" if all(r["status"] == "satisfied" for r in res) else "unsatisfied"})
    return out


def pick_mid(sg):
    """A frame in the middle of the subgoal: prefer snapshots taken in the middle of a motion (e.g. at the pre-grasp
    or pre-place pose), else the state after a non-final call."""
    bad = (sg["end"], sg["start"])
    mids = [c["mid"] for c in sg["calls"] if c.get("mid") is not None and c["mid"] not in bad]
    if mids:
        return mids[len(mids) // 2]
    afters = [c["after"] for c in sg["calls"][:-1] if c["after"] not in bad]
    return afters[len(afters) // 2] if afters else None


# ============================================================================ main loop
def rename_thin(sim):
    """RoboTwin calls the flat target pad a 'box'; name flat things *_pad instead of *_block."""
    ren = {}
    for n in list(sim.names):
        lo, hi = sim.aabb(n)
        if n.endswith("_block") and (hi[2] - lo[2]) < 0.02:
            ren[n] = n[:-len("_block")] + "_pad"
    for old, new in ren.items():
        sim.names[new] = sim.names.pop(old)
        sim.attr_of[new] = sim.attr_of.pop(old)
        for k, v in list(sim.entity2name.items()):
            if v == old:
                sim.entity2name[k] = new


def run(task, seeds, target, out, perturb_every, worker):
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    sim = SimEnv(task, seed_dir=str(out / "seedcache"), task_config="demo_randomized", seed_base_index=-1)
    rd = sim.rd
    T = rd.env
    n_ok = 0
    stats = {"task": task, "worker": worker, "tried": [], "ok": [], "failed": [], "unstable": []}
    rng = np.random.RandomState(1234 + worker)
    for seed in seeds:
        if n_ok >= target:
            break
        perturb = perturb_every > 0
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
        rec = Recorder(sim, out, task, seed, perturb, rng)
        rec.install()
        err = None
        info = {}
        try:
            rec.snap("start")
            ret = T.play_once()
            info = ret.get("info", {}) if isinstance(ret, dict) else {}
            ok = bool(T.plan_success and T.check_success())
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"{type(exc).__name__}: {exc}"
            log.warning("seed %d expert raised: %s\n%s", seed, err, traceback.format_exc())
        rec.uninstall()
        try:
            res = generate_episode_descriptions(task, [info], 1)
            instruction = str(np.random.RandomState(seed).choice(res[0]["unseen"]))
        except Exception as exc:  # noqa: BLE001
            instruction = task.replace("_", " ")
            log.warning("instruction failed: %s", exc)
        try:
            calls = translate(rec)
            subs = group_subgoals(calls, rec.states)
            checks = checker_labels(subs, rec.states)
            canon_ok = [all(c["subgoal"] == "satisfied" for c in checks if c["subgoal_index"] == k and c["sample"] == "end")
                        for k in range(len(subs))]
        except Exception as exc:  # noqa: BLE001
            log.warning("post-processing failed: %s\n%s", exc, traceback.format_exc())
            calls, subs, checks, canon_ok = [], [], [], []
            err = (err or "") + f" | post: {exc}"
        ep = {"episode_id": rec.ep_id, "task": task, "seed": seed, "seed_note": "RoboTwin collection seed (RoboDawn seed_base_index=-1); eval seeds are >=100000",
              "task_config": "demo_randomized", "instruction": instruction, "expert_info": info,
              "expert_success": ok, "error": err, "perturbed": perturb,
              "objects": list(sim.names), "states": rec.states, "events": rec.events, "calls": calls,
              "subgoals": [{k: v for k, v in s.items() if k != "calls"} | {"call_idx": [next(i for i, x in enumerate(calls) if x is c) for c in s["calls"]]} for s in subs],
              "checker": checks, "canonical_criteria_hold_at_end": canon_ok,
              "seconds": round(time.time() - t0, 1)}
        (out / "episodes" / f"{rec.ep_id}.json").write_text(json.dumps(ep, ensure_ascii=False, default=str))
        (stats["ok"] if ok else stats["failed"]).append(seed)
        if ok:
            n_ok += 1
        log.info("%s seed %d ok=%s subgoals=%d calls=%d %.0fs canon=%s", task, seed, ok, len(subs), len(calls),
                 time.time() - t0, canon_ok)
        rd.close_episode()
        (out / f"stats_{task}_w{worker}.json").write_text(json.dumps(stats))
    rd.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_seed", type=int, default=400)
    ap.add_argument("--perturb_every", type=int, default=0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--out", default="runs/oracle_expert_pilot")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("curobo").setLevel(logging.WARNING)
    run(a.task, list(range(a.start, a.max_seed, a.step)), a.n, Path(a.out), a.perturb_every, a.worker)
