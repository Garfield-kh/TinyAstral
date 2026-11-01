#!/usr/bin/env python
"""Single-arm harness H* (Guava, LIBERO-PRO, and the zero-shot drivers): ONE planner loop + ONE semantic tool layer.
Only the adapter changes between backends (LIBERO-PRO on robosuite, Guava on MuJoCo, the zero-shot simulators).

  planner (any OpenAI-compatible VLM) -> UnifiedHarness.call (semantic tools: locate / grasp / place / carry / lift /
  move / push / press / articulate / handover / bimanual / release / home / inspect / twist / verify / done / give_up)
  -> tools.SkillTools (execution, hold checks, re-grasps, in-hand geometry, IK-checked candidates, support-surface
  placement, hinged doors) + geom (shape solvers) -> adapter (perception, motion, embodiment constants).

What an adapter declares (PROFILES below / adapter attributes), never task names:
  arms, display units, controller retry count, re-grasp variants, release back-off steps, finger thresholds, whether the
  view is cleared after letting go, and the tool manifest + prompt (embodiment description). Capabilities are detected
  on the adapter object (IK feasibility + shape candidates -> surface placement / candidate grasps; instance labelling;
  arbitrary gripper orientations).
SYSTEM prompts: prompts.py. Benchmark protocol (episode budget, cameras, image encoding, success audit, result files):
the drivers near the bottom of this file together with adapters/guava_protocol.py and adapters/libero_protocol.py.

--gen: the benchmark-agnostic tool-layer improvements of flags.py. "default" = GEN_DEFAULTS[bench] (the setting used for
all reported results), "none" = the plain UnifiedHarness, "all" = every flag, or a comma-separated list; $HARNESS_GEN
overrides the default. Enabled flags also append a paragraph to the SYSTEM prompt describing the extra tool behaviour, so
the flag set is part of the planner prompt: a student must be evaluated with the flag set its training data was
collected with.

Command line (through harness/run.py):
  run.py guava  [--gen G] --task T --seed-base S --episodes N --out DIR [--offset K] [--url U] [--sam3 U]
  run.py libero [--gen G] --task 57 --seed 2 3 --out DIR [--url U]
Library use (zero-shot drivers): set $HARNESS_BENCH (guava|libero) and put harness/ on sys.path
(paths.use("single_arm")) BEFORE importing this module, then use make_harness(bench, A, prof, flags),
run_loop(D, H, url, model, thinking, max_tokens) and system_text(bench, base, flags)."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # harness/: paths, tools, geom
import paths as _paths  # noqa: E402
_paths.use("single_arm")
P = _paths.ROOT

import geom as Gm  # noqa: E402
from tools import SkillTools  # noqa: E402


def r2(v, k=3):
    return [round(float(x), k) for x in v]


def jdefault(x):
    return r2(x) if hasattr(x, "__len__") else float(x)


class Unknown(Exception):
    pass


# ====================================================================================== the semantic tool layer
class UnifiedHarness:
    def __init__(self, A, prof):
        self.A, self.P = A, prof
        self.T = SkillTools(A, prof)
        self.arms = tuple(prof["arms"])
        self.single = len(self.arms) == 1
        self.cm = prof.get("units", "m") == "cm"
        self.manifest = set(prof["tools"])
        self.pending_image = None
        if hasattr(A, "holding_ref"):
            A.holding_ref = self.T.holding
            A.held_est_fn = self.T.held_est
        self.side_grasp = hasattr(A, "cur_approach_vec")      # adapter supports side / edge / fixed-part grasps
        self.full_quat = hasattr(A, "quat_for")               # controller takes arbitrary orientations

    # ---------------------------------------------------------------- formatting in the adapter's display units
    def pos(self, v):
        return [round(float(x) * 100, 1) for x in v] if self.cm else r2(v)

    def z(self, v):
        return round(float(v) * 100, 1) if self.cm else round(float(v), 3)

    def xy_in(self, xy):
        return None if xy is None else [float(x) / (100 if self.cm else 1) for x in xy]

    def _dir(self, word):
        r, f = self.A.cam_dirs()
        word = str(word).lower().strip()
        return {"left": -r, "right": r, "front": -f, "back": f, "forward": f, "backward": -f, "away": f, "toward": -f}.get(word)

    def _dirvec(self, word):
        word = str(word).lower().strip()
        if word in ("up", "down"):
            return np.array([0, 0, 1.0 if word == "up" else -1.0])
        v = self._dir(word)
        return None if v is None else np.array([v[0], v[1], 0.0])

    def describe(self, names):
        A = self.A
        obs = A.perceive(names)
        out = {}
        for n in names:
            o = obs.get(n)
            out[n] = ({"visible": True, "center": self.pos(o["center"]), "size": self.pos(o["max"] - o["min"]),
                       "top_z": self.z(o["max"][2])} if o is not None else {"visible": False})
            if "#" in n:
                continue
            if hasattr(A, "instances"):                          # instances labelled in the image (stable numbering)
                inst = A.instances(n)
                if len(inst) > 1:
                    out[n]["instances"] = {f"{n} #{i}": {"center": self.pos(o2["center"]), "size": self.pos(o2["max"] - o2["min"])}
                                           for i, o2 in inst}
                    out[n]["note"] = f"{len(inst)} instances seen (labelled in the image); refer to them as '{n} #1'..'{n} #{len(inst)}'"
                    A.register([f"{n} #{i}" for i, _ in inst])
                    A.perceive([f"{n} #{i}" for i, _ in inst], refresh=False)
                    out[n].pop("center", None)
            elif hasattr(A, "masks"):                            # instances numbered left to right in the image
                try:
                    k = len(A.masks(n))
                except Exception:  # noqa: BLE001
                    k = 0
                if k > 1:
                    out[n]["instances"] = k
                    out[n]["note"] = f"{k} instances seen; name them '{n} #1'..'{n} #{k}' (left to right in the image)"
        return out

    def state_text(self):
        A, T = self.A, self.T
        if self.single:
            a = self.arms[0]
            h = A.hand_pose(a)
            held = T.holding.get(a)
            return (f"Gripper: tcp [{h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}], opening {A.width() / 0.08 * 100:.0f}%, "
                    f"holding: {held or 'nothing (per last verified grasp)'}.")
        parts = []
        for a in self.arms:
            parts.append(f"{a}: gripper at {self.pos(A.hand_pose(a)[:3])} cm, opening {A.opening(a) * 100:.0f}%, "
                         f"holding {T.holding.get(a) or 'nothing'}")
        return "Arms: " + "; ".join(parts) + "."

    def _exec(self, plan):
        r = self.T.execute([plan])
        a = r["arms"][plan["arm"]]
        out = {"code": a["code"], "ok": a["ok"]}
        if not self.single:
            out = {"arm": plan["arm"], **out}
        for k in ("pose_err_cm",):
            if k in a:
                out[k] = a[k]
        if a.get("extra"):
            out.update({k: (self.pos(v) if k == "grasp_point" else v) for k, v in a["extra"].items() if k != "close"})
        out["object_motion_cm"] = {n: m["moved_cm"] for n, m in r["obj_motion"].items()}
        return out

    # ---------------------------------------------------------------- relation -> object-centre target (box solver)
    def _target(self, obj, ref, relation, gap, xy=None):
        A = self.A
        o = A.cache.get(obj) or A.get(obj, refresh=False)
        half = (o["max"] - o["min"]) / 2 if o is not None else np.array([0.03, 0.03, 0.03])
        if ref == "table":
            if xy is None:
                raise ValueError("ref 'table' needs xy=[x, y]")
            return {"ref": None, "rel": [float(xy[0]), float(xy[1]), float(half[2]) + 0.01, 1, 0, 0, 0]}
        A.register([ref])
        r = A.get(ref)
        if r is None:
            raise RuntimeError(f"cannot see the {ref}")
        rc, rh = r["center"], (r["max"] - r["min"]) / 2
        relation = relation.lower().replace(" ", "_")
        if relation in ("in", "into", "inside"):
            dz = float(r["max"][2] - rc[2]) + float(half[2]) + 0.02
            return {"ref": ref, "rel": [0, 0, dz, 1, 0, 0, 0]}
        if relation in ("on", "onto", "on_top_of"):
            dz = float(r["max"][2] - rc[2]) + float(half[2]) + 0.01
            return {"ref": ref, "rel": [0, 0, dz, 1, 0, 0, 0]}
        word = {"left_of": "left", "right_of": "right", "front_of": "front", "in_front_of": "front", "behind": "back",
                "near": None, "next_to": None, "beside": None}.get(relation, "?")
        if word == "?":
            raise ValueError(f"unknown relation {relation}")
        if word is None:
            v = (o["center"][:2] - rc[:2]) if o is not None else -A.cam_dirs()[1]
            v = v / (np.linalg.norm(v) + 1e-9)
        else:
            v = self._dir(word)
        ext_r = float(np.abs(((r["pts"][:, :2] - rc[:2]) @ v)).max())
        ext_o = float(np.abs(half[:2] @ np.abs(v)))
        d = ext_r + ext_o + gap
        dz = -float(rc[2]) + float(half[2]) + 0.01
        return {"ref": ref, "rel": [float(v[0] * d), float(v[1] * d), dz, 1, 0, 0, 0]}

    # ---------------------------------------------------------------- dispatcher
    def _arm(self, args, allow_both=False):
        if self.single:
            return self.arms[0]
        a = args.get("arm")
        if a in self.arms or (allow_both and a == "both"):
            return a
        raise ValueError(f"arm must be {' or '.join(repr(x) for x in self.arms)}" + (" or 'both'" if allow_both else ""))

    def call(self, name, args):
        A, T = self.A, self.T
        # object / ref names are registered for perception at the start of a call, except for the gripper-only tools
        # and grasp (which registers its own object) of side-grasp adapters (as the LIBERO-PRO harness of the paper runs)
        own = self.side_grasp and (name in ("grasp", "inspect", "move", "twist", "lift", "release") or
                                   (name in ("place", "carry") and args.get("ref", "table") == "table"
                                    and args.get("bottom_cm") is not None and args.get("xy") is not None))
        if not own:
            A.register([args[k] for k in ("object", "ref") if isinstance(args.get(k), str) and args.get(k) != "table"])
        if name not in self.manifest:
            return {"code": "UNKNOWN_TOOL", "why": f"no tool named {name}"}
        if name in ("home", "release") and not self.single and args.get("arm") == "both":
            out = {}
            for x in self.arms:
                out[x] = self.call(name, {**args, "arm": x})
            return out
        if name == "locate":
            objs = args.get("objects") or args.get("object") or list(getattr(A, "scene_names", []))
            objs = [objs] if isinstance(objs, str) else list(objs)
            if hasattr(A, "set_descriptions") and isinstance(args.get("descriptions"), dict):
                A.set_descriptions(args["descriptions"])
            A.register(objs)
            return {"objects": self.describe(objs)}
        if name == "grasp":
            return self.grasp(self._arm(args), args)
        if name == "inspect":
            objs = args.get("objects") or args.get("object") or []
            objs = [objs] if isinstance(objs, str) else list(objs)
            A.register(objs)
            A.perceive(objs)
            im = A.crops(objs)
            if im is None:
                return {"code": "NOT_SEEN", "why": "none of these objects is visible"}
            self.pending_image = im
            return {"code": "OK", "note": "close-up views of the requested objects are attached to this turn's images (labelled)"}
        if name == "move":
            return self.move(self._arm(args), args)
        if name == "twist":
            deg = float(args.get("degrees", 45))
            r0 = A.tool_R()
            for d in A.twist_plan(deg):
                A.run({self.arms[0]: [d]})
            r1 = A.tool_R()
            ang = float(np.degrees(np.arctan2(np.cross(r0[:, 1], r1[:, 1]) @ (-r0[:, 0]), r0[:, 1] @ r1[:, 1])))
            return {"code": "OK" if abs(ang - deg) < 0.3 * abs(deg) + 3 else "SHORT", "turned_deg": round(ang, 1),
                    "opening": round(A.opening(), 3)}
        if name == "lift":
            return self.lift(self._arm(args, allow_both=True), float(args.get("height_cm", 15)))
        if name == "release":
            out = self.release(self._arm(args))
            if self.P.get("clear_view"):
                self._clear_view()
            return out
        if name in ("place", "carry"):
            out = self.place(name, self._arm(args), args)
            if name == "place" and out.get("ok") and self.P.get("clear_view"):
                self._clear_view()
            return out
        if name in ("push", "articulate"):
            return self.push_or_articulate(name, args)
        if name == "press":
            arm = self._arm(args)
            r = T.press(arm, args["object"])
            for k in ("pressed_point",):
                if k in r:
                    r[k] = self.pos(r[k])
            if r.get("lowest_gripper_z") is not None:
                r["lowest_gripper_z"] = self.z(r["lowest_gripper_z"])
            return {"arm": arm, **r} if not self.single else r
        if name == "handover":
            to = args.get("to_arm") or args.get("arm")
            if to not in self.arms or self.single:
                raise ValueError("to_arm must be 'left' or 'right'")
            return self.handover(args["object"], to)
        if name == "bimanual":
            return self.bimanual(args.get("calls") or [])
        if name == "home":
            return self._exec({"arm": self._arm(args), "tool": "home", "actions": [{"type": "home"}]})
        if name in ("verify", "done"):
            ok, res = self.evaluate(args.get("checks") or [])
            return {"all_pass": ok, "checks": res}
        if name == "give_up":
            return {"code": "GAVE_UP"}
        return {"code": "UNKNOWN_TOOL", "why": f"no tool named {name}"}

    # ---------------------------------------------------------------- tools
    def grasp(self, arm, args):
        A, T = self.A, self.T
        obj = args["object"]
        A.register([obj])
        if T.holding.get(arm):
            return {"code": "BUSY", "why": f"already holding {T.holding[arm]}; place or release it first"}
        contact = None
        if self.side_grasp:
            ap = str(args.get("approach", "top")).lower().strip()
            if ap in ("top", "above", "down", ""):
                A.cur_approach_vec = None
            else:
                v = self._dirvec(ap)
                if v is None or abs(v[2]) > 0:
                    return {"code": "BAD_ARG", "why": "approach must be top/left/right/front/back"}
                A.cur_approach_vec = -v
            A.cur_side = None
            if args.get("edge") and A.cur_approach_vec is None:
                v = self._dirvec(args["edge"])
                if v is None or abs(v[2]) > 0:
                    return {"code": "BAD_ARG", "why": "edge must be left/right/front/back"}
                A.cur_side = v[:2]
            T.lift_test = bool(args.get("lift_test", True)) and A.cur_approach_vec is None
        elif args.get("part") and hasattr(A, "grasp_candidates"):
            contact = args.get("part")
        T.cancelled.pop(arm, None)
        p = T.grasp(arm, obj, contact)
        if p.get("skip"):
            if p["skip"] == "PLAN_FAIL" or (p["skip"] == "NOT_SEEN" and "why" not in p):
                return {"code": "NOT_SEEN", "why": f"cannot see the {obj}"}
            out = {"code": p["skip"], "why": p.get("why")}
            return {"arm": arm, **out} if not self.single else out
        out = self._exec(p)
        if not T.lift_test and out.get("code") == "POSE_ERR":
            out.update(code="OK", ok=True)
        return out

    def move(self, arm, args):
        A = self.A
        if "direction" in args or self.single:
            v = self._dirvec(args.get("direction", ""))
            if v is None:
                return {"code": "BAD_ARG", "why": "direction must be left/right/front/back/up/down"}
            dist = float(args.get("distance_cm", 10)) / 100
            h = A.hand_pose(arm)
            n = max(1, int(np.ceil(dist / 0.03)))
            for i in range(1, n + 1):
                A.run({arm: [{"type": "move", "pose": [*(h[:3] + v * dist * i / n), *h[3:]], "contact": True}]})
            h1 = A.hand_pose(arm)
            went = float((h1[:3] - h[:3]) @ v)
            return {"code": "OK" if went >= 0.7 * dist else "SHORT", "moved_cm": round(went * 100, 1),
                    "asked_cm": round(dist * 100, 1), "opening": round(A.opening(arm), 3)}
        d = np.array([args.get("dx", 0), args.get("dy", 0), args.get("dz", 0)], float) / 100
        h = A.hand_pose(arm)
        code = self.T.seq(arm, [np.r_[h[:3] + d, h[3:]]])
        return {"arm": arm, "code": code, "gripper_at": self.pos(A.hand_pose(arm)[:3])}

    def lift(self, arm, h_cm):
        A = self.A
        arms = list(self.arms) if arm == "both" else [arm]
        if self.single:
            h = A.hand_pose(arm)
            z = h_cm / 100
            A.move([h[0], h[1], z], *([h[3:]] if self.full_quat else []))
            return {"code": "OK", "tcp_z": round(float(A.hand_pose()[2]), 3)}
        items = {}
        for a in arms:
            t = A.hand_pose(a)
            z = h_cm / 100
            if self.P.get("lift_never_lower") and z < t[2] + 0.02:      # never lower a held load by "lifting"
                z = t[2] + h_cm / 100
            items[a] = [{"type": "move", "pose": [t[0], t[1], z, *t[3:]]}]
        ok = self.T._run_retry(items)
        out = {}
        for a in arms:
            pe = Gm.pos_err(A.hand_pose(a), items[a][0]["pose"])
            out[a] = {"code": "OK" if ok and pe <= 0.03 else ("PLAN_FAIL" if not ok else "POSE_ERR"), "tcp_z": self.z(A.hand_pose(a)[2])}
        return out if len(arms) > 1 else {"arm": arm, **out[arm]}

    def release(self, arm):
        A, T = self.A, self.T
        p = T.gripper(arm, 1.0)
        out = self._exec(p)
        if len(self.P.get("retreat", (0.08,))) > 1:
            T.retreat(arm)
        else:
            h = A.hand_pose(arm)
            cav = getattr(A, "cur_approach_vec", None)
            back = -np.asarray(cav) if cav is not None else np.array([0, 0, 1.0])
            A.move(h[:3] + 0.08 * back, *([h[3:]] if self.full_quat else []))
            if back[2] == 0:
                h = A.hand_pose(arm)
                A.move([h[0], h[1], max(h[2], self.P["safe_z"])], h[3:])
        if self.side_grasp:
            A.cur_approach_vec = None
        T.drop_hold(arm)
        return out

    def _clear_view(self):
        """after letting go: back to the home pose so the arm does not hide the scene from the camera."""
        A = self.A
        h = A.hand_pose()
        A.move([h[0], h[1], max(h[2], self.P["safe_z"])], h[3:])
        A.run({self.arms[0]: [{"type": "home"}]})

    def place(self, name, arm, args):
        A, T = self.A, self.T
        obj = args["object"]
        if T.holding.get(arm) != obj:
            out = {"code": "NOT_HELD", "why": f"{obj} is not held (grasp it first)"}
            return {"arm": arm, **out} if not self.single else out
        if hasattr(A, "feasible") and hasattr(A, "grasp_candidates"):     # support-surface placement, IK-checked
            ref = args.get("ref", "table")
            xy = self.xy_in(args.get("xy"))
            out = T.place_surface(arm, obj, ref, args.get("relation", self.P.get("default_relation", "near")), xy,
                                  release=(name == "place"), height=float(args.get("height_cm", 10)) / 100,
                                  align=bool(args.get("align", False)))
            if "target_xyz" in out:
                out["target_xyz"] = self.pos(out["target_xyz"])
            if "object_now" in out:
                out["object_now"] = {"center": self.pos(out["object_now"]["center"]), "bottom_z": self.z(out["object_now"]["bottom_z"])}
            out["ok"] = out["code"] == "OK"
            return {"arm": arm, **out}
        if args.get("ref", "table") == "table" and args.get("bottom_cm") is not None and args.get("xy") is not None and self.side_grasp:
            o = A.cache.get(obj) or A.get(obj, refresh=False)
            half = float((o["max"][2] - o["min"][2]) / 2) if o is not None else 0.03
            xy = args["xy"]
            tgt = {"ref": None, "rel": [float(xy[0]), float(xy[1]), float(args["bottom_cm"]) / 100 + half + 0.01, 1, 0, 0, 0]}
        else:
            tgt = self._target(obj, args.get("ref", "table"), str(args.get("relation", "near")),
                               float(args.get("gap_cm", 3)) / 100, args.get("xy"))
        p = T.place(arm, obj, tgt, release=(name == "place"))
        if p.get("skip"):
            return {"code": p["skip"]}
        out = self._exec(p)
        if name == "place":
            T.holding.pop(arm, None)
            T.hold.pop(arm, None)
            if self.side_grasp:
                A.cur_approach_vec = None
        return out

    def push_or_articulate(self, name, args):
        A, T = self.A, self.T
        obj = args["object"]
        if name == "articulate" and hasattr(A, "feasible") and (args.get("degrees") is not None or not args.get("ref")):
            arm = args.get("arm") if args.get("arm") in self.arms else "auto"
            return T.articulate_hinged(arm, obj, args.get("motion", "open"), float(args.get("degrees", 80)))
        arm = self.arms[0] if self.single else self._arm(args)
        dist = req_dist = float(args.get("distance_cm", 10)) / 100
        if getattr(A, "inject", None) and A.inject[0] == "push_scale":
            dist *= float(A.inject[1])
        if name == "push":
            v = self._dir(args.get("direction", ""))
            if v is None:
                return {"code": "BAD_ARG", "why": "direction must be left/right/front/back"}
        else:
            ref = args.get("ref")
            o, r = A.get(obj), (A.get(ref) if ref else None)
            if o is None or (ref and r is None):
                return {"code": "NOT_SEEN"}
            v = (r["center"][:2] - o["center"][:2]) if r is not None else self._dir("back")
            v = v / (np.linalg.norm(v) + 1e-9)
            ax = int(np.argmax(np.abs(v)))           # sliding joints move along a principal horizontal axis of the body
            v = np.eye(2)[ax] * np.sign(v[ax])
            if str(args.get("motion", "close")) == "open":
                return {"code": "UNSUPPORTED", "why": "open needs a handle grasp: grasp the handle, then use carry"}
        if T.holding.get(arm):
            return {"code": "BUSY", "why": f"holding {T.holding[arm]}; release it first"}
        first = A.first.get(obj)
        o0 = A.cache.get(obj)
        p = T.push(obj, v, dist, args.get("height_cm") and float(args["height_cm"]) / 100, arm=arm)
        if p.get("skip"):
            return {"code": p["skip"]}
        out = self._exec(p)
        o1 = A.get(obj)
        if o1 is not None and first is not None:
            prog = float(np.dot(o1["center"][:2] - first["center"][:2], v))
            out["moved_along_direction_cm_since_first_seen"] = round(prog * 100, 1)
        if o1 is not None and o0 is not None:
            step = float(np.dot(o1["center"][:2] - o0["center"][:2], v))
            out["moved_this_call_cm"] = round(step * 100, 1)
            out["code"] = "OK" if step >= 0.5 * req_dist else "SHORT"
            out["ok"] = out["code"] == "OK"
        out.pop("pose_err_cm", None)
        if name == "articulate":
            out["check_hint"] = ["slid", obj, args.get("ref"), round(dist * 100 * 0.6)]
        return out

    def handover(self, obj, to_arm):
        """put the object down between the arms with the giving arm, move that arm home, grasp it with to_arm."""
        frm = [a for a in self.arms if a != to_arm][0]
        if self.T.holding.get(frm) != obj:
            raise ValueError(f"the {frm} arm does not hold {obj}")
        o = self.A.get(obj)
        y = float(o["center"][1]) if o is not None else 0.0
        sx = float(self.A.cam_dirs()[0][0])
        mid = np.array([(0.04 if to_arm == self.P["lateral_arm"][1] else -0.04) * sx, min(max(y, -0.10), -0.02)])
        r1_ = self.call("place", {"arm": frm, "object": obj, "ref": "table", "relation": "table",
                                  "xy": [mid[0] * (100 if self.cm else 1), mid[1] * (100 if self.cm else 1)]})
        if r1_.get("code") != "OK":
            return {"code": "HANDOVER_FAIL", "put_down": r1_}
        self.call("home", {"arm": frm})
        r2_ = self.call("grasp", {"arm": to_arm, "object": obj})
        return {"code": r2_.get("code"), "mode": "via the table", "put_down_at": self.pos(mid), to_arm: r2_}

    def bimanual(self, calls):
        if len(calls) != 2:
            raise ValueError("bimanual needs exactly two calls")
        cs = [(c["name"], c.get("arguments") or {}) for c in calls]
        if {x[1].get("arm") for x in cs} != set(self.arms):
            raise ValueError("bimanual needs one call per arm (left and right)")
        if {x[0] for x in cs} == {"grasp"}:
            for _, x in cs:
                if self.T.holding.get(x["arm"]):
                    raise ValueError(f"{x['arm']} already holds {self.T.holding[x['arm']]}")
                self.A.register([x["object"]])
            out = self.T.bimanual_grasp([x for _, x in cs])
            for a, r in out.items():
                if "grasp_point" in r:
                    r["grasp_point"] = self.pos(r["grasp_point"])
            return out
        out = {}
        for n, x in cs:
            out[x["arm"]] = self.call(n, x)
        return out

    # ---------------------------------------------------------------- predicates (generic, perception + proprioception)
    def _g(self, n, now):
        o = now.get(n)
        if o is None:
            raise Unknown(f"cannot see the {n}")
        return o

    def _holding(self, o, tcp, w):
        if w >= self.P["open_w"] or w <= self.P["closed_w"]:
            return False
        return bool(np.all(tcp >= o["min"] - 0.03) and np.all(tcp <= o["max"] + 0.03))

    @staticmethod
    def _inside(a, c, margin=0.0):
        xy = a["center"][:2]
        return bool(np.all(xy > c["min"][:2] + margin) and np.all(xy < c["max"][:2] - margin) and a["min"][2] < c["max"][2] + 0.03)

    def evaluate(self, checks):
        A = self.A
        names = []
        for c in checks:
            for a in c[1:]:
                if isinstance(a, str) and a not in names and self._dir(a) is None:
                    names.append(a)
        A.register(names)
        now = A.perceive(names)
        hands = {a: (A.hand_pose(a)[:3], self.T.W(a)) for a in self.arms}
        out, allok = [], True
        for c in checks:
            if not isinstance(c, (list, tuple)) or not c:
                out.append({"check": c, "ok": False, "why": "malformed check"})
                allok = False
                continue
            p, args = str(c[0]), list(c[1:])
            try:
                ok, why = self._pred(p, args, now, hands)
            except Unknown as e:
                ok, why = None, str(e)
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"bad check: {type(e).__name__}: {e}"
            out.append({"check": [p, *args], "ok": ok, "why": why})
            if ok is not True:
                allok = False
        return allok, out

    def _pred(self, p, args, now, hands):
        A, T = self.A, self.T
        g = lambda n: self._g(n, now)  # noqa: E731
        if self.P.get("same_object_guard") and p in ("in", "on", "next_to", "left_of", "right_of", "front_of", "behind") and len(args) >= 2:
            a, b = now.get(args[0]), now.get(args[1])
            if a is not None and b is not None and np.linalg.norm(a["center"] - b["center"]) < 0.015:
                return False, f"{args[0]} and {args[1]} are the same object in the camera view"
        if p == "in":
            a, c = g(args[0]), g(args[1])
            d = float(np.linalg.norm(a["center"][:2] - c["center"][:2]))
            return self._inside(a, c), f"{args[0]} centre {d * 100:.0f} cm from {args[1]} centre, bottom {a['min'][2] * 100:.0f} cm vs rim {c['max'][2] * 100:.0f} cm"
        if p == "not_in":
            a, c = g(args[0]), g(args[1])
            return (not self._inside(a, c)), f"{args[0]} {'is' if self._inside(a, c) else 'is not'} inside {args[1]}"
        if p == "on":
            a, b = g(args[0]), g(args[1])
            over = bool(np.all(a["center"][:2] > b["min"][:2] - 0.025) and np.all(a["center"][:2] < b["max"][:2] + 0.025))
            snap = float(self.P.get("bottom_snap", 0.0))   # perception reports bottoms below this height as 0
            above = bool(a["min"][2] >= b["max"][2] - 0.03 and (a["min"][2] > 0.004 or (snap > 0 and b["max"][2] < snap)))
            return over and above, f"{args[0]} over {args[1]}: {over}; bottom {a['min'][2] * 100:.0f} cm vs top {b['max'][2] * 100:.0f} cm"
        if p == "on_table":
            a = g(args[0])
            return float(a["min"][2]) < 0.025, f"{args[0]} bottom {a['min'][2] * 100:.1f} cm above the table"
        if p in ("left_of", "right_of", "front_of", "behind"):
            a, b = g(args[0]), g(args[1])
            if p in ("left_of", "right_of") and self.P.get("pixel_lr", "u" in a and "u" in b):     # image columns
                d = b["u"] - a["u"] if p == "left_of" else a["u"] - b["u"]
                return d > 3, f"{args[0]} is {abs(d):.0f} px {'left' if b['u'] > a['u'] else 'right'} of {args[1]} in the image"
            v = self._dir({"left_of": "left", "right_of": "right", "front_of": "front", "behind": "back"}[p])
            d = float(np.dot(a["center"][:2] - b["center"][:2], v))
            return d > 0.02, f"{args[0]} is {d * 100:.0f} cm {p.replace('_of', '').replace('front', 'in front of')} {args[1]}"
        if p in ("next_to", "near"):
            a, b = g(args[0]), g(args[1])
            d = float(np.linalg.norm(a["center"][:2] - b["center"][:2]))
            top = bool(a["min"][2] > b["max"][2] + 0.01 and np.all(a["center"][:2] > b["min"][:2]) and np.all(a["center"][:2] < b["max"][:2]))
            return d <= 0.28 and not top, f"centres {d * 100:.0f} cm apart{', on top of it' if top else ''}"
        if p in ("moved", "displaced", "closer", "shrunk"):
            a = g(args[0])
            a0 = A.first.get(args[0])
            if a0 is None:
                raise Unknown(f"no initial observation of {args[0]}")
            if p == "moved":
                v = self._dir(args[1])
                if v is None:
                    return False, f"unknown direction {args[1]}"
                prog = float(np.dot(a["center"][:2] - a0["center"][:2], v))
                return prog * 100 >= float(args[2]), f"{args[0]} moved {prog * 100:.1f} cm {args[1]} since first seen"
            if p == "displaced":
                d = float(np.linalg.norm(a["center"][:2] - a0["center"][:2]))
                return d * 100 >= float(args[1]), f"{args[0]} displaced {d * 100:.1f} cm since first seen"
            if p == "closer":
                b, b0 = g(args[1]), A.first.get(args[1])
                if b0 is None:
                    raise Unknown(f"no initial observation of {args[1]}")
                d = float(np.linalg.norm(a["center"][:2] - b["center"][:2]))
                d0 = float(np.linalg.norm(a0["center"][:2] - b0["center"][:2]))
                r = float(args[2]) if len(args) > 2 else 0.75
                return d <= r * d0, f"{args[0]}-{args[1]} distance {d * 100:.0f} cm (initially {d0 * 100:.0f} cm)"
            e0 = a0["max"][:2] - a0["min"][:2]
            ax = int(np.argmax(e0))
            d = float(e0[ax] - (a["max"][ax] - a["min"][ax]))
            return d * 100 >= float(args[1]), f"{args[0]} footprint shrank {d * 100:.1f} cm along its long axis"
        if p == "slid":
            a, a0, r0 = g(args[0]), A.first.get(args[0]), A.first.get(args[1])
            if a0 is None or r0 is None:
                raise Unknown("no initial observation")
            v = r0["center"][:2] - a0["center"][:2]
            ax = int(np.argmax(np.abs(v)))
            v = np.eye(2)[ax] * np.sign(v[ax])
            prog = float(np.dot(a["center"][:2] - a0["center"][:2], v))
            return prog * 100 >= float(args[2]), f"{args[0]} slid {prog * 100:.1f} cm into {args[1]} since first seen"
        if p == "height":
            est = [T.held_est(x) for x, o in T.holding.items() if o == args[0] and x in T.hold]
            if est:                                   # held: gripper pose + in-hand geometry (the hand occludes it)
                bot = max(e[1] for e in est)
                return bot * 100 >= float(args[1]), f"{args[0]} held; bottom {bot * 100:.1f} cm above the table (from the gripper pose)"
            a = g(args[0])
            return float(a["min"][2]) * 100 >= float(args[1]), f"{args[0]} bottom {a['min'][2] * 100:.1f} cm above the table"
        if p == "holding":
            if not self.single and len(args) >= 2 and args[0] in self.arms:
                arm, obj = args[0], args[1]
                ok = T.holding.get(arm) == obj and self.T.W(arm) > self.P["closed_w"]
                return ok, f"{arm} holds {T.holding.get(arm)}; opening {A.opening(arm) * 100:.0f}%"
            a = now.get(args[0]) or A.cache.get(args[0])
            if a is None:
                raise Unknown(f"cannot see the {args[0]}")
            hs = [x for x, (tcp, w) in hands.items() if self._holding(a, tcp, w)]
            tcp, w = hands[self.arms[0]]
            if self.single:
                return bool(hs), f"gripper width {w * 100:.1f} cm, tcp {'inside' if self._holding(a, tcp, 0.04) else 'outside'} its box"
            return bool(hs), f"held by {hs or 'no gripper'}"
        if p == "released":
            if not self.single and args and args[0] in self.arms:
                o = A.opening(args[0])
                return o > self.P["open_w"], f"{args[0]} gripper opening {o * 100:.0f}%"
            if all(w >= self.P["open_w"] for _, w in hands.values()):
                return True, "gripper open"
            a = now.get(args[0])
            if a is None:
                raise Unknown(f"cannot see the {args[0]}")
            held = [x for x, (tcp, w) in hands.items() if self._holding(a, tcp, w)]
            return not held, (f"gripper width {hands[self.arms[0]][1] * 100:.1f} cm" if self.single else f"held by {held or 'no gripper'}")
        if p == "lifted":
            a = now.get(args[0]) or A.cache.get(args[0])
            if a is None:
                raise Unknown(f"cannot see the {args[0]}")
            need = float(args[1]) / 100 if len(args) > 1 else 0.08
            hs = [x for x, (tcp, w) in hands.items() if self._holding(a, tcp, w) or T.holding.get(x) == args[0]] if not self.single else \
                [x for x, (tcp, w) in hands.items() if self._holding(a, tcp, w)]
            h = max([float(hands[x][0][2]) for x in hs], default=float(hands[self.arms[0]][0][2]))
            return bool(hs) and h >= need + 0.01, f"holding: {bool(hs)}; gripper {h * 100:.0f} cm above the table"
        if p == "pressed":
            return args[0] in T.pressed, f"a press reached the top of {args[0]}: {args[0] in T.pressed}"
        return False, f"unknown predicate {p}"


# ====================================================================================== profiles (embodiment, no tasks)
def guava_profile():
    import adapter_guava as AG
    return {"arms": ("right",), "units": "m", "retry": 2, "regrasp": (90.0,), "backoff": 0.08, "retreat": (0.08,),
            "closed_w": AG.CLOSED_W, "open_w": AG.OPEN_W, "safe_z": AG.SAFE_Z, "clear_view": False,
            "same_object_guard": False, "default_relation": "near", "pixel_lr": True,
            "tools": {"locate", "grasp", "lift", "place", "carry", "push", "articulate", "release", "home", "verify", "done", "give_up"}}


def libero_profile():
    import adapter_libero as AL
    return {"arms": ("right",), "units": "m", "retry": 0, "regrasp": (90.0,), "backoff": 0.08, "retreat": (0.08,),
            "closed_w": AL.CLOSED_W, "open_w": AL.OPEN_W, "safe_z": AL.SAFE_Z, "clear_view": True,
            "same_object_guard": True, "default_relation": "near", "pixel_lr": False,   # LIBERO records have no image
            # column: left/right is measured in metres along the camera's right axis
            "tools": {"locate", "inspect", "grasp", "lift", "move", "twist", "place", "carry", "push", "articulate", "release",
                      "home", "verify", "done", "give_up"}}


# ====================================================================================== --gen flags (flags.py) and public API
GEN_DEFAULT = "default"              # Guava / LIBERO: the per-benchmark default set below
GEN_DEFAULTS = {                     # held_track and step_open stay off; Guava has no instance
    "guava": ("post_release", "honest_done", "fail_post"),               # tracking, so inst_lock applies to LIBERO only
    "libero": ("post_release", "honest_done", "fail_post", "inst_lock"),
}


def _bench():
    """which benchmark this process runs: $HARNESS_BENCH (run.py sets it from its sub-command; library users set it
    before importing this module)."""
    b = os.environ.get("HARNESS_BENCH", "libero")
    if b not in ("guava", "libero", "robotwin"):
        raise RuntimeError(f"HARNESS_BENCH={b!r}: expected guava, libero or robotwin")
    return b


BENCH = _bench()
INFO = {"bench": BENCH, "gen_default": GEN_DEFAULT, "gen_defaults": {k: list(v) for k, v in GEN_DEFAULTS.items()}}


def gen_flags(spec, bench):
    spec = (spec or "").strip()
    if spec in ("", "none", "0", "off"):
        return set()
    import flags as _G
    if spec == "default":
        return set(GEN_DEFAULTS[bench])
    if spec in ("all", "1", "on"):
        return set(_G.FLAGS)
    fl = {x.strip() for x in spec.split(",") if x.strip()}
    bad = fl - set(_G.FLAGS)
    if bad:
        raise SystemExit(f"unknown --gen flags {sorted(bad)}; known: {list(_G.FLAGS)}")
    return fl


def _make_harness(A, prof, flags):
    if not flags:
        return UnifiedHarness(A, prof)
    import flags as _G
    return _G.GenHarness(A, prof, set(flags))


def _system_addendum(bench, flags):
    if not flags:
        return ""
    import flags as _G
    f = getattr(_G, "system_addendum", None)
    return (f(bench, set(flags)) or "") if f else ""


def make_harness(bench, A, prof, flags=None):
    """public: the Guava / LIBERO tool layer; flags=None -> the default --gen set (GEN_DEFAULT / $HARNESS_GEN)."""
    if flags is None:
        flags = gen_flags(os.environ.get("HARNESS_GEN", GEN_DEFAULT), bench)
    return _make_harness(A, prof, set(flags))


def system_text(bench, base, flags=None):
    """public: the SYSTEM prompt a Guava / LIBERO driver must send (base = guava_protocol.SYSTEM / libero_protocol.SYSTEM)."""
    if flags is None:
        flags = gen_flags(os.environ.get("HARNESS_GEN", GEN_DEFAULT), bench)
    return base + _system_addendum(bench, set(flags))


# ====================================================================================== the planner loop (shared)
XML_CALLS = False      # --xml-calls: also accept <function=NAME><parameter=K>V</parameter></function> (off = original parsers)


def parse_xml_call(text):
    m = re.search(r"<function=([\w.-]+)>(.*?)</function>", text or "", re.S)
    if not m:
        raise ValueError("no tool call")
    args = {}
    for k, v in re.findall(r"<parameter=([\w.-]+)>(.*?)</parameter>", m.group(2), re.S):
        v = v.strip()
        try:
            args[k] = json.loads(v)
        except Exception:  # noqa: BLE001
            args[k] = v
    return m.group(1), args


def parse(text):
    parse.last_xml = False
    try:
        return _parse(text)
    except Exception:  # noqa: BLE001
        if not XML_CALLS:
            raise
        think = re.search(r"<think>(.*?)</think>", text or "", re.S)
        name, args = parse_xml_call(text)
        parse.last_xml = True
        return (think.group(1).strip() if think else ""), name, args


def _parse(text):
    think = ""
    m = re.search(r"<think>(.*?)</think>", text or "", re.S)
    if m:
        think = m.group(1).strip()
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for src in (t, text or ""):              # the call after the reasoning; else anywhere (also inside <think>)
        m = re.search(r"<tool_call>(.*?)</tool_call>", src, re.S)
        body = m.group(1) if m else src[src.find("{"): src.rfind("}") + 1]
        try:
            v, _ = json.JSONDecoder().raw_decode(body.strip())
            return think, v["name"], v.get("arguments", {}) or {}
        except Exception:  # noqa: BLE001
            if src is not t:
                raise
    raise ValueError("no tool call")


def chat(url, body, timeout):
    import urllib.request
    rq = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    j = json.load(urllib.request.urlopen(rq, timeout=timeout))
    m = j["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or m.get("reasoning"), j.get("usage"), j["choices"][0].get("finish_reason")


class Branch:
    """recovery-branch replay for run_loop:
    rec = recorded base steps (episode.json "steps"), fork = last replayed decision (its call runs WITH the perturbation),
    hook = object with before(name, args) -> tag|None and after(name, args, res) (+ .detail, .applied), code = result-code
    function used for the divergence check. Decisions <= fork answer with the recorded raw/reasoning/call (no model call);
    the visible clock (LIBERO "Time used") is set to the recorded one during the replay; a replayed decision < fork whose
    result code differs from the recording ends the episode as replay_diverged (S["replay_ok"] = False)."""
    TIME = re.compile(r"Time used: (\d+)/")

    def __init__(self, rec, fork, hook, code):
        self.rec, self.fork, self.hook, self.code = rec, fork, hook, code


def run_loop(D, H, url, model, thinking, max_tokens, replay=None, branch=None):
    """the Guava / LIBERO planner loop. D = benchmark driver (protocol: budget, observation rendering, success logging);
    H = UnifiedHarness (or flags.GenHarness). replay = list of recorded calls [(name, args)] to execute instead of asking
    the planner (parity tests); branch = Branch replay of recovery branches. H.done_reject_suffix (GenHarness only) may
    replace the text appended to a rejected done."""
    S = {"msgs": [{"role": "system", "content": D.system}], "steps": [], "rejects": 0, "end": D.default_end,
         "n_actions": 0, "result_text": None, "usage": [], "dec": 0, "first_success": None, "llm_s": 0.0}
    if branch is not None:
        S["replay_ok"] = True
    t0 = time.time()
    D.t0 = t0

    def set_clock(dec):
        nonlocal t0
        m = branch.TIME.search(branch.rec[dec].get("obs_text") or "") if dec < len(branch.rec) else None
        if m:
            t0 = time.time() - float(m.group(1))
            D.t0 = t0
            if hasattr(D, "start_clock") and dec > 0:
                D.start_clock(t0)

    while S["dec"] < D.cap:
        brep = branch is not None and S["dec"] <= branch.fork
        if brep:
            set_clock(S["dec"])
        el = time.time() - t0
        if D.wall_visible and el >= D.wall:
            S["end"] = "budget"
            break
        if not D.wall_visible and el >= D.wall:
            break
        dec = S["dec"]
        if replay is not None and dec >= len(replay):
            S["end"] = "replay_end"
            break
        content, img_name = D.observe(dec, H)
        if branch is not None and dec == 0 and hasattr(D, "start_clock"):
            D.start_clock(t0)
        txt = (D.first_text() if dec == 0 else f"Result of {S['steps'][-1]['call']['name']}: {S['result_text']}\n")
        txt += H.state_text() + D.budget_text(el, S)
        if H.pending_image is not None:
            content += D.closeup(dec, H.pending_image)
            H.pending_image = None
            txt = "(second image = close-ups)\n" + txt
        S["msgs"].append({"role": "user", "content": content + [{"type": "text", "text": txt}]})
        req, seen = [], 0
        for m in reversed(S["msgs"]):
            if isinstance(m["content"], list):
                seen += 1
                if seen > D.keep_turns:
                    m = {"role": m["role"], "content": [c for c in m["content"] if c["type"] == "text"]}
                else:
                    m = {"role": m["role"], "content": [D.encode(c) if c["type"] == "image_url" else c for c in m["content"]]}
            req.append(m)
        req.reverse()
        text, err, rc, think, name, args, fin = None, None, None, "", None, None, None
        tl = time.time()
        if brep:
            r0 = branch.rec[dec]
            text, rc, think, fin = r0["raw"], r0.get("reasoning"), r0.get("think") or "", r0.get("finish_reason")
            name, args = r0["call"]["name"], r0["call"]["arguments"]
        elif replay is not None:
            name, args = replay[dec]
            text = json.dumps({"name": name, "arguments": args})
        else:
            for attempt in range(D.attempts):
                left = D.wall - (time.time() - t0)
                if D.wall_visible and left <= 1:
                    break
                try:
                    body = {"model": model, "messages": req, "temperature": 0.0 if attempt == 0 else 0.6, "max_tokens": max_tokens,
                            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
                    D.record(body, dec, attempt)
                    text, rc, usage, fin = chat(url, body, timeout=max(5.0, left) if D.wall_visible else 600)
                    S["usage"].append(usage)
                    think, name, args = parse(text)
                    think = think or rc or ""
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:300]
                    S.setdefault("bad_outputs", []).append({"k": dec, "attempt": attempt, "text": (text or "")[-1500:], "reasoning": (rc or "")[-500:]})
                    text = None
                    if "Connection refused" in err or "Max retries" in err or "URLError" in err:
                        S["infra_error"] = err
                    time.sleep(D.sleep(attempt))
        S["llm_s"] += time.time() - tl
        if text is None:
            S["end"] = "budget" if (D.wall_visible and D.wall - (time.time() - t0) <= 1) else "invalid_model_output"
            S["steps"].append({"k": dec, "image": img_name, "obs_text": txt, "raw": None, "error": err,
                               "bad_outputs": [b for b in S.get("bad_outputs", []) if b["k"] == dec]})
            break
        if D.wall_visible and time.time() - t0 >= D.wall:
            S["end"] = "budget"
            S["steps"].append({"k": dec, "image": img_name, "obs_text": txt, "raw": text, "late": True})
            break
        # history carries the call only, never the reasoning
        S["msgs"].append({"role": "assistant",
                          "content": text.split("</think>")[-1].lstrip("\n") if "</think>" in text else text})
        st = {"k": dec, "image": img_name, "obs_text": txt, "raw": text, "think": think, "reasoning": rc,
              "call": {"name": name, "arguments": args}, "t_start": round(time.time() - t0, 1), "finish_reason": fin,
              "xml_call": bool(getattr(parse, "last_xml", False)) and replay is None and not brep}
        if brep:
            st["replayed"] = True
        S["n_actions"] += 1
        D.before_call()
        tag = branch.hook.before(name, args) if (branch is not None and dec == branch.fork) else None
        if tag:
            st["perturbation"] = tag
        try:
            res = H.call(name, args)
        except Exception as e:  # noqa: BLE001
            res = {"code": "ERROR", "why": f"{type(e).__name__}: {e}"[:200]}
            D.after_error()
        finally:
            if branch is not None:
                D.A.inject = None
        if tag:
            branch.hook.after(name, args, res)
            st["perturbation_detail"] = dict(branch.hook.detail, applied=branch.hook.applied)
        st["result"] = res
        st["success_after"] = D.success()              # logging only, never shown to the planner
        st["t_end"] = round(time.time() - t0, 1)
        st["trace"] = D.trace()
        if st["success_after"] and S["first_success"] is None:
            S["first_success"] = st["t_end"]
        S["steps"].append(st)
        S["dec"] += 1
        S["result_text"] = json.dumps(res, default=jdefault)
        if branch is not None and dec < branch.fork and branch.code(res) != branch.code(branch.rec[dec].get("result")):
            S["replay_ok"] = False
            S["end"] = "replay_diverged"
            S["diverged_at"] = {"k": dec, "call": name, "recorded": branch.code(branch.rec[dec].get("result")), "replayed": branch.code(res)}
            break
        if name == "done":
            if res.get("all_pass") or S["rejects"] >= D.max_reject:
                S["end"] = "model_done" if res.get("all_pass") else "done_after_rejections"
                break
            S["rejects"] += 1
            hook = getattr(H, "done_reject_suffix", None)
            suf = hook(res) if hook else None
            S["result_text"] += suf if suf is not None else " -> done REJECTED: the checks did not all pass; the task is not complete yet."
        if name == "give_up":
            S["end"] = "give_up"
            break
    else:
        S["end"] = D.cap_end
    if S["end"] == "action_limit" and time.time() - t0 >= D.wall:
        S["end"] = "wall_timeout"
    S["seconds"] = round(time.time() - t0, 1)
    return S


# ====================================================================================== benchmark drivers (protocol)
def _jpg(arr, path, q):
    from PIL import Image
    Image.fromarray(np.asarray(arr)).convert("RGB").save(path, quality=q)


class Driver:
    keep_turns, attempts, max_reject, wall_visible, default_end, cap_end = 3, 3, 2, False, "action_limit", "action_limit"

    def record(self, body, k, attempt):
        pass

    def before_call(self):
        pass

    def after_error(self):
        pass

    def trace(self):
        return None

    def closeup(self, dec, im):
        return []


class GuavaDriver(Driver):
    """Guava paper protocol (adapters/guava_protocol.py): seed_base + k, push direction by k parity, caps 25/30, frontview
    (sideview for the shell game), 512x512; success = score() at stop (+3x5 settle audits for physical tasks)."""

    def __init__(self, engine, task, seed, k, folder):
        import guava_protocol as HG
        from adapter_guava import GuavaAdapter
        self.HG, self.engine, self.task, self.seed, self.k, self.folder = HG, engine, task, seed, k, folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "img").mkdir(exist_ok=True)
        rob = engine.env.rob
        if task in ("push_pot", "push_basket"):
            rob.push_directions = ["left" if k % 2 == 0 else "right"]
        ep = engine.episode(k, folder / "_guava")
        ep.reset(seed=seed)
        self.instruction = getattr(rob, "task_instruction", None) or engine.cfg.task_name
        self.A = GuavaAdapter(engine)
        self.A.start_episode()
        self.system = HG.SYSTEM
        self.cap = self.wall = HG.CAPS[task]
        self.wall = 1e9
        self.attempts = 3

    def sleep(self, attempt):
        return 2 + 10 * attempt

    def observe(self, dec, H):
        self.A.refresh()
        ip = self.folder / "img" / f"{dec:03d}.jpg"
        _jpg(self.A.rgb(), ip, 90)
        return [{"type": "image_url", "image_url": {"url": str(ip)}}], ip.name

    def encode(self, c):
        from PIL import Image
        return {"type": "image_url", "image_url": {"url": self.HG.img_url(np.asarray(Image.open(c["image_url"]["url"])))}}

    def first_text(self):
        return f"Task: {self.instruction}\n"

    def budget_text(self, el, S):
        return f"\nCalls used: {S['n_actions']}/{self.cap}."

    def after_error(self):
        pass

    def before_call(self):
        pass

    def success(self):
        return self.A.success()

    def finish(self, S, model, t0):
        S2 = {**S, "instruction": self.instruction, "prefix": ""}
        res = self.HG._finish(self.engine, self.task, self.A, S2, self.folder, model, self.seed, self.k, t0,
                              extra={"harness": "unified"})
        return res


class LiberoDriver(Driver):
    """LIBERO-PRO MotorMind protocol (adapters/libero_protocol.py): saved initial state k, 32 settle ticks, 450 s wall clock,
    40 calls; success = BDDL predicate when the agent ends the episode."""
    wall_visible, default_end, cap_end = True, "budget", "call_limit"

    def __init__(self, env, folder):
        from adapter_libero import LiberoAdapter
        import libero_protocol as HL
        self.HL = HL
        self.env, self.folder = env, folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "img").mkdir(exist_ok=True)
        self.A = LiberoAdapter(env)
        self.A.cur_approach_vec = None
        self.A.start_episode()
        self.system = HL.SYSTEM
        self.cap, self.wall = HL.MAX_CALLS, HL.WALL_S
        self.instruction = env.instruction

    def start_clock(self, t0):
        self.A.deadline = self.env.deadline = t0 + self.wall

    def sleep(self, attempt):
        return 2

    def observe(self, dec, H):
        if dec == 0:
            self.start_clock(self.t0)
        ip = self.folder / "img" / f"{dec:03d}.jpg"
        _jpg(self.A.annotated_rgb(), ip, 90)
        return [{"type": "image_url", "image_url": {"url": str(ip)}}], ip.name

    def closeup(self, dec, im):
        cp = self.folder / "img" / f"{dec:03d}_closeup.jpg"
        _jpg(im, cp, 92)
        return [{"type": "image_url", "image_url": {"url": str(cp)}}]

    def encode(self, c):
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(Path(c["image_url"]["url"]).read_bytes()).decode()}}

    def first_text(self):
        return f"Task: {self.instruction}\n"

    def budget_text(self, el, S):
        return f"\nTime used: {el:.0f}/{self.wall:.0f} s. Calls used: {S['n_actions']}."

    def success(self):
        return self.A.success()


# ---------------------------------------------------------------- entry points
def main_guava(a, flags):
    """Guava episodes k = offset .. offset + episodes - 1 (seed = seed_base + k); flags = --gen set."""
    import guava_protocol as HG
    eng = HG.make_engine(a.task, a.sam3)
    a.out.mkdir(parents=True, exist_ok=True)
    for k in range(a.offset, a.offset + a.episodes):
        seed = a.seed_base + k
        d = a.out / f"{a.task}__k{k:03d}_s{seed}"
        if (d / "result.json").exists():
            continue
        try:
            t0 = time.time()
            D = GuavaDriver(eng, a.task, seed, k, d)
            D.system = D.system + _system_addendum("guava", flags)
            H = _make_harness(D.A, guava_profile(), flags)
            replay = _load_replay(a, d.name)
            S = run_loop(D, H, a.url, a.model, a.thinking, a.max_tokens, replay=replay)
            base_sys = HG.SYSTEM                 # guava_protocol._finish writes its module SYSTEM into episode.json: make it
            HG.SYSTEM = D.system                 # the SYSTEM actually sent (base + addendum), then restore
            try:
                r = D.finish(S, a.model, t0)
            finally:
                HG.SYSTEM = base_sys
            if flags:
                _tag_result(d, flags)
            print(json.dumps({x: r[x] for x in ("task", "seed", "success", "end_reason", "decisions", "seconds")}), flush=True)
        except Exception as e:  # noqa: BLE001
            d.mkdir(parents=True, exist_ok=True)
            (d / "error.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}))
            print("EPISODE_ERROR", a.task, seed, type(e).__name__, str(e)[:200], flush=True)


def main_libero(a0, flags):
    """LIBERO-PRO episodes of one task, one per --seed (saved initial state); flags = --gen set."""
    from env_libero import LiberoEnv
    for s in a0.seed:
        a = argparse.Namespace(**{**vars(a0), "seed": s})
        d = a.out / f"t{str(a.task).replace(':', '_')}_s{s}"
        if (d / "result.json").exists():
            continue
        try:
            env = LiberoEnv(str(a.task), s)
            D = LiberoDriver(env, d)
            D.system = D.system + _system_addendum("libero", flags)
            H = _make_harness(D.A, libero_profile(), flags)
            S = run_loop(D, H, a.url, a.model, a.thinking, a.max_tokens, replay=_load_replay(a, d.name))
            success = D.A.success()
            from PIL import Image
            Image.fromarray(D.A.rgb()).save(d / "img" / "final.jpg", quality=90)
            res = {"task": a.task, "suite": env.suite_name, "task_id": env.task_id, "seed": a.seed, "instruction": env.instruction,
                   "success": success, "success_ever": S["first_success"] is not None or success, "first_success_s": S["first_success"],
                   "end_reason": S["end"], "calls": S["n_actions"], "rejects": S["rejects"], "seconds": S["seconds"],
                   "llm_s": round(S["llm_s"], 1), "perception": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in D.A.stats.items()},
                   "model": a.model, "url": a.url, "thinking": a.thinking, "objects": list(D.A.names), "infra_error": S.get("infra_error"),
                   "harness": "unified"}
            (d / "episode.json").write_text(json.dumps({**res, "system": D.system, "steps": S["steps"], "usage": S["usage"]}, default=jdefault))
            (d / "result.json").write_text(json.dumps(res, default=str))
            if flags:
                _tag_result(d, flags)
            print(json.dumps({x: res[x] for x in ("task", "seed", "success", "success_ever", "end_reason", "calls", "seconds")}), flush=True)
            env.close_episode()
        except Exception as e:  # noqa: BLE001
            d.mkdir(parents=True, exist_ok=True)
            (d / "error.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}))
            print("EPISODE_ERROR", a.task, s, type(e).__name__, str(e)[:200], flush=True)


def _tag_result(d, flags):
    """record which harness / flags produced the episode (a side file: result.json / episode.json stay in the old format)."""
    try:
        (d / "harness_info.json").write_text(json.dumps({"harness": "harness/run.py", "gen": sorted(flags), **INFO}, indent=1))
    except Exception:  # noqa: BLE001
        pass


def _load_replay(a, key):
    """--replay DIR: recorded calls of the episode with the same folder name (parity tests, no planner)."""
    if not getattr(a, "replay", None):
        return None
    for f in (Path(a.replay) / key / "episode.json", Path(a.replay) / "episode.json"):
        if f.exists():
            st = json.loads(f.read_text())["steps"]
            return [(s["call"]["name"], s["call"]["arguments"]) for s in st if s.get("call")]
    raise FileNotFoundError(f"no recorded episode for {key} under {a.replay}")


def _parser():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="bench", required=True)
    g = sp.add_parser("guava")
    g.add_argument("--task", required=True)
    g.add_argument("--seed-base", type=int, required=True)
    g.add_argument("--episodes", type=int, default=1)
    g.add_argument("--offset", type=int, default=0)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--sam3", default=os.environ.get("GUAVA_SAM3_URL", "http://127.0.0.1:8314"))
    g.add_argument("--thinking", type=int, default=0)
    lb = sp.add_parser("libero")
    lb.add_argument("--task", required=True)
    lb.add_argument("--seed", type=int, nargs="+", required=True)
    lb.add_argument("--out", type=Path, required=True)
    lb.add_argument("--thinking", type=int, default=1)
    for x in (g, lb):
        x.add_argument("--url", default=_paths.DEFAULT_URL)
        x.add_argument("--model", default="qwen")
        x.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096)
        x.add_argument("--replay", default=None)
        x.add_argument("--xml-calls", dest="xml_calls", type=int, default=0)
    return ap


def main(argv):
    """CLI (through run.py): guava|libero [--gen default|none|all|f1,f2] <arguments of _parser()>."""
    gen = os.environ.get("HARNESS_GEN", GEN_DEFAULT)
    if "--gen" in argv:
        i = argv.index("--gen")
        gen = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    flags = gen_flags(gen, argv[0])
    a = _parser().parse_args(argv)
    global XML_CALLS
    XML_CALLS = bool(a.xml_calls)
    if a.bench == "guava":
        main_guava(a, flags)
    elif a.bench == "libero":
        main_libero(a, flags)
    else:
        raise SystemExit(f"unknown benchmark {a.bench}")
