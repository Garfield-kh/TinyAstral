"""harness shared tool layer. Generic primitives, arguments = object + relation + constraint; all geometry is computed
by the tool (from the adapter's current state / asset annotations), never taken from a recorded trajectory.
No task names, no per-task parameters.

  grasp(arm, obj, contact=None, pre_dist=0.1, depth=0, grip=0.0)       approach -> close -> graded hold check
  place(arm, obj, target, fp=None, pre_dist=0.1, dist=0.02, release=True, constraint={})
        target = {"ref": object or None, "rel": pose7 in the ref frame}; the held object's functional point goes there
  transport(arm, obj, target)            object target pose -> hand pose through the measured object-in-hand T_GO
  articulate(arm, obj, joint, dq)        closed-loop arc / slide of a held handle about / along the joint
  move_pose(arm, target) / move_delta(arm, dx, dy, dz, quat=None, axis="world") / gripper(arm, val) / home(arm)
  bimanual(plans)                        any two of the above executed together (index-paired)

Every execution returns per arm: ok, code, pose_err_cm, rot_err_deg (hand vs its last target), obj_motion, held, extra.
Codes: OK, PLAN_FAIL, POSE_ERR, EMPTY_GRASP, SLIPPED, NOT_HELD (dependent action cancelled), RELEASE_FAIL, ARTIC_SHORT.
command success is never reported as task / subgoal success."""
from __future__ import annotations

import numpy as np
import transforms3d as t3d

import geom as Gm

POS_TOL, ROT_TOL = 0.02, 10.0
GAP_MIN = 0.02             # measured finger opening (fraction of full stroke) after closing below this = empty
MICRO_LIFT = 0.005         # hold verification lift (m), undone afterwards
FOLLOW_MIN = 0.0025
RETRY = 2                  # re-plans after a planner failure / re-grasps after a failed hold check


def resolve(A, target, centers=None):
    """target: pose7 | {ref, rel} (ref-frame pose) | {ref, offset_cm, approach bins} (fingertip target, world axes,
    offset from the reference object's centre as given by perception `centers`, default = adapter boxes)."""
    if isinstance(target, dict) and "offset_cm" in target:
        w = Gm.ref_target_to_world(target, centers or A.centers())
        return A.hand_from_tcp(w[:3], w[3:]) if "roll_deg" in target else w
    if isinstance(target, dict):
        return Gm.compose(A.obj_pose(target["ref"]), target["rel"]) if target.get("ref") else np.asarray(target["rel"], float)
    return np.asarray(target, float)


def _motion(s0, s1):
    out = {}
    for n, o in s1["objects"].items():
        d = float(np.linalg.norm(np.asarray(o["pose"][:3]) - np.asarray(s0["objects"][n]["pose"][:3])))
        dq = [round(b - a, 3) for a, b in zip(s0["objects"][n]["q"], o["q"]) if abs(b - a) > 0.01]
        if d > 0.005 or dq:
            out[n] = {"moved_cm": round(d * 100, 1), **({"dq": dq} if dq else {})}
    return out


class Tools:
    def __init__(self, A):
        self.A = A
        self.holding = {}       # arm -> obj (verified)
        self.T_GO = {}          # arm -> pose7 of the held object in the hand frame
        self.cancelled = {}     # arm -> obj whose grasp failed (place / transport of it are cancelled)
        self.log = []

    # ------------------------------------------------------------------ plans
    def grasp(self, arm, obj, contact=None, pre_dist=0.1, depth=0.0, grip=0.0):
        acts = None
        for _ in range(1 + RETRY):
            acts = self.A.grasp_plan(obj, arm, pre_dist, depth, grip, contact)
            if acts is not None:
                break
        if acts is None:
            return {"arm": arm, "tool": "grasp", "obj": obj, "skip": "PLAN_FAIL"}
        return {"arm": arm, "tool": "grasp", "obj": obj, "actions": acts, "verify": obj,
                "regen": (arm, obj, contact, pre_dist, depth, grip)}

    def _regrasp(self, p):
        """failed hold: open, back off, re-plan the grasp from the object's current pose, retry (RETRY times)."""
        arm, obj, contact, pre_dist, depth, grip = p["regen"]
        code = "PLAN_FAIL"
        for i in range(RETRY):
            self.A.run({arm: [{"type": "grip", "val": 1.0}]})
            self.A.run({arm: [{"type": "move", "pose": Gm.translate(self.A.hand_pose(arm), [0, 0, 0.05]).tolist()}]})
            acts = self.A.grasp_plan(obj, arm, max(pre_dist, 0.05), depth, grip, contact if i == 0 else None)
            if acts is None:
                continue
            for d in acts:
                self._run_retry({arm: [d]})
            code, ex = self._verify(arm, obj, False)
            if code == "OK":
                return code, {**ex, "regrasps": i + 1}
        return code, {"regrasps": RETRY}

    def _run_retry(self, items):
        ok = self.A.run(items)
        n = 0
        while not ok and n < RETRY:
            n += 1
            ok = self.A.run(items)
        return ok

    def place(self, arm, obj, target, fp=None, pre_dist=0.1, dist=0.02, release=True, constraint=None):
        if self.cancelled.get(arm) == obj:
            return {"arm": arm, "tool": "place", "obj": obj, "skip": "NOT_HELD"}
        acts = self.A.place_plan(obj, arm, resolve(self.A, target), fp, pre_dist, dist, release, constraint or {})
        if acts is None:
            return {"arm": arm, "tool": "place", "obj": obj, "skip": "PLAN_FAIL"}
        return {"arm": arm, "tool": "place", "obj": obj, "actions": acts, "release": obj if release else None}

    def transport(self, arm, obj, target):
        if self.holding.get(arm) != obj:
            return {"arm": arm, "tool": "transport", "obj": obj, "skip": "NOT_HELD"}
        hand = Gm.p7(Gm.mat(resolve(self.A, target)) @ Gm.inv(Gm.mat(self.T_GO[arm])))
        return {"arm": arm, "tool": "transport", "obj": obj, "actions": [{"type": "move", "pose": hand.tolist()}]}

    def move_pose(self, arm, target, constraint=None):
        d = {"type": "move", "pose": resolve(self.A, target).tolist()}
        if constraint is not None:
            d["constraint"] = constraint
        return {"arm": arm, "tool": "move_pose", "actions": [d]}

    def move_delta(self, arm, dx=0.0, dy=0.0, dz=0.0, quat=None, axis="world"):
        h = self.A.hand_pose(arm).copy()
        if axis == "world":
            h[:3] += [dx, dy, dz]
        else:
            v = Gm.mat(h)[:3, 0]
            h[:3] += -dz * v / np.linalg.norm(v)
        if quat is not None:
            h[3:] = quat
        return {"arm": arm, "tool": "move_delta", "actions": [{"type": "move", "pose": h.tolist()}]}

    def gripper(self, arm, val):
        return {"arm": arm, "tool": "gripper", "actions": [{"type": "grip", "val": float(val)}]}

    def home(self, arm):
        return {"arm": arm, "tool": "home", "actions": [{"type": "move", "pose": self.A.home_pose(arm)}]}

    # ------------------------------------------------------------------ checks
    def _verify(self, arm, obj, co_closing):
        gap = self.A.opening(arm)
        held = self.A.held()
        contact = held.get(arm) == obj
        other = "left" if arm == "right" else "right"
        ex = {"opening": round(gap, 3), "contact": contact}
        if gap < GAP_MIN and not contact:
            return "EMPTY_GRASP", ex
        if not (self.A.is_articulated(obj) or co_closing or held.get(other) == obj or self.holding.get(other) == obj):
            z0 = self.A.obj_pose(obj)[2]
            h = self.A.hand_pose(arm)
            self.A.run({arm: [{"type": "move", "pose": Gm.translate(h, [0, 0, MICRO_LIFT]).tolist()}]})
            ex["follow_cm"] = round((self.A.obj_pose(obj)[2] - z0) * 100, 2)
            self.A.run({arm: [{"type": "move", "pose": list(h)}]})
            follow = ex["follow_cm"] / 100 >= FOLLOW_MIN
            if not (follow or contact):
                return "SLIPPED", ex
        elif not contact and gap < 2 * GAP_MIN:
            return "SLIPPED", ex
        return "OK", ex

    def _register_close(self, arm, obj, code):
        if code == "OK":
            self.holding[arm] = obj
            self.T_GO[arm] = Gm.rel(self.A.hand_pose(arm), self.A.obj_pose(obj))
            self.cancelled.pop(arm, None)
        else:
            self.cancelled[arm] = obj
            self.holding.pop(arm, None)

    # ------------------------------------------------------------------ execution
    def execute(self, plans):
        """plans: list of plan dicts (at most one per arm), executed together index-paired."""
        s0 = self.A.snapshot()
        res = {p["arm"]: {"tool": p["tool"], "ok": True, "code": "OK", "extra": {}} for p in plans}
        live = []
        for p in plans:
            if p.get("skip"):
                res[p["arm"]].update(ok=False, code=p["skip"])
            else:
                live.append(p)
        n = max([len(p["actions"]) for p in live], default=0)
        last_target = {}
        for k in range(n):
            items = {p["arm"]: [p["actions"][k]] for p in live if k < len(p["actions"])}
            ok = self._run_retry(items)
            for a, L in items.items():
                if L[0]["type"] == "move":
                    last_target[a] = L[0]["pose"]
                if not ok:
                    res[a].update(ok=False, code="PLAN_FAIL")
            for p in live:
                if k < len(p["actions"]) and p["actions"][k]["type"] == "grip":
                    a, v = p["arm"], p["actions"][k]["val"]
                    if p.get("verify") and v < 0.5:
                        co = any(q is not p and q.get("verify") == p["verify"] for q in live)
                        code, ex = self._verify(a, p["verify"], co)
                        if code != "OK" and p.get("regen"):
                            code, ex2 = self._regrasp(p)
                            ex.update(ex2)
                        self._register_close(a, p["verify"], code)
                        res[a]["extra"].update(ex)
                        if code != "OK":
                            res[a].update(ok=False, code=code)
                    elif v < 0.5:
                        h = self.A.held().get(a)
                        if h and self.A.opening(a) >= GAP_MIN:
                            self.holding[a] = h
                            self.T_GO[a] = Gm.rel(self.A.hand_pose(a), self.A.obj_pose(h))
                    else:
                        obj = self.holding.pop(a, None)
                        self.T_GO.pop(a, None)
                        self.cancelled.pop(a, None)
                        if obj and self.A.held().get(a) == obj and not self.A.is_articulated(obj):
                            res[a].update(ok=False, code="RELEASE_FAIL")
        s1 = self.A.snapshot()
        for a, t in last_target.items():
            pe = round(Gm.pos_err(s1["hands"][a], t) * 100, 2)
            re = round(Gm.rot_err_deg(s1["hands"][a], t), 1)
            res[a].update(pose_err_cm=pe, rot_err_deg=re)
            if res[a]["ok"] and (pe > POS_TOL * 100 or re > ROT_TOL):
                res[a].update(ok=False, code="POSE_ERR")
        for a in res:
            res[a]["held"] = s1["held"].get(a)
        mo = _motion(s0, s1)
        out = {"arms": res, "obj_motion": mo}
        self.log.append(out)
        return out

    def articulate(self, arm, obj, joint, dq, max_step=(0.12, 0.03), iters=40):
        """closed loop on the measured joint value; hand rotated about (revolute) / slid along (prismatic) the joint."""
        s0 = self.A.snapshot()
        if self.holding.get(arm) != obj:
            return {"arms": {arm: {"tool": "articulate", "ok": False, "code": "NOT_HELD"}}, "obj_motion": {}}
        J = self.A.joints(obj)[joint]
        q0, qt = J["q"], J["q"] + dq
        tol = 0.02 if J["type"] == "revolute" else 0.004
        ok = True
        for _ in range(iters):
            J = self.A.joints(obj)[joint]
            d = qt - J["q"]
            if abs(d) < tol:
                break
            lim = max_step[0] if J["type"] == "revolute" else max_step[1]
            st = float(np.clip(d, -lim, lim))
            h = self.A.hand_pose(arm)
            nh = Gm.rot_about_axis(h, J["origin"], J["axis"], st) if J["type"] == "revolute" else Gm.translate(h, np.asarray(J["axis"]) * st)
            ok = self.A.run({arm: [{"type": "move", "pose": nh.tolist()}]})
            if not ok or abs(self.A.joints(obj)[joint]["q"] - J["q"]) < 1e-3:
                break
        qa = self.A.joints(obj)[joint]["q"]
        short = abs(qt - qa) > max(3 * tol, 0.15 * abs(dq))
        code = "PLAN_FAIL" if not ok else ("ARTIC_SHORT" if short else "OK")
        out = {"arms": {arm: {"tool": "articulate", "ok": code == "OK", "code": code,
                              "extra": {"dq_asked": round(dq, 3), "dq_actual": round(qa - q0, 3)}}},
               "obj_motion": _motion(s0, self.A.snapshot())}
        self.log.append(out)
        return out


# ====================================================================================== shared object-level skills
_motion_orig = _motion


def _motion_safe(s0, s1):
    """objects first seen after s0 have no reference pose: ignore them (perception-backed snapshots can differ)."""
    return _motion_orig(s0, {**s1, "objects": {n: o for n, o in s1["objects"].items() if n in s0["objects"]}})


class SkillTools(Tools):
    """Object-level skills on top of Tools, shared by every backend (single_arm/planner.py). Nothing here knows a task or a
    benchmark; the adapter supplies perception / motion / embodiment constants through its profile `prof`:
      closed_w, open_w     finger width (m) below which the gripper closed on nothing / above which it is open
      retry                re-sends of a motion the controller reported as failed
      regrasp              variants tried after a failed hold check (yaw offset in degrees, or "next" = next candidate)
      backoff              lift (m) after opening a failed grasp
      retreat              release back-off steps (m)
    Capabilities are detected on the adapter: grasp_candidates + feasible (IK-checked grasp candidates, support-surface
    placement), quat_for (arbitrary gripper orientations), several arms.
    Verification = lift test from perception (the object must rise with the gripper); in-hand object geometry (T_GO
    generalised to point clouds) is recorded at every verified grasp and used to place / measure held objects."""
    LIFT_TEST = 0.05

    def __init__(self, A, prof):
        super().__init__(A)
        self.P = dict(prof)
        self.arms = tuple(self.P.get("arms", ("right",)))
        self.single = len(self.arms) == 1
        self.lift_test = True
        self.hold = {}             # arm -> in-hand geometry of the held object (hand frame)
        self.pressed = set()
        self._tried = {}           # arm -> candidate keys tried in the current grasp

    # ---------------------------------------------------------------- adapter access (single-arm adapters take no arm)
    def W(self, arm):
        return self.A.width() if self.single else self.A.width(arm)

    def mv(self, arm, p, q=None):
        return self.A.move(p, q) if self.single else self.A.move(p, q, arm=arm)

    def _run_retry(self, items):
        ok = self.A.run(items)
        n = 0
        while not ok and n < self.P.get("retry", 2):
            n += 1
            ok = self.A.run(items)
        return ok

    def execute(self, plans):
        g = globals()
        orig, g["_motion"] = g["_motion"], _motion_safe
        try:
            out = super().execute(plans)
        finally:
            g["_motion"] = orig
        for p in plans:
            a = out["arms"].get(p["arm"], {})
            if p.get("verify") and a.get("code") == "POSE_ERR":    # the lift test moves the hand on purpose
                a.update(ok=True, code="OK")
            if p.get("info") and p["arm"] in out["arms"]:
                out["arms"][p["arm"]].setdefault("extra", {}).update(p["info"])
        return out

    # ---------------------------------------------------------------- grasp
    def grasp(self, arm, obj, contact=None, pre_dist=0.1, depth=0.0, grip=0.0):
        if not (hasattr(self.A, "grasp_candidates") and hasattr(self.A, "feasible")):
            return super().grasp(arm, obj, contact, pre_dist, depth, grip)
        self._tried[arm] = set()
        g = self._candidate_plan(arm, obj, contact)
        if "skip" in g:
            return {"arm": arm, "tool": "grasp", "obj": obj, **g}
        return {"arm": arm, "tool": "grasp", "obj": obj, "actions": g["actions"], "verify": obj, "info": g["info"],
                "regen": (arm, obj, contact, pre_dist, depth, grip)}

    def _candidate_plan(self, arm, obj, part, own_side=False, max_tries=24):
        """first IK-feasible (pre-grasp, grasp) pair among the adapter's ranked candidates (shape analysis of the point
        cloud), skipping candidates already tried in this grasp."""
        res = self.A.grasp_candidates(obj, arm, part, own_side=own_side)
        if res.get("skip"):
            return res
        tried, capped = 0, None
        for c in res["cands"]:
            if c["key"] in self._tried.get(arm, set()) or c["key"][0] == capped:
                continue
            tried += 1
            if self.A.feasible(arm, [c["pre"]]) and self.A.feasible(arm, [c["grasp"]]):
                self._tried.setdefault(arm, set()).add(c["key"])
                self.A.last_grasp[arm] = {**c, "obs": res["obs"]}
                return {"actions": [{"type": "grip", "val": 1.0}, {"type": "move", "pose": list(c["pre"])},
                                    {"type": "move", "pose": list(c["grasp"])}, {"type": "grip", "val": 0.0}],
                        "info": c["info"]}
            if tried >= max_tries:
                capped = c["key"][0]          # rest of this spec skipped; later specs still get a try each
        return {"skip": "UNREACHABLE", "why": f"no reachable grasp of {obj} for the {arm} arm ({tried} approaches planned); "
                                              "try the other arm, move the object closer, or another part"}

    def _verify(self, arm, obj, co_closing):
        A = self.A
        if not self.lift_test:            # part fixed to furniture: judged by the finger width
            w = self.W(arm)
            ex = {"opening": round(A.opening(arm), 3), "lift_test": "skipped (fixed part): judged by finger width"}
            return ("OK" if w > self.P["closed_w"] else "EMPTY_GRASP"), ex
        w = self.W(arm)
        ex = {"opening": round(A.opening(arm), 3), "close": getattr(A, "last_close", None)}
        if w <= self.P["closed_w"]:
            return "EMPTY_GRASP", ex
        keep = getattr(A, "keep_robot", None)
        if keep is not None:
            A.keep_robot = True               # the object is in the fingers: robot points are not dropped
        o0 = A.get(obj)
        h = A.hand_pose(arm)
        self.mv(arm, Gm.translate(h, [0, 0, self.LIFT_TEST])[:3])
        o1 = A.get(obj)
        if keep is not None:
            A.keep_robot = keep
        w1 = self.W(arm)
        if w1 <= self.P["closed_w"]:
            return "SLIPPED", {**ex, "lift_test": "fingers closed fully during the lift"}
        if o0 is None or o1 is None or A.perceive([obj], refresh=False)[obj] is None:
            ex["lift_test"] = "object not visible; judged by finger width only"
            return "OK", ex
        dz = float(o1["center"][2] - o0["center"][2])
        ex["rose_cm"] = round(dz * 100, 1)
        return ("OK" if dz >= self.LIFT_TEST * 0.5 else "SLIPPED"), ex

    def _regrasp(self, p):
        arm, obj, contact, pre_dist, depth, grip = p["regen"]
        A = self.A
        if not self.lift_test:            # side / fixed-part grasp: retry once from the current geometry, same approach
            A.run({arm: [{"type": "grip", "val": 1.0}]})
            h = A.hand_pose(arm)
            back = -np.asarray(A.cur_approach_vec if A.cur_approach_vec is not None else [0, 0, -1.0])
            A.run({arm: [{"type": "move", "pose": Gm.translate(h, 0.06 * back).tolist()}]})
            acts = A.grasp_plan(obj, arm, pre_dist, depth, grip, 0.0)
            if acts is None:
                return "PLAN_FAIL", {"regrasps": 1}
            for d in acts:
                A.run({arm: [d]})
            code, ex = self._verify(arm, obj, False)
            return code, {**ex, "regrasps": 1}
        code = "PLAN_FAIL"
        if getattr(A, "inject", None):     # When a failure is injected, skip automatic re-grasp.
            A.run({arm: [{"type": "grip", "val": 1.0}]})
            A.run({arm: [{"type": "move", "pose": Gm.translate(A.hand_pose(arm), [0, 0, 0.08]).tolist()}]})
            return self._verify(arm, obj, False)[0], {"regrasps": 0}
        variants = tuple(self.P.get("regrasp", (90.0,)))
        for i, var in enumerate(variants):
            A.run({arm: [{"type": "grip", "val": 1.0}]})
            A.run({arm: [{"type": "move", "pose": Gm.translate(A.hand_pose(arm), [0, 0, self.P.get("backoff", 0.08)]).tolist()}]})
            if var == "next":
                g = self._candidate_plan(arm, obj, contact)
                acts = g.get("actions")
                if acts is not None:
                    p["info"] = g["info"]
            else:
                acts = A.grasp_plan(obj, arm, pre_dist, depth, grip, var)
            if acts is None:
                continue
            for d in acts:
                A.run({arm: [d]})
            code, ex = self._verify(arm, obj, False)
            if code == "OK":
                return code, {**ex, "regrasps": i + 1}
        return code, {"regrasps": len(variants)}

    def _register_close(self, arm, obj, code):
        super()._register_close(arm, obj, code)
        if code == "OK":
            self.set_hold(arm)
        else:
            self.hold.pop(arm, None)

    # ---------------------------------------------------------------- in-hand object geometry (T_GO on point clouds)
    def set_hold(self, arm):
        """object centre / bottom / inner floor / major axis in the hand frame, from the grasp-time observation and the
        grasp pose (the object moves rigidly with the fingers afterwards)."""
        g = getattr(self.A, "last_grasp", {}).get(arm)
        if not g or g.get("obs") is None:
            self.hold.pop(arm, None)
            return
        o = g["obs"]
        R = Gm.mat(g["grasp"])[:3, :3]
        t = np.asarray(g["grasp"][:3], float)
        c = np.asarray(o["center"], float).copy()
        if str(g["info"].get("grasp", "")).startswith("body centre"):
            c[:2] = g["p"][:2]                 # parallel fingers centre the body between them
        b = np.r_[c[:2], float(o["min"][2])]
        _, u, _ = Gm.top_axes(o["pts"])
        bc = Gm.body_center(o["pts"])
        fz = Gm.floor_z(o["pts"], bc)
        fl = np.r_[bc, fz if fz is not None else float(o["min"][2])]
        self.hold[arm] = {"c": R.T @ (c - t), "b": R.T @ (b - t), "size": np.asarray(o["max"], float) - np.asarray(o["min"], float),
                          "u": R.T @ np.r_[u, 0.0], "floor": R.T @ (fl - t), "top": float(o["max"][2] - t[2])}

    def held_est(self, arm):
        """(centre, bottom z) of the held object predicted from the gripper pose and the in-hand geometry."""
        h = self.hold.get(arm)
        if not h:
            return None
        hp = self.A.hand_pose(arm)
        R = Gm.mat(hp)[:3, :3]
        t = np.asarray(hp[:3], float)
        return t + R @ h["c"], float((t + R @ h["b"])[2])

    def drop_hold(self, arm):
        self.holding.pop(arm, None)
        self.T_GO.pop(arm, None)
        self.hold.pop(arm, None)

    # ---------------------------------------------------------------- push
    def push(self, obj, direction_xy, distance, height=None, arm=None):
        arm = arm or self.arms[0]
        acts = self.A.push_plan(obj, direction_xy, distance, height) if self.single else self.A.push_plan(obj, direction_xy, distance, height, arm=arm)
        if acts is None:
            return {"arm": arm, "tool": "push", "obj": obj, "skip": "NOT_SEEN"}
        return {"arm": arm, "tool": "push", "obj": obj, "actions": acts}

    # ---------------------------------------------------------------- support-surface placement (IK-checked)
    def surface_target(self, arm, obj, ref, relation, xy):
        """(xy, bottom z, why) for the held object's bottom: table point, inner floor of a container (or of a container
        held by the other hand), clear of an arching handle, just above the rim of a deep container, top surface of a
        support; beside a reference for left_of / right_of / front_of / behind."""
        A = self.A
        rel = str(relation or "on").lower().replace(" ", "_")
        if ref in (None, "table") or rel == "table":
            if xy is None:
                raise ValueError("ref 'table' needs xy")
            return np.asarray(xy[:2], float), 0.005, "table point"
        for ha, ho in self.holding.items():
            if ho == ref and ha in self.hold and rel in ("in", "into", "inside", "on", "onto", "on_top_of", "at"):
                hp = A.hand_pose(ha)
                fl = np.asarray(hp[:3]) + Gm.mat(hp)[:3, :3] @ self.hold[ha]["floor"]
                return fl[:2], fl[2] + 0.015, "held container floor"
        r = A.get(ref)
        if r is None:
            raise RuntimeError(f"cannot see {ref}; locate it with a visual description")
        pts = r["pts"]
        size = np.asarray(r["max"]) - np.asarray(r["min"])
        if rel in ("on", "onto", "on_top_of") and size[2] >= 0.02:
            c0 = Gm.body_center(pts)
            f0 = Gm.floor_z(pts, c0, r=0.025)
            if f0 is not None and f0 < float(r["max"][2]) - 0.02 and min(size[:2]) > 0.07:
                rel = "in"                       # concave reference (bowl, basket, pan): "on" means nest inside
        if rel in ("in", "into", "inside"):
            c = Gm.body_center(pts)
            ar = Gm.arch(pts, 0.0)
            why_in = "container floor"
            if ar is not None:                   # keep clear of a handle arching over the middle
                m = np.asarray(ar["m"], float)
                w = Gm.extent((pts[:, :2] - c) @ m)
                est = self.held_est(arm)
                hold_xy = est[0][:2] if est is not None else c
                sgn = 1.0 if float((hold_xy - c) @ m) >= 0 else -1.0
                c = c + sgn * m * 0.26 * w
                why_in = "container floor beside its handle"
            f = Gm.floor_z(pts, c)
            fz = f if f is not None else float(r["min"][2]) + 0.01
            rim = float(np.percentile(pts[:, 2], 70 if ar is not None else 95))
            if rim - fz > 0.06:                  # deep container: release just above the rim and let it drop in
                return c, rim + 0.01, why_in + " (released above the rim)"
            return c, fz + 0.012, why_in
        if rel in ("on", "onto", "on_top_of", "at", "over", "above"):
            c, top = Gm.top_center(pts)
            if size[2] < 0.015:                  # flat target (mat, pad, coaster): centre of the whole footprint
                c = Gm.body_center(pts)
            return c, top + 0.008, "top surface"
        dirs = self.P["dirs"]
        word = {"left_of": "left", "right_of": "right", "front_of": "front", "in_front_of": "front", "behind": "back"}.get(rel)
        if word is None:
            raise ValueError(f"unknown relation {relation}")
        o = A.cache.get(obj)
        half = ((np.asarray(o["max"][:2]) - np.asarray(o["min"][:2])) / 2) if o is not None else np.array([0.03, 0.03])
        v = np.asarray(dirs[word], float)
        d = float(np.abs(size[:2] / 2) @ np.abs(v)) + float(half @ np.abs(v)) + 0.03
        return np.asarray(r["center"][:2]) + v * d, 0.005, "beside"

    def place_surface(self, arm, obj, ref, relation="on", xy=None, release=True, height=0.10, align=False):
        """carry the held object over the target, lower it until its bottom is at the target height and release.
        The hand pose is chosen among yaw turns (aligned with ref's long side when align) and, last, tilts, by the
        arm's motion planner; closed-loop xy correction from the cameras when no in-hand estimate exists."""
        A = self.A
        txy, tbot, why = self.surface_target(arm, obj, ref, relation, xy)
        est = self.held_est(arm)
        if est is not None:
            ow, obot = est
        else:
            o = A.get(obj)
            if o is None:
                return {"code": "NOT_SEEN", "why": f"cannot see the held {obj}"}
            ow, obot = np.asarray(o["center"], float), float(o["min"][2])
        tcp = np.asarray(A.hand_pose(arm)[:3], float)
        if not release:
            tbot = max(tbot, float(height))
        q0 = A.hand_pose(arm)[3:]
        rp = A.cache.get(ref) if ref not in (None, "table") else None
        clear = (float(rp["max"][2]) + (tcp[2] - obot) + 0.04) if rp is not None else 0.0
        yaws = [0, 20, -20, 40, -40, 60, -60, 90, -90]
        if align and arm in self.hold and rp is not None:
            _, ur, _ = Gm.pca2(rp["pts"][:, :2])
            uo = (Gm.mat(A.hand_pose(arm))[:3, :3] @ self.hold[arm]["u"])[:2]
            dl = np.degrees(np.arctan2(ur[1], ur[0]) - np.arctan2(uo[1], uo[0]))
            dl = (dl + 90) % 180 - 90
            yaws = [y for y in (dl + e for e in (0, 180, -180, 8, -8)) if -200 <= y <= 200]
        rdir = Gm.unit(np.r_[txy - np.asarray(self.P["arm_base"][arm], float), 0.0])
        kax = Gm.unit(np.cross([0, 0, 1.0], rdir))
        rots = [(y, 0.0) for y in yaws] + [(0.0, t) for t in (20.0, -20.0, 35.0, -35.0, 45.0, -45.0)]
        bpt = np.r_[ow[:2], obot]
        choice = None
        for yaw, tilt in rots:
            Rm = t3d.axangles.axangle2mat([0, 0, 1.0], np.radians(yaw)) if not tilt else t3d.axangles.axangle2mat(kax, np.radians(tilt))
            q = Gm.quat_mul(np.asarray(t3d.quaternions.mat2quat(Rm)), q0)
            off = Rm @ (ow - tcp)
            boff = Rm @ (bpt - tcp)
            dest = np.r_[txy - off[:2], tbot - boff[2]]
            safe = max(dest[2] + 0.07, clear)
            if not A.feasible(arm, [np.r_[dest, q]]):
                continue
            if not A.feasible(arm, [np.r_[dest[:2], safe, q]]):
                safe = dest[2] + 0.04
                if not A.feasible(arm, [np.r_[dest[:2], safe, q]]):
                    continue
            choice = (yaw if not tilt else f"tilt {tilt}", q, dest, safe)
            break
        if choice is None:
            return {"code": "UNREACHABLE", "why": f"the {arm} arm cannot reach over the target ({why}) with {obj}; "
                                                  "bring the target closer to this arm, use the other arm, or handover"}
        yaw, q, dest, safe = choice
        up = max(tcp[2], safe)
        ok = self.seq(arm, [np.r_[tcp[:2], up, q0], np.r_[tcp[:2], up, q], np.r_[dest[:2], safe, q]])
        if ok == "PLAN_FAIL":
            return {"code": "UNREACHABLE", "why": f"cannot carry {obj} over the target ({why}); try the other arm or handover"}
        corr = []
        for _ in range(0 if arm in self.hold else 2):
            o2 = A.get(obj)
            if o2 is None:
                break
            err = txy - np.asarray(o2["center"][:2])
            if np.linalg.norm(err) < 0.004 or np.linalg.norm(err) > 0.05:
                break
            corr.append(round(float(np.linalg.norm(err)) * 100, 1))
            t = A.hand_pose(arm)
            dest[:2] += err
            self.seq(arm, [np.r_[t[:2] + err, t[2], t[3:]]])
        ok = self.seq(arm, [np.r_[dest, q]])
        out = {"code": "OK" if ok != "PLAN_FAIL" else "PLAN_FAIL", "target": why, "target_xyz": np.r_[txy, tbot],
               "xy_corrections_cm": corr}
        if yaw:
            out["turned_deg"] = yaw if isinstance(yaw, str) else round(float(yaw))
        if release:
            est = self.held_est(arm)
            self.seq(arm, [{"type": "grip", "val": 1.0}])
            self.retreat(arm, self.P.get("place_retreat"))
            self.drop_hold(arm)
            self.cancelled.pop(arm, None)
            if est is not None and obj in A.cache and hasattr(A, "set_prior"):   # dead-reckoned position prior
                A.set_prior(obj, est[0][:2], tbot)
            o3 = A.get(obj)
            if o3 is not None:
                out["object_now"] = {"center": np.asarray(o3["center"]), "bottom_z": float(o3["min"][2])}
        return out

    # ---------------------------------------------------------------- motion helpers for multi-step skills
    def seq(self, arm, poses):
        """run poses (tcp pose7 or action dicts) one by one; stops at the first planner failure (PLAN_FAIL) or a pose
        error > 3 cm (POSE_ERR). Returns the code."""
        for p in poses:
            d = p if isinstance(p, dict) else {"type": "move", "pose": [float(x) for x in p]}
            ok = self._run_retry({arm: [d]})
            if d["type"] == "move":
                if not ok:
                    return "PLAN_FAIL"
                if Gm.pos_err(self.A.hand_pose(arm), d["pose"]) > 0.03:
                    return "POSE_ERR"
        return "OK"

    def retreat(self, arm, steps=None):
        """back off along the approach axis (straight up for a top grasp) in short steps."""
        for dz in (steps or self.P.get("retreat", (0.08,))):
            h = self.A.hand_pose(arm)
            self.seq(arm, [np.r_[h[:3] + [0, 0, dz], h[3:]]])

    # ---------------------------------------------------------------- press / hinged door / hand-over / two-hand grasp
    def press(self, arm, obj):
        A = self.A
        o = A.get(obj)
        if o is None:
            return {"code": "NOT_SEEN"}
        c, top = Gm.top_center(o["pts"])
        p = np.r_[c, top]
        q = A.quat_for([0, 0, -1.0], self.P.get("press_closing", [-1.0, 0, 0]))
        above, down = np.r_[p + [0, 0, 0.08], q], np.r_[p - [0, 0, 0.02], q]
        code = self.seq(arm, [{"type": "grip", "val": 0.0}, above])
        low = None
        if code == "OK":
            A.run({arm: [{"type": "move", "pose": down.tolist()}]})
            low = float(A.hand_pose(arm)[2])
            A.run({arm: [{"type": "move", "pose": above.tolist()}]})
        reached = low is not None and low <= top + 0.05
        if reached:
            self.pressed.add(obj)
        return {"code": "OK" if reached else (code if code != "OK" else "NOT_REACHED"), "pressed_point": p,
                "lowest_gripper_z": low}

    def articulate_hinged(self, arm, obj, motion="open", degrees=80.0, step=6.0):
        """open / close a hinged front door: handle and hinge from the point cloud (geom.door_geometry), grasp the
        handle from the front, then move the gripper along the door arc about the vertical hinge axis in small steps,
        turning the wrist with the door; stops when the arm cannot follow. No joint annotations are used."""
        A = self.A
        o = A.get(obj)
        if o is None:
            return {"code": "NOT_SEEN"}
        d = Gm.door_geometry(o["pts"], 0.0, self.P["forward"])
        f, r = d["forward"], d["lateral"]
        hx, hy, hz = d["handle"]
        hinge = d["hinge"]
        good = self.P["lateral_arm"][-1 if d["hinge_lat"] < d["handle_lat"] else 1]   # the hinge-side arm follows furthest
        if arm not in self.arms or not self.holding.get(good):
            arm = good
        other = [a for a in self.arms if a != arm]
        for x in other:
            if self.holding.get(x) is None:
                self.home(x)
        r0 = np.r_[hx, hy] - hinge
        sign = -1.0 if d["hinge_lat"] < d["handle_lat"] else 1.0
        if float(np.cross(r, f)) < 0:
            sign = -sign
        if str(motion) == "close":
            sign = -sign
        b = np.radians(60)
        a = np.r_[f * np.cos(b), -np.sin(b)]
        best = None
        for dz in (0.0, -0.02, -0.04):
            p = np.r_[np.r_[hx, hy] - f * 0.005, hz + dz]
            for c in (np.r_[r, 0.0], np.r_[-r, 0.0]):
                q = A.quat_for(a, c)
                pre, hg = np.r_[p - a * 0.06, q], np.r_[p, q]
                if A.feasible(arm, [pre]) and A.feasible(arm, [hg]):
                    best = (p, q, pre, hg)
                    break
            if best:
                break
        if best is None:
            return {"code": "UNREACHABLE", "arm": arm, "why": f"cannot reach the door handle of {obj} with the {arm} arm; try the other arm"}
        p, q, pre, hg = best
        self.seq(arm, [{"type": "grip", "val": 1.0}, pre, hg, {"type": "grip", "val": 0.0}])
        steps, done = int(degrees // step), 0
        for k in range(1, steps + 1):
            th = sign * np.radians(step * k)
            Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            xy = hinge + Rz @ r0
            qk = Gm.quat_mul(Gm.yaw_quat(th), q)
            if self.seq(arm, [np.r_[xy, p[2], qk]]) == "PLAN_FAIL":
                break
            done = int(step * k)
        self.seq(arm, [{"type": "grip", "val": 1.0}])
        h = A.hand_pose(arm)
        self.seq(arm, [np.r_[h[:3] + np.r_[-f * 0.06, 0.04], h[3:]]])
        out = {"code": "EXECUTED" if done >= 50 else "SHORT", "arm": arm, "swing_deg": done, "handle_found": d["found"],
               "note": "door angle is not measured; judge it from the images"}
        if arm != good:
            out["hint"] = f"the door swings toward the {good} side; the {good} arm can follow it further"
        return out

    def home(self, arm):
        return self.execute([{"arm": arm, "tool": "home", "actions": [{"type": "home"}]}])

    def bimanual_grasp(self, calls):
        """one grasp per arm executed together; for one shared object each hand takes a grasp on its own half, both
        close, then a joint 5 cm lift test (judged by finger opening when the camera rise is not confirmed)."""
        A = self.A
        objs = {x["arm"]: x["object"] for x in calls}
        shared = len(set(objs.values())) == 1
        plans, out = {}, {}
        for x in calls:
            self._tried[x["arm"]] = set()
            g = self._candidate_plan(x["arm"], x["object"], x.get("part"), own_side=shared)
            if "skip" in g:
                out[x["arm"]] = {"code": g["skip"], "why": g.get("why")}
            else:
                plans[x["arm"]] = g
        if out:
            return {**out, **{a: {"code": "NOT_RUN", "why": "the other arm had no reachable grasp"} for a in plans}}
        n = max(len(g["actions"]) for g in plans.values())
        for k in range(n - 1):
            self._run_retry({a: [g["actions"][k]] for a, g in plans.items() if k < len(g["actions"]) - 1})
        A.run({a: [{"type": "grip", "val": 0.0}] for a in plans})
        codes, ex = {}, {}
        for a in plans:
            ex[a] = {"opening": round(A.opening(a), 3)}
            codes[a] = "EMPTY_GRASP" if self.W(a) <= self.P["closed_w"] else "OK"
        live = [a for a in plans if codes[a] == "OK"]
        if live:
            if getattr(A, "keep_robot", None) is not None:
                A.keep_robot = True
            o0 = {objs[a]: A.get(objs[a]) for a in live}
            A.run({a: [{"type": "move", "pose": Gm.translate(A.hand_pose(a), [0, 0, self.LIFT_TEST]).tolist()}] for a in live})
            o1 = {o: A.get(o) for o in set(objs[a] for a in live)}
            if getattr(A, "keep_robot", None) is not None:
                A.keep_robot = False
            for a in live:
                o = objs[a]
                if self.W(a) <= self.P["closed_w"]:
                    codes[a] = "SLIPPED"
                    continue
                if o0.get(o) is None or o1.get(o) is None:
                    ex[a]["lift_test"] = "object not visible; judged by finger opening"
                    continue
                dz = float(o1[o]["center"][2] - o0[o]["center"][2])
                ex[a]["rose_cm"] = round(dz * 100, 1)
                if dz < self.LIFT_TEST * 0.5:
                    codes[a] = "SLIPPED"
                    if shared:            # one object, two hands: the camera rise test is unreliable
                        codes[a] = "OK"
                        ex[a]["note"] = "held by both grippers (finger opening); visual rise not confirmed"
        for a, g in plans.items():
            if codes[a] == "OK":
                self.holding[a] = objs[a]
                self.set_hold(a)
            out[a] = {"code": codes[a], **g["info"], **ex[a]}
        return out
