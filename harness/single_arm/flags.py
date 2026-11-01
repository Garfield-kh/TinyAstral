"""flags (--gen): benchmark-agnostic tool-layer improvements for the single-arm Guava / LIBERO path.

GenHarness(A, prof, flags) is planner.UnifiedHarness plus switchable additions. flags = set() -> no method of this file
changes anything: every override returns the parent's call untouched and self.T stays the parent's SkillTools, so the result
stream is byte-identical to UnifiedHarness.

Flags (name, what the flag changes):
  post_release  after place releases, the object is re-measured by the cameras: off the target ->
                OFF_TARGET {dx_cm, dy_cm}; a tall object that lost a third of its height -> OFF_TARGET (fell over); a re-measure
                > 10 cm away AND implausible (bottom > 3 cm up, footprint ratio outside 0.55..1.8) -> MEASURE_UNSURE, ok=null;
                not found -> RELEASED_NOT_SEEN, ok=null. release: object_now / not-seen note only (no target to compare).
  honest_done   done rejected only because of ok=null checks gets its own wording; a
                predicate on an object let go > 10 cm away from its release point, implausible and untouched since -> ok=null;
                a grasp judged OK from the finger width alone (object not visible in the lift test) says so (held_check).
  inst_lock     grasp locks onto the instance the planner last located (until something moves it); a fresh
                detection > 5 cm away -> look once more around the located position (adapter instances + position prior); not
                found there -> TARGET_MOVED, nothing executed. Needs adapter .instances and .track (LIBERO); else parent.
  held_track    at a verified grasp the in-hand geometry is recorded from the hand pose measured at closing and the
                pre-grasp observation; place targets the held object through hand pose x in-hand transform instead of a fresh
                (gripper-occluded) segmentation; the height predicate uses it too.
  step_open     release opens just past the contact opening, backs off straight up, then opens fully. Needs an
                adapter that can open to a given width (.grip_to(arm, opening)); Guava (binary grasp()/release()) and LIBERO
                (incremental robosuite command) have none -> parent order.
  fail_post     SLIPPED -> lower back to the closing height before the fingers open (no drop from
                the lift-test height); a hold the parent already judged failed with the fingers still fully open after
                closing -> NOT_CLOSED (refines the code only, never overrides the parent's OK; re-grasp as any failed hold);
                place whose carry / lowering failed far from the target (hand > 3 cm off in xy) -> the object is NOT released,
                PLAN_FAIL, still held.
No task names, no simulator truth: only adapter perception / proprioception calls (perceive, get, cache, first, instances,
hand_pose, opening, width, run, move)."""
from __future__ import annotations

import numpy as np

import planner  # noqa: E402  (first: sets sys.path for geom / tools)
import geom as Gm  # noqa: E402
import tools  # noqa: E402
from planner import Unknown  # noqa: E402

FLAGS = ("post_release", "honest_done", "inst_lock", "held_track", "step_open", "fail_post")
DEFAULT_ON = ("post_release", "honest_done", "fail_post")      # recommendation; the entry decides
T_FLAGS = {"post_release", "held_track", "step_open", "fail_post"}   # flags that need GenSkillTools

LOCK_M = 0.05            # inst_lock: radius within which a located instance stays the same one (m)
JUMP_M = 0.10            # a re-measurement further than this from the release point is treated as a jump (m)
OFF_TABLE_M = 0.03       # height above the table below which a point counts as on the surface (m)
OFF_BESIDE_M = 0.04
NOTCLOSED_FRAC = 0.97    # opening fraction above this after closing = the fingers never closed on anything
PART_OPEN = 0.25         # opening fraction treated as a partially open gripper
FAIL_XY_M = 0.03


def _implausible(o, size0):
    """metres: not the same object resting somewhere -- bottom well above the table, or footprint far off its size."""
    if o is None:
        return False
    if float(o["min"][2]) > 0.03:
        return True
    s0 = np.asarray(size0 if size0 is not None else [0, 0, 0], float)
    a = float(np.max((np.asarray(o["max"]) - np.asarray(o["min"]))[:2]))
    b = float(np.max(s0[:2]))
    return b > 0.01 and not (0.55 <= a / b <= 1.8)


def _size(o):
    return None if o is None else (np.asarray(o["max"], float) - np.asarray(o["min"], float))


# ====================================================================================== tool layer
class GenSkillTools(tools.SkillTools):
    def __init__(self, A, prof, flags):
        super().__init__(A, prof)
        self.flags = set(flags)
        self.last_place = None        # {"arm", "obj", "target"(object-centre xyz), "hand_target"}
        self._close = {}              # arm -> (hand pose at closing, pre-grasp observation)
        self._skip_rel = None         # fail_post: release skipped in the current execute
        self._fail_seen = False
        self._pend_full = None
        self.gen_notes = {}           # arm -> extra fields of the last execute (merged by GenHarness)

    def _on(self, f):
        return f in self.flags

    # ---------------------------------------------------------------- place: record target, held_track, release markers
    def place(self, arm, obj, target, fp=None, pre_dist=0.1, dist=0.02, release=True, constraint=None):
        if not (self.flags & T_FLAGS):
            return super().place(arm, obj, target, fp, pre_dist, dist, release, constraint)
        if self.cancelled.get(arm) == obj:
            return {"arm": arm, "tool": "place", "obj": obj, "skip": "NOT_HELD"}
        w = tools.resolve(self.A, target)
        est = self.held_center(arm) if self._on("held_track") else None
        if est is not None:
            acts = self._plan_with_estimate(obj, arm, w, est, fp, pre_dist, dist, release, constraint or {})
        else:
            acts = self.A.place_plan(obj, arm, w, fp, pre_dist, dist, release, constraint or {})
        self.last_place = {"arm": arm, "obj": obj, "target": np.asarray(w[:3], float).copy(), "held_track": est is not None}
        if acts is None:
            return {"arm": arm, "tool": "place", "obj": obj, "skip": "PLAN_FAIL"}
        if release and (self._on("fail_post") or self._step_open_ok()):
            acts = [dict(d) for d in acts]
            i = next((k for k, d in enumerate(acts) if d["type"] == "grip" and d["val"] >= 0.5), None)
            if i is not None:
                acts[i]["gen_release"] = True
                if self._step_open_ok():
                    acts.append({"type": "grip", "val": 1.0, "gen_full": True})
        p = {"arm": arm, "tool": "place", "obj": obj, "actions": acts, "release": obj if release else None}
        if est is not None:
            p["info"] = {"gen_held_track": {"object_est": [round(float(x), 3) for x in est]}}
        return p

    def _plan_with_estimate(self, obj, arm, w, est, *a):
        """adapter place_plan with the held object's observation centred on the tracked estimate (the plan's TCP-object
        offset then comes from the in-hand transform, not from the occluded segmentation)."""
        A = self.A
        o = A.cache.get(obj)
        if o is None:
            return A.place_plan(obj, arm, w, *a)
        fake = dict(o, center=np.asarray(est, float))
        orig = A.__dict__.get("get")
        A.get = lambda name, refresh=True, _g=type(A).get: fake if name == obj else _g(A, name, refresh)
        try:
            return A.place_plan(obj, arm, w, *a)
        finally:
            if orig is None:
                del A.get
            else:
                A.get = orig

    def _step_open_ok(self):
        return self._on("step_open") and hasattr(self.A, "grip_to")

    # ---------------------------------------------------------------- execution hooks
    def execute(self, plans):
        if not (self.flags & T_FLAGS):
            return super().execute(plans)
        self._skip_rel, self._fail_seen, self._pend_full = None, False, None
        self.gen_notes = {}
        saved = {p["arm"]: (self.holding.get(p["arm"]), self.T_GO.get(p["arm"]), self.hold.get(p["arm"])) for p in plans}
        marked = any(d.get("gen_release") for p in plans for d in p.get("actions", []))
        out = super().execute(plans)
        if not marked:
            return out
        for p in plans:
            a = p["arm"]
            r = out["arms"].get(a)
            if r is None:
                continue
            if self._skip_rel == a:                       # fail_post: release refused -> still holding
                h, tg, hd = saved[a]
                if h is not None:
                    self.holding[a] = h
                    if tg is not None:
                        self.T_GO[a] = tg
                    if hd is not None:
                        self.hold[a] = hd
                r.update(ok=False, code="PLAN_FAIL")
                r.setdefault("extra", {})["gen_fail_post"] = {"released": False, **self.gen_notes.get(a, {})}
            elif self.gen_notes.get(a, {}).get("step_open") and r.get("code") == "RELEASE_FAIL" and self.A.held().get(a) is None:
                # the partial opening was what the release check saw; fully open now and nothing held -> parent codes
                pe, re_ = r.get("pose_err_cm", 0.0), r.get("rot_err_deg", 0.0)
                code = "PLAN_FAIL" if self._fail_seen else ("POSE_ERR" if (pe > tools.POS_TOL * 100 or re_ > tools.ROT_TOL) else "OK")
                r.update(ok=code == "OK", code=code)
                r.setdefault("extra", {})["gen_step_open"] = self.gen_notes[a]["step_open"]
            elif self.gen_notes.get(a, {}).get("step_open"):
                r.setdefault("extra", {})["gen_step_open"] = self.gen_notes[a]["step_open"]
        return out

    def _run_retry(self, items):
        if not (self.flags & T_FLAGS):
            return super()._run_retry(items)
        rel = [(a, L[0]) for a, L in items.items() if L and (L[0].get("gen_release") or L[0].get("gen_full"))]
        if not rel:
            if self._skip_rel is not None and self._skip_rel in items:    # rest of a refused release: nothing moves
                return True
            ok = super()._run_retry(items)
            if not ok:
                self._fail_seen = True
            return ok
        a, d = rel[0]
        if d.get("gen_full"):
            if self._pend_full == a:
                self._pend_full = None
                return super()._run_retry({a: [{"type": "grip", "val": 1.0}]})
            return True
        if self._on("fail_post") and self._fail_seen:
            last = self._last_move.get(a)
            if last is not None:
                err = float(np.linalg.norm(np.asarray(self.A.hand_pose(a)[:2]) - np.asarray(last[:2])))
                if err > FAIL_XY_M:
                    self._skip_rel = a
                    self.gen_notes[a] = {"why": f"the hand stopped {err * 100:.0f} cm from the drop point; the object was not let go",
                                         "hand_off_cm": round(err * 100, 1)}
                    return False
        if self._step_open_ok():
            o = float(self.A.opening(a))
            if o + PART_OPEN < 1.0 and o < 0.75:
                for v in (min(1.0, o + 0.05), min(1.0, o + PART_OPEN)):
                    self.A.grip_to(a, v)
                self._pend_full = a
                self.gen_notes[a] = {"step_open": "partial open, back off up, then open fully"}
                return True
        return super()._run_retry({a: [{k: v for k, v in d.items() if k != "gen_release"}]})

    @property
    def _last_move(self):
        lm = {}
        for e in getattr(self, "_cur_moves", []):
            lm[e[0]] = e[1]
        return lm

    # Tools.execute does not expose the last move target before a grip; record it through A.run's items
    def _note_moves(self, items):
        for a, L in items.items():
            if L and L[0].get("type") == "move":
                self.__dict__.setdefault("_cur_moves", []).append((a, L[0]["pose"]))

    # ---------------------------------------------------------------- grasp checks (held_track, fail_post, honest held_check)
    def _verify(self, arm, obj, co_closing):
        if not (self.flags & T_FLAGS):
            return super()._verify(arm, obj, co_closing)
        A = self.A
        h0 = np.asarray(A.hand_pose(arm), float).copy()
        pre = A.cache.get(obj)
        self._close[arm] = (h0, None if pre is None else dict(pre))
        open0 = float(A.opening(arm))
        code, ex = super()._verify(arm, obj, co_closing)
        if self._on("fail_post") and code == "SLIPPED" and self.lift_test and "closed fully" not in str(ex.get("lift_test", "")):
            h1 = A.hand_pose(arm)
            if float(h1[2] - h0[2]) > 0.01:              # lower back to the closing height before anything opens
                self.mv(arm, h0[:3] + [0, 0, 0.005])
                ex = {**ex, "gen_fail_post": "lowered back before opening"}
        if self._on("fail_post") and self.lift_test and code != "OK" and open0 > NOTCLOSED_FRAC:
            # refine only: never overrides the parent's OK (a wide object can be held with the fingers ~fully open)
            code = "NOT_CLOSED"
            ex = {**ex, "gen_fail_post": "the fingers stayed fully open: nothing between them"}
        return code, ex

    def _register_close(self, arm, obj, code):
        super()._register_close(arm, obj, code)
        if code == "OK" and self._on("held_track") and arm not in self.hold:
            h0, pre = self._close.get(arm, (None, None))
            if h0 is not None and pre is not None:
                R = Gm.mat(h0)[:3, :3]
                t = h0[:3]
                c = np.asarray(pre["center"], float)
                b = np.r_[c[:2], float(pre["min"][2])]
                sz = _size(pre)
                self.hold[arm] = {"c": R.T @ (c - t), "b": R.T @ (b - t), "size": sz, "u": np.zeros(3),
                                  "floor": R.T @ (b - t), "top": float(pre["max"][2] - t[2]), "gen": True}

    def held_center(self, arm):
        e = self.held_est(arm)
        return None if e is None else np.asarray(e[0], float)


def _wrap_run(T):
    """route A.run through GenSkillTools so the last commanded move per arm is known at a release (fail_post)."""
    A = T.A
    orig = A.run

    def run(items):
        T._note_moves(items)
        return orig(items)
    return run


# ====================================================================================== semantic layer
class GenHarness(planner.UnifiedHarness):
    def __init__(self, A, prof, flags=frozenset()):
        super().__init__(A, prof)
        self.flags = set(flags or ())
        bad = self.flags - set(FLAGS)
        if bad:
            raise ValueError(f"unknown flags {sorted(bad)}")
        self._loc = {}           # inst_lock: name -> located centre
        self._rel_at = {}        # honest_done / post_release: name -> {"xy", "size"} where it was let go
        if self.flags & T_FLAGS:
            self.T = GenSkillTools(A, prof, self.flags)
            if "fail_post" in self.flags:
                A.run = _wrap_run(self.T)
            if hasattr(A, "holding_ref"):
                A.holding_ref = self.T.holding
                A.held_est_fn = self.T.held_est

    def _on(self, f):
        return f in self.flags

    # ---------------------------------------------------------------- dispatcher hooks
    def call(self, name, args):
        if not self.flags:
            return super().call(name, args)
        A, T = self.A, self.T
        obj = args.get("object") if isinstance(args, dict) else None
        arm = self.arms[0] if self.single else args.get("arm")
        held_before = T.holding.get(arm) if arm in self.arms else None
        pre_size = _size(A.first.get(held_before)) if held_before else None
        if isinstance(T, GenSkillTools):
            T.last_place = None
            T.__dict__["_cur_moves"] = []
        r = super().call(name, args)
        if not isinstance(r, dict):
            return r
        if self._on("inst_lock") and name == "locate":
            for n, d in (r.get("objects") or {}).items():
                if isinstance(d, dict) and d.get("visible") and A.cache.get(n) is not None:
                    self._loc[n] = np.asarray(A.cache[n]["center"], float).copy()
        if name in ("grasp", "push", "place", "carry", "articulate", "press") and isinstance(obj, str):
            self._loc.pop(obj, None)
            self._rel_at.pop(obj, None)
        if name == "place" and held_before == obj and T.holding.get(arm) != obj:
            self._after_place(r, obj, args, pre_size)
        if name == "release" and held_before and T.holding.get(arm) is None:
            self._after_release(r, held_before, pre_size)
        return r

    # ---------------------------------------------------------------- post_release
    def _after_place(self, r, obj, args, size0):
        A, T = self.A, self.T
        lp = getattr(T, "last_place", None)
        tgt = None if not lp or lp.get("obj") != obj else lp["target"]
        hand = np.asarray(A.hand_pose(self.arms[0] if self.single else args.get("arm"))[:3], float)
        let_go = tgt[:2] if tgt is not None else hand[:2]
        if self._on("honest_done"):
            self._rel_at[obj] = {"xy": np.asarray(let_go, float).copy(), "size": size0}
        if not self._on("post_release") or r.get("code") != "OK" or tgt is None:
            return
        ref = args.get("ref", "table")
        rel = str(args.get("relation", self.P.get("default_relation", "near"))).lower().replace(" ", "_")
        names = [obj] + ([ref] if isinstance(ref, str) and ref != "table" else [])
        now = A.perceive(names)
        o3 = now.get(obj)
        rec = {"target": self.pos(tgt)}
        if o3 is None:
            r.update(code="RELEASED_NOT_SEEN", ok=None,
                     why=f"released, but the cameras cannot find {obj} now; check the images (it may have fallen or bounced away)")
            r["gen_post_release"] = {**rec, "seen": False}
            return
        c = np.asarray(o3["center"], float)
        jump = float(np.linalg.norm(c[:2] - np.asarray(let_go[:2], float)))
        rec.update(object_now=self.pos(c), bottom_z=self.z(o3["min"][2]), jump_cm=round(jump * 100, 1))
        if jump > JUMP_M and _implausible(o3, size0):
            r.update(code="MEASURE_UNSURE", ok=None,
                     why=(f"released; the cameras now report {obj} {jump * 100:.0f} cm from where it was let go: either a mis-detection "
                          f"or it slid / bounced away; look at the image: if {obj} sits on the target, the placement worked"))
            r["gen_post_release"] = rec
            return
        if self._on("honest_done"):
            self._rel_at[obj] = {"xy": c[:2].copy(), "size": _size(o3)}
        h0 = float(size0[2]) if size0 is not None else 0.0
        tall = size0 is not None and h0 >= 0.05 and h0 >= 1.2 * float(min(size0[0], size0[1]))
        hz = float(o3["max"][2] - o3["min"][2])
        if tall and hz < 0.65 * h0 and rel not in ("in", "into", "inside"):
            r.update(code="OFF_TARGET", ok=False, off_target={"dx_cm": None, "dy_cm": None},
                     why=f"released, but the object fell over (height {hz * 100:.0f} cm, was {h0 * 100:.0f} cm)")
            r["gen_post_release"] = {**rec, "fell": True}
            return
        d = c[:2] - tgt[:2]
        off = False
        rr = now.get(ref) if len(names) > 1 else None
        if rel in ("in", "into", "inside", "on", "onto", "on_top_of") and len(names) > 1:
            if rr is None:
                rec["ref_seen"] = False
            else:
                m = 0.01 if rel in ("in", "into", "inside") else -0.005
                lo, hi = np.asarray(rr["min"][:2], float) + m, np.asarray(rr["max"][:2], float) - m
                off = bool(np.any(c[:2] < lo) or np.any(c[:2] > hi))
                rec["over_ref"] = not off
        elif len(names) == 1:
            off = float(np.linalg.norm(d)) > OFF_TABLE_M
        else:
            off = float(np.linalg.norm(d)) > OFF_BESIDE_M
        rec["placed_offset_cm"] = [round(float(x) * 100, 1) for x in d]
        r["gen_post_release"] = rec
        if off:
            r.update(code="OFF_TARGET", ok=False, off_target={"dx_cm": round(float(d[0]) * 100, 1), "dy_cm": round(float(d[1]) * 100, 1)},
                     why="released, but the cameras measure the object away from the target; re-grasp and place again if the task needs it")

    def _after_release(self, r, obj, size0):
        A = self.A
        hand = np.asarray(A.hand_pose(self.arms[0])[:3], float)
        if self._on("honest_done"):
            self._rel_at[obj] = {"xy": hand[:2].copy(), "size": size0}
        if not self._on("post_release"):
            return
        o3 = A.perceive([obj]).get(obj)
        if o3 is None:
            r["gen_post_release"] = {"seen": False, "note": f"the cameras cannot find {obj} after letting go; check the images"}
            return
        c = np.asarray(o3["center"], float)
        jump = float(np.linalg.norm(c[:2] - hand[:2]))
        r["gen_post_release"] = {"object_now": self.pos(c), "bottom_z": self.z(o3["min"][2]), "from_gripper_cm": round(jump * 100, 1)}
        if self._on("honest_done") and not (jump > JUMP_M and _implausible(o3, size0)):
            self._rel_at[obj] = {"xy": c[:2].copy(), "size": _size(o3)}

    # ---------------------------------------------------------------- inst_lock + honest held_check
    def grasp(self, arm, args):
        if not self.flags:
            return super().grasp(arm, args)
        A, T = self.A, self.T
        obj = args.get("object")
        rec = None
        loc = self._loc.get(obj) if self._on("inst_lock") else None
        if loc is not None and not T.holding.get(arm) and hasattr(A, "instances") and hasattr(A, "track"):
            o = A.perceive([obj]).get(obj)
            if o is not None and float(np.linalg.norm(np.asarray(o["center"][:2]) - loc[:2])) > LOCK_M:
                d0 = float(np.linalg.norm(np.asarray(o["center"][:2]) - loc[:2]))
                rec = {"located": self.pos(loc), "seen": self.pos(o["center"]), "jump_cm": round(d0 * 100, 1), "relocked": False}
                try:
                    cand = [i for _, i in A.instances(obj)]
                except Exception:  # noqa: BLE001
                    cand = []
                best = min(cand, key=lambda i: float(np.linalg.norm(np.asarray(i["center"][:2]) - loc[:2])), default=None)
                if best is not None and float(np.linalg.norm(np.asarray(best["center"][:2]) - loc[:2])) <= LOCK_M:
                    A.cache[obj] = dict(best)
                    A.track[obj] = (np.asarray(best["center"], float).copy(), np.asarray(A.hand_pose()[:3], float).copy(), False)
                    rec["relocked"] = True
                else:
                    A.cache[obj] = o
                    self._loc.pop(obj, None)
                    return {"code": "TARGET_MOVED", "ok": False, "gen_inst_lock": rec,
                            "why": (f"{obj} is no longer where it was located: the cameras now find it {d0 * 100:.0f} cm away; it was "
                                    f"pushed, or this detection is another object. Nothing was executed: check the images, locate {obj} "
                                    f"(with a visual description if needed), then grasp again")}
        out = super().grasp(arm, args)
        if rec is not None and isinstance(out, dict):
            out["gen_inst_lock"] = rec
        if self._on("honest_done") and isinstance(out, dict) and out.get("code") == "OK" and "not visible" in str(out.get("lift_test", "")):
            out["held_check"] = (f"finger opening only: the cameras could not see {obj} during the lift test, so that the held thing is "
                                 f"{obj} is not confirmed by the cameras")
        if isinstance(T, GenSkillTools) and isinstance(out, dict):
            ht = (T.hold.get(arm) or {}).get("gen")
            if ht and out.get("code") == "OK" and self._on("held_track"):
                out["gen_held_track"] = "in-hand pose recorded at closing"
        return out

    # ---------------------------------------------------------------- place: keep the hold when fail_post refused the release
    def place(self, name, arm, args):
        if not self.flags:
            return super().place(name, arm, args)
        T = self.T
        before = (T.holding.get(arm), T.T_GO.get(arm), T.hold.get(arm))
        out = super().place(name, arm, args)
        if isinstance(T, GenSkillTools) and T._skip_rel == arm and before[0] is not None:
            T.holding[arm] = before[0]
            if before[1] is not None:
                T.T_GO[arm] = before[1]
            if before[2] is not None:
                T.hold[arm] = before[2]
            T.cancelled.pop(arm, None)
            if isinstance(out, dict):
                out["code"], out["ok"] = "PLAN_FAIL", False
                out["why"] = f"could not bring {args.get('object')} to the target; it was NOT released and is still held"
        return out

    # ---------------------------------------------------------------- release (step_open ordering)
    def release(self, arm):
        if not (self._on("step_open") and hasattr(self.A, "grip_to")):
            return super().release(arm)
        A, T = self.A, self.T
        o = float(A.opening(arm))
        if o + PART_OPEN >= 1.0 or o >= 0.75 or not T.holding.get(arm):
            return super().release(arm)
        for v in (min(1.0, o + 0.05), min(1.0, o + PART_OPEN)):
            A.grip_to(arm, v)
        h = A.hand_pose(arm)
        cav = getattr(A, "cur_approach_vec", None)
        back = -np.asarray(cav) if cav is not None else np.array([0, 0, 1.0])
        A.move(h[:3] + 0.08 * back, *([h[3:]] if self.full_quat else []))
        out = self._exec(T.gripper(arm, 1.0))
        if back[2] == 0:
            h = A.hand_pose(arm)
            A.move([h[0], h[1], max(h[2], self.P["safe_z"])], h[3:])
        if self.side_grasp:
            A.cur_approach_vec = None
        T.drop_hold(arm)
        out["gen_step_open"] = "partial open, back off, then open fully"
        return out

    # ---------------------------------------------------------------- honest_done: predicates on released objects
    def _g(self, n, now):
        o = super()._g(n, now)
        if not self._on("honest_done"):
            return o
        ra = self._rel_at.get(n)
        if ra is not None and n not in self.T.holding.values():
            d = float(np.linalg.norm(np.asarray(o["center"][:2], float) - ra["xy"]))
            if d > JUMP_M and _implausible(o, ra.get("size")):
                raise Unknown(f"{n} is measured {d * 100:.0f} cm from where it was let go and nothing touched it since: probably a "
                              f"mis-detection; judge it from the images")
        return o

    def done_reject_suffix(self, res):
        if not self._on("honest_done"):
            return None
        cks = (res or {}).get("checks") or []
        if cks and all(c.get("ok") is not False for c in cks):
            return (" -> done REJECTED: some checks could not be measured (ok=null), nothing measured false; judge those from the "
                    "images and call done without them if the images show they hold.")
        return None


def system_addendum(bench: str, flags) -> str:
    flags = set(flags or ())
    s = []
    if "post_release" in flags:
        s.append("- After place lets go, the tool re-measures the object: code OFF_TARGET with off_target {dx_cm, dy_cm} when it is "
                 "away from the target or fell over, MEASURE_UNSURE (ok=null) when the re-measurement looks implausible, "
                 "RELEASED_NOT_SEEN (ok=null) when it cannot be found; judge ok=null results from the images.")
    if "honest_done" in flags:
        s.append("- A check with ok=null could not be measured reliably: judge it from the images and leave it out of done if the "
                 "images show it holds. grasp may report held_check when the cameras could not confirm the object in the hand.")
    if "inst_lock" in flags:
        s.append("- grasp keeps to the instance you located last; TARGET_MOVED means it is no longer there and nothing was executed: "
                 "locate it again.")
    if "fail_post" in flags:
        s.append("- grasp may return NOT_CLOSED (the fingers stayed open). If place cannot reach the drop point it returns PLAN_FAIL "
                 "and the object stays in the gripper.")
    return ("\n" + "\n".join(s) + "\n") if s else ""
