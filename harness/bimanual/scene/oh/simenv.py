"""RoboTwin 2.0 simulation backend for our harness.

Reuses RoboDawn's RoboTwinDiscreteEnv for the official episode protocol (seed validation by the expert, instruction
sampling, same task object) and its cameras, but executes motions itself:
  * every arm motion = one RoboTwin planner call (robot.<arm>_plan_path, optional straight-line constraint like the
    expert's approach moves) followed by dense execution (Base_Task.take_dense_action);
  * the task checker NEVER ends or freezes anything: it is only sampled (privately) for scoring;
  * physics is hooked to record gripper/object contacts (for touched()/holding()) every few physics steps.

Units exposed to the agents: metres; x/y = RoboTwin world frame (x right, y away from the robot), z = height above the
table top.  Internally RoboTwin uses world z (table top ~0.74 m).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import paths as _paths  # noqa: E402
RD = Path(os.environ.get("ROBODAWN_ROOT", str(_paths.ROBODAWN)))
sys.path.insert(0, str(RD))
from harness.robotwin.env import (RoboTwinDiscreteEnv, wrist_to_tcp, tcp_to_wrist, quat_angle_deg,  # noqa: E402
                                  ORIENTATION_PRESETS, annotate_head_image, encode_png, enhance_image)
import transforms3d as t3d  # noqa: E402
import sapien  # noqa: E402
from envs.utils.actor_utils import Actor, ArticulationActor  # noqa: E402

log = logging.getLogger("oh.sim")
SKIP_ACTORS = {"wall", "table", "last_actor"}   # last_actor: stale alias left by the expert replay (stack_blocks_*)
SAMPLE_EVERY = 5          # physics steps between contact / checker samples


def pretty_model(model: str) -> str:
    """'021_cup' -> 'cup', '105_sauce-can' -> 'sauce can'."""
    s = model.split("/")[0]
    if "_" in s and s.split("_")[0].isdigit():
        s = s.split("_", 1)[1]
    return s.replace("-", " ").replace("_", " ").strip()


COLOR_NAMES = {(1, 0, 0): "red", (0, 1, 0): "green", (0, 0, 1): "blue", (1, 1, 0): "yellow"}


class SimEnv:
    def __init__(self, task: str, seed_dir: str, task_config: str = "demo_randomized", seed_base_index: int = -1):
        # seed_base_index=-1 -> RoboDawn's seed base 100000*(1+index) = 0 = RoboTwin's data-collection seeds 0,1,2,...
        self.task = task
        self.rd = RoboTwinDiscreteEnv(task, task_config=task_config, seed=seed_base_index, record_video=False)
        self.rd.seed_cache_dir = Path(seed_dir)
        self.rd.seed_cache = self.rd._load_seed_cache()
        self.T = None
        self.names: dict[str, Actor] = {}
        self.touch_log: list[dict] = []          # {"t": motion idx, "arm":, "obj":}
        self.motion_idx = 0
        self.checker_ever = False
        self.checker_ever_first = None
        self.physics_steps = 0
        self.initial: dict = {}

    # ------------------------------------------------------------------ episode
    def reset(self, episode_index: int):
        t0 = time.time()
        ep = self.rd.reset_episode(episode_index)
        self.T = self.rd.env
        self.T.eval_success = False
        self.T.plan_success = True
        self._orig_check = type(self.T).check_success.__get__(self.T)
        self.touch_log, self.motion_idx, self.checker_ever, self.checker_ever_first = [], 0, False, None
        self._fingers()
        self._hook_physics()
        self._build_names()
        self.initial = {n: self.object_state(n) for n in self.names}
        self.reset_seconds = time.time() - t0
        return ep

    def close_episode(self):
        self.rd.close_episode()

    def close(self):
        self.rd.close()

    def _fingers(self):
        robot = self.T.robot
        self.finger_links = {}
        for arm in ("left", "right"):
            names = set()
            for j in (robot.left_gripper if arm == "left" else robot.right_gripper):
                try:
                    names.add(j[0].get_child_link().get_name())
                except Exception:  # noqa: BLE001
                    pass
            self.finger_links[arm] = names
        self.finger2arm = {f: a for a, fs in self.finger_links.items() for f in fs}

    def _build_names(self):
        """Human-readable unique names for the task's objects (oracle perception: knows what everything is)."""
        cands = []
        seen_ids = set()
        for attr, v in vars(self.T).items():
            if not isinstance(v, Actor) or attr in SKIP_ACTORS or v.config is None:
                continue
            if id(v) in seen_ids:          # alias attribute (e.g. last_actor) of an object already listed
                continue
            seen_ids.add(id(v))
            try:
                model = v.get_name()
            except Exception:  # noqa: BLE001
                continue
            base = pretty_model(model)
            if base in ("box", "target box", "") or model in ("box", "target_box"):
                col = self._box_color(v)
                base = f"{col} block" if col else attr.replace("_", " ")
            cands.append((attr, base, v))
        counts = {}
        for _, b, _ in cands:
            counts[b] = counts.get(b, 0) + 1
        self.names, self.attr_of = {}, {}
        seen = {}
        for attr, b, v in sorted(cands, key=lambda c: float(c[2].get_pose().p[0])):
            if counts[b] > 1:   # duplicates: number them from left (-x) to right (+x)
                seen[b] = seen.get(b, 0) + 1
                name = f"{b} {seen[b]}"
            else:
                name = b
            name = name.replace(" ", "_")
            self.names[name] = v
            self.attr_of[name] = attr
        self.entity2name = {}          # per_scene_id -> name (entity NAMES are not unique: RoboTwin calls every block "box")
        for n, v in self.names.items():
            try:
                if isinstance(v, ArticulationActor):
                    for l in v.actor.get_links():
                        self.entity2name[l.entity.per_scene_id] = n
                else:
                    self.entity2name[v.actor.per_scene_id] = n
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _box_color(v):
        try:
            for comp in v.actor.get_components():
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    for s in comp.render_shapes:
                        c = s.material.base_color
                        key = tuple(int(round(x)) for x in c[:3])
                        return COLOR_NAMES.get(key)
        except Exception:  # noqa: BLE001
            pass
        return None

    # ------------------------------------------------------------------ physics hook
    def _hook_physics(self):
        scene = self.T.scene
        if getattr(scene, "_oh_hooked", False):
            scene._oh_owner = self
            return
        orig = scene.step
        scene._oh_owner = self

        def step():
            orig()
            own = scene._oh_owner
            own.physics_steps += 1
            if own.physics_steps % SAMPLE_EVERY == 0:
                own._sample()
        scene.step = step
        scene._oh_hooked = True

    def _sample(self):
        for arm, obj, pos in self.contact_points():
            self.touch_log.append({"t": self.motion_idx, "arm": arm, "obj": obj, "pos": pos})
        if not self.checker_ever and self.checker_now():
            self.checker_ever = True
            self.checker_ever_first = self.motion_idx

    def contacts(self) -> set:
        """{(arm, object name, finger link)} gripper-object contacts right now."""
        out = set()
        try:
            for c in self.T.scene.get_contacts():
                e0, e1 = c.bodies[0].entity, c.bodies[1].entity
                for ea, eb in ((e0, e1), (e1, e0)):
                    a = ea.get_name()
                    if a in self.finger2arm and eb.per_scene_id in self.entity2name and len(c.points) > 0:
                        out.add((self.finger2arm[a], self.entity2name[eb.per_scene_id], a))
        except Exception:  # noqa: BLE001
            pass
        return out

    def contact_points(self) -> list:
        """[(arm, object name, mean contact position xyz world)] for every finger-object contact right now."""
        out = []
        try:
            for c in self.T.scene.get_contacts():
                e0, e1 = c.bodies[0].entity, c.bodies[1].entity
                for ea, eb in ((e0, e1), (e1, e0)):
                    a = ea.get_name()
                    if a in self.finger2arm and eb.per_scene_id in self.entity2name and len(c.points) > 0:
                        p = np.mean([np.asarray(pt.position, float) for pt in c.points], 0)
                        out.append((self.finger2arm[a], self.entity2name[eb.per_scene_id], p.tolist()))
        except Exception:  # noqa: BLE001
            pass
        return out

    def speed(self, name: str) -> float:
        """Linear speed (m/s) of an object (root link for articulations)."""
        v = self.names[name]
        try:
            if isinstance(v, ArticulationActor):
                return float(np.linalg.norm(v.actor.get_links()[0].get_linear_velocity()))
            comp = v.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if comp is None:
                return 0.0          # static object
            return float(np.linalg.norm(comp.get_linear_velocity()))
        except Exception:  # noqa: BLE001
            return 0.0

    def finger_contacts(self) -> dict:
        d = {}
        for a, o, f in self.contacts():
            d.setdefault((a, o), set()).add(f)
        return d

    def held(self) -> dict:
        """{arm: object} where both fingers of the arm touch the object."""
        out = {}
        for (a, o), fs in self.finger_contacts().items():
            if len(fs) >= 2:
                out[a] = o
        return out

    # ------------------------------------------------------------------ privileged scoring (never shown to agents)
    def checker_now(self) -> bool:
        try:
            return bool(self._orig_check())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ geometry
    @property
    def table_z(self) -> float:
        return 0.74 + float(getattr(self.T, "table_z_bias", 0.0))

    def aabb(self, name: str):
        v = self.names[name]
        if isinstance(v, ArticulationActor):
            bs = []
            for l in v.actor.get_links():
                try:
                    bs.append(l.compute_global_aabb_tight())
                except RuntimeError:      # visual-only link without collision shapes
                    pass
            lo = np.min([b[0] for b in bs], 0)
            hi = np.max([b[1] for b in bs], 0)
        else:
            comp = v.actor.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
            lo, hi = comp.compute_global_aabb_tight()
        return np.asarray(lo, float), np.asarray(hi, float)

    def object_state(self, name: str) -> dict:
        v = self.names[name]
        lo, hi = self.aabb(name)
        p = v.get_pose()
        yaw = math.degrees(t3d.euler.quat2euler(np.asarray(p.q), axes="sxyz")[2])
        return {"center": ((lo + hi) / 2).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "quat": np.asarray(p.q).tolist(),
                "pos": np.asarray(p.p).tolist(), "yaw_deg": yaw}

    def to_table(self, p):
        p = np.asarray(p, float).copy()
        p[2] -= self.table_z
        return p

    def from_table(self, p):
        p = np.asarray(p, float).copy()
        p[2] += self.table_z
        return p

    # ------------------------------------------------------------------ arm state
    def wrist_pose(self, arm):
        return np.asarray(self.T.get_arm_pose(arm), float)

    def tcp(self, arm):
        return wrist_to_tcp(self.wrist_pose(arm))

    def opening(self, arm):
        return self.rd.measured_gripper_opening(arm)

    def gripper_cmd(self, arm):
        r = self.T.robot
        return float(r.get_left_gripper_val() if arm == "left" else r.get_right_gripper_val())

    def approach(self, arm):
        R = t3d.quaternions.quat2mat(self.wrist_pose(arm)[3:])
        return R[:, 0]

    # ------------------------------------------------------------------ execution
    def plan(self, arm, wrist_target, straight=False):
        robot = self.T.robot
        f = robot.left_plan_path if arm == "left" else robot.right_plan_path
        try:
            res = f(np.asarray(wrist_target, float).tolist(), constraint_pose=[1, 1, 1, 0, 0, 0] if straight else None)
        except Exception as exc:  # noqa: BLE001
            return {"status": f"error {exc}"}
        if straight and res.get("status") != "Success":     # fall back to a free path
            try:
                res = f(np.asarray(wrist_target, float).tolist(), constraint_pose=None)
                res["straight_fallback"] = True
            except Exception as exc:  # noqa: BLE001
                return {"status": f"error {exc}"}
        return res

    def execute_arms(self, targets: dict, straight=False) -> dict:
        """targets: {arm: wrist pose(7)} executed simultaneously. If any arm cannot be planned, NOTHING moves
        (no silent partial execution). A straight-line request that only has a free path is executed on the free
        path and reported as a degradation ("straight_fallback")."""
        self.motion_idx += 1
        self.T.plan_success = True
        plans, rep = {}, {}
        for arm, tgt in targets.items():
            res = self.plan(arm, tgt, straight)
            if res.get("status") != "Success":
                rep[arm] = {"ok": False, "why": f"motion planner failed ({res.get('status')})"}
            else:
                plans[arm] = res
        if rep:                                      # at least one arm unplannable -> abort all
            for arm in plans:
                rep[arm] = {"ok": False, "why": "not executed because the other arm could not be planned"}
            return rep
        seq = {"left_arm": plans.get("left"), "left_gripper": None, "right_arm": plans.get("right"), "right_gripper": None}
        self.T.take_dense_action(seq, save_freq=None)
        self._settle(20)
        for arm, tgt in targets.items():
            reached = self.wrist_pose(arm)
            perr = float(np.linalg.norm(wrist_to_tcp(reached) - wrist_to_tcp(np.asarray(tgt))))
            aerr = quat_angle_deg(reached[3:], np.asarray(tgt)[3:])
            ok = perr < 0.015 and aerr < 8
            rep[arm] = {"ok": ok, "pos_err_m": round(perr, 3), "ang_err_deg": round(aerr, 1)}
            if plans[arm].get("straight_fallback"):
                rep[arm]["straight_fallback"] = True
            if not ok:
                rep[arm]["why"] = f"stopped {perr*100:.1f} cm / {aerr:.0f} deg from the target (blocked or contact)"
        return rep

    def set_gripper(self, arm, value) -> None:
        self.motion_idx += 1
        T = self.T
        T.plan_success = True
        if arm == "left":
            res = T.set_gripper(set_tag="left", left_pos=float(value))
            seq = {"left_arm": None, "left_gripper": res, "right_arm": None, "right_gripper": None}
        else:
            res = T.set_gripper(set_tag="right", right_pos=float(value))
            seq = {"left_arm": None, "left_gripper": None, "right_arm": None, "right_gripper": res}
        T.take_dense_action(seq, save_freq=None)
        self._settle_gripper(arm, float(value))

    def _settle(self, n):
        robot = self.T.robot
        lg, rg = self.gripper_cmd("left"), self.gripper_cmd("right")
        for _ in range(n):
            robot.set_gripper(lg, "left")
            robot.set_gripper(rg, "right")
            self.T.scene.step()

    def _settle_gripper(self, arm, target, max_steps=400, stable_needed=20):
        robot = self.T.robot
        other = "right" if arm == "left" else "left"
        ov = self.gripper_cmd(other)
        prev, stable = self.opening(arm), 0
        for _ in range(max_steps):
            robot.set_gripper(target, arm)
            robot.set_gripper(ov, other)
            self.T.scene.step()
            cur = self.opening(arm)
            stable = stable + 1 if abs(cur - prev) < 5e-4 else 0
            prev = cur
            if stable >= stable_needed:
                break

    # ------------------------------------------------------------------ images
    def images(self) -> dict:
        """{'agent': PIL annotated overview, 'top': PIL annotated top view}"""
        obs = self.rd.observe()
        out = {}
        for cam, key in (("agent_camera", "agent"), ("top_camera", "top")):
            try:
                out[key] = annotate_head_image(self.rd, obs, camera=cam)
            except Exception as exc:  # noqa: BLE001
                log.warning("image %s failed: %s", cam, exc)
        return out
