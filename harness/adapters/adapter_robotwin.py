"""RoboTwin (SAPIEN) adapter: the ONLY module of harness that touches the simulator. A LIBERO / MuJoCo backend implements
the same methods. Units: metres, world frame. hand pose = the frame the arm planner moves, pose7 = [x,y,z,qw,qx,qy,qz].
Backend geometry queries (grasp_plan / place_plan) use the backend's own contact-point / functional-point annotations
(RoboTwin: choose_grasp_pose / get_place_pose of the base task, never a task script)."""
from __future__ import annotations

import numpy as np
import sapien

from envs.utils import ArmTag
from envs.utils.action import Action
from envs.utils.actor_utils import Actor, ArticulationActor

ARMS = ("left", "right")


def act2dict(a):
    if a.action == "move":
        return {"type": "move", "pose": [float(x) for x in a.target_pose],
                **({"constraint": list(a.args["constraint_pose"])} if getattr(a, "args", {}).get("constraint_pose") is not None else {})}
    return {"type": "grip", "val": float(a.target_gripper_pos)}


def dict2act(arm, d):
    if d["type"] == "move":
        kw = {"constraint_pose": d["constraint"]} if d.get("constraint") is not None else {}
        return Action(ArmTag(arm), "move", target_pose=list(d["pose"]), **kw)
    return Action(ArmTag(arm), "close", target_gripper_pos=float(d["val"]))  # "close" just sets the target position


class RoboTwinAdapter:
    def __init__(self, sim):
        self.sim, self.T = sim, sim.T
        self.by_id = {id(v): n for n, v in sim.names.items()}

    # ---------------------------------------------------------------- robot
    def hand_pose(self, arm):
        return np.asarray(self.T.get_arm_pose(ArmTag(arm)), float)

    def tcp_point(self, pose7):
        from harness.robotwin.env import wrist_to_tcp
        return np.asarray(wrist_to_tcp(np.asarray(pose7, float)), float)[:3]

    def hand_from_tcp(self, tcp, q):
        """hand pose whose fingertip centre is at tcp with orientation q (inverse of tcp_point)."""
        h0 = np.r_[0.0, 0.0, 0.0, np.asarray(q, float)]
        return np.r_[np.asarray(tcp, float) - self.tcp_point(h0), np.asarray(q, float)]

    def centers(self):
        out = {}
        for n in self.objects():
            lo, hi = self.aabb(n)
            out[n] = (lo + hi) / 2
        return out

    def run(self, items: dict) -> bool:
        """items {arm: [action dict]} executed index-paired (both arms together). Returns plan success."""
        seq = [(ArmTag(a), [dict2act(a, d) for d in L]) for a, L in items.items() if L]
        if not seq:
            return True
        self.T.plan_success = True
        self.T.move(*seq)
        ok = bool(self.T.plan_success)
        self.T.plan_success = True
        return ok

    def gripper_cmd(self, arm):
        r = self.T.robot
        return float(r.get_left_gripper_val() if arm == "left" else r.get_right_gripper_val())

    def opening(self, arm):
        return float(self.sim.opening(arm))

    def held(self) -> dict:
        return dict(self.sim.held())

    def home_pose(self, arm):
        r = self.T.robot
        return list(map(float, r.left_original_pose if arm == "left" else r.right_original_pose))

    # ---------------------------------------------------------------- backend geometry (annotations of the assets)
    def grasp_plan(self, obj, arm, pre_dist, depth, grip, contact):
        self.T.plan_success = True
        tag, acts = self.T.grasp_actor(self.sim.names[obj], ArmTag(arm), pre_grasp_dis=pre_dist, grasp_dis=depth,
                                       gripper_pos=grip, contact_point_id=contact)
        return [act2dict(a) for a in acts] if tag is not None else None

    def place_plan(self, obj, arm, target7, fp, pre_dist, dist, release, extra):
        self.T.plan_success = True
        tag, acts = self.T.place_actor(self.sim.names[obj], ArmTag(arm), list(map(float, target7)), functional_point_id=fp,
                                       pre_dis=pre_dist, dis=dist, is_open=release, **extra)
        return [act2dict(a) for a in acts] if tag is not None else None

    # ---------------------------------------------------------------- scene
    def name_of(self, v):
        return self.by_id.get(id(v))

    def objects(self):
        return list(self.sim.names)

    def obj_pose(self, name):
        p = self.sim.names[name].get_pose()
        return np.concatenate([np.asarray(p.p, float), np.asarray(p.q, float)])

    def aabb(self, name):
        lo, hi = self.sim.aabb(name)
        return np.asarray(lo, float), np.asarray(hi, float)

    def is_articulated(self, name):
        return isinstance(self.sim.names[name], ArticulationActor)

    def joints(self, name):
        v = self.sim.names[name]
        if not isinstance(v, ArticulationActor):
            return []
        a = v.actor
        q = np.asarray(a.get_qpos(), float)
        out = []
        for i, j in enumerate(a.get_active_joints()):
            M = j.get_global_pose().to_transformation_matrix()
            out.append({"idx": i, "type": "prismatic" if "prismatic" in str(j.type) else "revolute", "q": float(q[i]),
                        "origin": M[:3, 3].copy(), "axis": M[:3, 0].copy()})
        return out

    def table_z(self):
        return float(self.sim.table_z)

    def success(self):
        try:
            return bool(self.T.check_success())
        except Exception:  # noqa: BLE001
            return False

    def images(self):
        try:
            return self.sim.images()
        except Exception:  # noqa: BLE001
            return {}

    # ---------------------------------------------------------------- state
    def snapshot(self):
        objs = {}
        for n in self.objects():
            lo, hi = self.aabb(n)
            objs[n] = {"pose": self.obj_pose(n).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                       "q": [j["q"] for j in self.joints(n)]}
        return {"hands": {a: self.hand_pose(a).tolist() for a in ARMS},
                "opening": {a: self.opening(a) for a in ARMS},
                "cmd": {a: self.gripper_cmd(a) for a in ARMS},
                "held": self.held(), "objects": objs}

    def save_state(self):
        """restorable, JSON-serialisable sim state (actor poses, articulation qpos, robot qpos, gripper commands)."""
        T = self.T
        acts = {}
        for k, v in vars(T).items():
            if isinstance(v, Actor) and k not in acts:
                try:
                    p = v.actor.get_pose()
                    acts[k] = [list(map(float, p.p)), list(map(float, p.q))]
                except Exception:  # noqa: BLE001
                    pass
        arts = [[a.get_name(), list(map(float, a.get_qpos()))] for a in T.scene.get_all_articulations()]
        return {"actors": acts, "arts": arts, "grip": [self.gripper_cmd("left"), self.gripper_cmd("right")]}

    def restore_state(self, st, settle=20):
        T = self.T
        for k, (p, q) in st["actors"].items():
            v = getattr(T, k, None)
            if isinstance(v, Actor):
                v.actor.set_pose(sapien.Pose(p, q))
                comp = v.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                if comp is not None:
                    comp.set_linear_velocity([0, 0, 0])
                    comp.set_angular_velocity([0, 0, 0])
        for a, (name, q) in zip(T.scene.get_all_articulations(), st["arts"]):
            q = np.asarray(q)
            a.set_qpos(q)
            a.set_qvel(np.zeros_like(q))
            for j, qq in zip(a.get_active_joints(), q):
                try:
                    j.set_drive_target(float(qq))
                except Exception:  # noqa: BLE001
                    pass
        for _ in range(settle):
            T.robot.set_gripper(st["grip"][0], "left")
            T.robot.set_gripper(st["grip"][1], "right")
            T.scene.step()
        T.plan_success = True


# ====================================================================================================================
# Camera-only RoboTwin adapter for the single-arm harness layer (single_arm/planner.py + tools.SkillTools). Same interface as
# adapter_guava / adapter_libero: table frame (x, y = world, z = height above the table), metres; poses are TCP
# (fingertip centre) pose7. Perception = SAM3 + native head / wrist RGB-D fusion
# robot state = proprioception; feasibility = the arm's own motion planner. No asset contact / functional points, no
# simulator object poses, no joint values, no task names. success() is logging only.
ARM_BASE = {"left": np.array([-0.30, -0.35]), "right": np.array([0.30, -0.35])}
GAP_MIN, OPEN_FRAC = 0.02, 0.8          # finger opening fractions: below = closed on nothing, above = open
BOTTOM_SNAP = 0.015                     # perception: a bottom within 1.5 cm of the table is reported as the table
PRE = 0.08
TILTS = (0.0, 25.0, 45.0)

