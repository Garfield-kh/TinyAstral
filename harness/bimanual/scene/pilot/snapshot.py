"""Manual snapshot / restore of a RoboTwin scene: task actors (pose + zero velocity) and the robot articulations
(qpos, zero qvel, drive targets = qpos, gripper commands). Used to branch failure injections from key moments."""
import numpy as np
import sapien


def _actors(T):
    from envs.utils.actor_utils import Actor
    out = {}
    for k, v in vars(T).items():
        if isinstance(v, Actor) and id(v) not in {id(x) for x in out.values()}:
            out[k] = v
    return out


def save(T):
    acts = {}
    for k, v in _actors(T).items():
        try:
            p = v.actor.get_pose()
            acts[k] = (np.asarray(p.p).copy(), np.asarray(p.q).copy())
        except Exception:
            pass
    arts = []
    for a in T.scene.get_all_articulations():
        arts.append((a, a.get_qpos().copy()))
    grip = (float(T.robot.get_left_gripper_val()), float(T.robot.get_right_gripper_val()))
    return {"actors": acts, "arts": arts, "grip": grip}


def restore(T, snap, settle=20):
    av = _actors(T)
    for k, (p, q) in snap["actors"].items():
        v = av.get(k)
        if v is None:
            continue
        v.actor.set_pose(sapien.Pose(p, q))
        comp = v.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if comp is not None:
            comp.set_linear_velocity([0, 0, 0])
            comp.set_angular_velocity([0, 0, 0])
    for a, q in snap["arts"]:
        a.set_qpos(q)
        a.set_qvel(np.zeros_like(q))
        for j, qq in zip(a.get_active_joints(), q):
            try:
                j.set_drive_target(float(qq))
            except Exception:
                pass
    lg, rg = snap["grip"]
    for _ in range(settle):
        T.robot.set_gripper(lg, "left")
        T.robot.set_gripper(rg, "right")
        T.scene.step()
    T.plan_success = True
