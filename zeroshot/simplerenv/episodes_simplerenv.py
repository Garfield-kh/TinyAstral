"""SimplerEnv visual-matching episode manifest, transcribed from the official scripts
(SimplerEnv/scripts/rt1_*_visual_matching.sh, rt1x_bridge.sh / octo_bridge.sh), plus env build / reset exactly as
simpler_env/evaluation/maniskill2_evaluator.py does it (obs_mode rgbd, official control mode, segmentation on for the overlay,
ray tracing for the drawer scenes). Only the TimeLimit is raised (official horizon kept in ep["max_steps"] and enforced by the
runner's strict metric), because a tool-calling agent decides when to stop.

Episode keys: coke_h / coke_v / coke_s (pick coke can horizontal / vertical / standing), move_near, drawer_open,
drawer_close, apple_drawer, br_spoon, br_carrot, br_stack, br_eggplant."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _simpler_repo():
    """SimplerEnv clone: $SIMPLERENV_REPO, else $TINYASTRAL_THIRD_PARTY/SimplerEnv, else <repo>/third_party/SimplerEnv."""
    if os.environ.get("SIMPLERENV_REPO"):
        return os.environ["SIMPLERENV_REPO"]
    tp = os.environ.get("TINYASTRAL_THIRD_PARTY", str(Path(__file__).resolve().parents[2] / "third_party"))
    return os.path.join(tp, "SimplerEnv")


SE = _simpler_repo()
INP = SE + "/ManiSkill2_real2sim/data/real_inpainting/"
GOOGLE = dict(robot="google_robot_static", control_freq=3, sim_freq=513,
              control_mode="arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner")
WIDOWX = dict(control_freq=5, sim_freq=500, control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos")
URDF_VM = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]
DRAWER_POSES = [  # robot x, y, yaw, overlay  (rt1_drawer_visual_matching.sh)
    (0.644, -0.179, -0.03, "open_drawer_a0.png"), (0.765, -0.182, -0.02, "open_drawer_a1.png"),
    (0.889, -0.203, -0.06, "open_drawer_a2.png"), (0.652, 0.009, 0.0, "open_drawer_b0.png"),
    (0.752, 0.009, 0.0, "open_drawer_b1.png"), (0.851, 0.035, 0.0, "open_drawer_b2.png"),
    (0.665, 0.224, 0.0, "open_drawer_c0.png"), (0.765, 0.222, -0.025, "open_drawer_c1.png"),
    (0.865, 0.222, -0.025, "open_drawer_c2.png")]
GROUPS = {"pick_coke_can": ["coke_h", "coke_v", "coke_s"], "move_near": ["move_near"], "drawer": ["drawer_open", "drawer_close"],
          "apple_drawer": ["apple_drawer"], "br_spoon": ["br_spoon"], "br_carrot": ["br_carrot"], "br_stack": ["br_stack"],
          "br_eggplant": ["br_eggplant"]}


def manifest(key):
    out = []
    if key.startswith("coke_"):
        opt = {"coke_h": "lr_switch", "coke_v": "laid_vertically", "coke_s": "upright"}[key]
        for u in URDF_VM:
            for ox in np.linspace(-0.35, -0.12, 5):
                for oy in np.linspace(-0.02, 0.42, 5):
                    out.append({**GOOGLE, "env_name": "GraspSingleOpenedCokeCanInScene-v0", "scene_name": "google_pick_coke_can_1_v4",
                                "overlay": "google_coke_can_real_eval_1.png", "robot_xy": (0.35, 0.20), "robot_yaw": 0.0,
                                "obj": {"init_xy": [float(ox), float(oy)]}, "kw": {opt: True, "urdf_version": u}, "max_steps": 80})
    elif key == "move_near":
        for u in URDF_VM:
            for e in range(60):
                out.append({**GOOGLE, "env_name": "MoveNearGoogleBakedTexInScene-v0", "scene_name": "google_pick_coke_can_1_v4",
                            "overlay": "google_move_near_real_eval_1.png", "robot_xy": (0.35, 0.21), "robot_yaw": -0.09,
                            "obj": {"episode_id": e}, "kw": {"urdf_version": u}, "max_steps": 80})
    elif key in ("drawer_open", "drawer_close"):
        verb = "Open" if key == "drawer_open" else "Close"
        for u in ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", None]:
            for which in ("Top", "Middle", "Bottom"):
                for x, y, yaw, ov in DRAWER_POSES:
                    out.append({**GOOGLE, "env_name": f"{verb}{which}DrawerCustomInScene-v0", "scene_name": "dummy_drawer", "overlay": ov,
                                "robot_xy": (x, y), "robot_yaw": yaw, "obj": {"init_xy": [0.0, 0.0]}, "rt": True,
                                "kw": {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True,
                                       "urdf_version": u}, "max_steps": 113})
    elif key == "apple_drawer":
        for u in ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", None]:
            for x, y, yaw, ov in (DRAWER_POSES[0], DRAWER_POSES[3], DRAWER_POSES[6]):
                for ox in np.linspace(-0.08, -0.02, 3):
                    for oy in np.linspace(-0.02, 0.08, 3):
                        out.append({**GOOGLE, "env_name": "PlaceIntoClosedTopDrawerCustomInScene-v0", "scene_name": "dummy_drawer",
                                    "overlay": ov, "robot_xy": (x, y), "robot_yaw": yaw, "obj": {"init_xy": [float(ox), float(oy)]},
                                    "rt": True, "kw": {"station_name": "mk_station_recolor", "light_mode": "simple",
                                                       "disable_bad_material": True, "urdf_version": u, "model_ids": "baked_apple_v2"},
                                    "max_steps": 200})
    elif key.startswith("br_"):
        if key == "br_eggplant":
            base = {**WIDOWX, "robot": "widowx_sink_camera_setup", "scene_name": "bridge_table_1_v2", "overlay": "bridge_sink.png",
                    "robot_xy": (0.127, 0.06), "max_steps": 120, "env_name": "PutEggplantInBasketScene-v0"}
        else:
            base = {**WIDOWX, "robot": "widowx", "scene_name": "bridge_table_1_v1", "overlay": "bridge_real_eval_1.png",
                    "robot_xy": (0.147, 0.028), "max_steps": 60,
                    "env_name": {"br_spoon": "PutSpoonOnTableClothInScene-v0", "br_carrot": "PutCarrotOnPlateInScene-v0",
                                 "br_stack": "StackGreenCubeOnYellowCubeBakedTexInScene-v0"}[key]}
        for e in range(24):
            out.append({**base, "robot_yaw": 0.0, "obj": {"episode_id": e}, "kw": {}})
    else:
        raise KeyError(key)
    for i, e in enumerate(out):
        e["key"], e["idx"] = key, i
    return out


def build(ep, step_cap=2000):
    import mani_skill2_real2sim.envs  # noqa: F401
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    import os
    kw = {**({"shader_dir": "rt"} if ep.get("rt") else {}), **ep["kw"]}
    if os.environ.get("SIMPLER_CPU_RENDER", "0") != "1":     # GPU renderer on the first visible CUDA device
        kw["renderer_kwargs"] = {"device": "cuda:0"}
    env = build_maniskill2_env(ep["env_name"], **kw, obs_mode="rgbd", robot=ep["robot"], sim_freq=ep["sim_freq"],
                               control_mode=ep["control_mode"], control_freq=ep["control_freq"], max_episode_steps=step_cap,
                               scene_name=ep["scene_name"], camera_cfgs={"add_segmentation": True}, rgb_overlay_path=INP + ep["overlay"])
    cam = "overhead_camera" if "google_robot" in ep["robot"] else "3rd_view_camera"
    return env, cam


def reset(env, ep):
    import sapien.core as sapien
    from transforms3d.euler import euler2quat
    q = (sapien.Pose(q=euler2quat(0, 0, ep["robot_yaw"])) * sapien.Pose(q=[0, 0, 0, 1])).q
    opts = {"robot_init_options": {"init_xy": np.array(ep["robot_xy"]), "init_rot_quat": q},
            "obj_init_options": ({"init_xy": np.array(ep["obj"]["init_xy"])} if "init_xy" in ep["obj"] else {"episode_id": ep["obj"]["episode_id"]})}
    return env.reset(options=opts)
