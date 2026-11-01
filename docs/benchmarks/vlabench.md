# VLABench (zero-shot)

VLABench is one of the three zero-shot benchmarks: it was never used for harness search or training. The paper uses
the **first 10 official evaluation episodes of every task** in five tracks, 500 episodes in total:

- `track_1_in_distribution`
- `track_2_cross_category`
- `track_3_common_sense`
- `track_4_semantic_instruction`
- `track_6_unseen_texture`

Episodes come from `VLABench/configs/evaluation/tracks/<track>.json`. The planner gets the LIBERO single-arm prompt
and tool set of H\*. One frame sentence is changed to describe the VLABench robot-base axes. The planner input is only:

- the official instruction
- the camera image
- the gripper state
- the previous tool result

Code: `zeroshot/vlabench/`

| file | role |
|---|---|
| `run_vlabench.py` | episode driver (`--track --task --episodes`); `--probe` renders one reset without a model |
| `env_vlabench.py` | builds an official evaluation episode as the official evaluator does. Exposes only robot-side primitives: proprioception, Cartesian waypoints through dm_control IK, gripper, RGB-D cameras. Checks the official success condition after every `env.step` |
| `adapter_vlabench.py` | H\* tool-layer adapter (a `LiberoAdapter` from `harness/adapters/adapter_libero.py` with VLABench simulator access) |
| `egl_nvidia.json` | EGL vendor file pointing to the driver's `libEGL_nvidia.so.0` |
| `run_all.py` | local worker pool over the 500-episode set. Resumable |
| `score_vlabench.py` | official measure from the `result.json` files |

## Install

1. Clone and install VLABench (paper runs: commit `cf588fe60c0c7282174fe979f5913170cfe69017`):
   ```bash
   mkdir -p third_party && cd third_party
   git clone https://github.com/OpenMOSS/VLABench.git && git -C VLABench checkout cf588fe60c0c7282174fe979f5913170cfe69017
   conda create -y -n zs_vlabench python=3.10 && conda activate zs_vlabench
   cd VLABench && pip install -r requirements.txt && pip install -e . && python scripts/download_assets.py
   ```
   The paper environment had mujoco 3.2.2, dm_control 1.0.22, numpy 1.25.0 and open3d 0.18.0. Also install
   `zeroshot/requirements.txt` and the H\* requirements listed at the repository root.
2. Set up headless EGL rendering on an NVIDIA GPU:
   ```bash
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   export __EGL_VENDOR_LIBRARY_FILENAMES=$PWD/zeroshot/vlabench/egl_nvidia.json   # only if the system has no NVIDIA EGL vendor file
   export MUJOCO_EGL_DEVICE_ID=0
   ```

## What the code expects from the upstream package

No VLABench source is bundled. `env_vlabench.py` uses only these upstream interfaces:

- **Episodes:** `VLABench/configs/evaluation/tracks/<track>.json`. `VLABench.envs.load_env(task, episode_config=cfg, random_init=False, eval=False, run_mode="eval")`
  is called after importing `VLABench.robots` and `VLABench.tasks`. This is the same call the official evaluator
  makes (`VLABench/evaluation/evaluator/base.py`).
- **Instruction:** `env.task.get_instruction()`.
- **Stepping:** `env.step(joint targets + finger targets)`. `timestep.last()` is the official success event.
- **Robot:**
  - `env.robot.get_qpos`
  - `env.robot.get_qpos_from_ee_pos` (dm_control IK, as in the official evaluator)
  - `end_effector_site`, `link_base`
  - the MJCF bodies `left_finger` / `right_finger` and joints `finger_joint1/2`
- **Cameras:** `env.physics.render(camera_id, depth=...)`, `physics.data.cam_xpos/cam_xmat`, `physics.model.cam_fovy`.
  The wrist camera is not used.
- **Official metrics:**
  - `env.get_task_progress()` and `env.get_intention_score(threshold=0.1)`
  - `configs/task_config.json[...].evaluation.max_episode_length`, found through `VLABench.configs.name2config` and
    `VLABench.utils.utils.find_key_by_value`

## Environment variables

| variable | meaning | default |
|---|---|---|
| `TINYASTRAL_URL` | OpenAI-compatible planner endpoint | `http://127.0.0.1:8000/v1` |
| `VLABENCH_ROOT` | the `VLABench/VLABench` package directory. VLABench also sets this itself when imported | derived from `VLABENCH_REPO` or the installed package |
| `VLABENCH_REPO` | VLABench clone | `$TINYASTRAL_THIRD_PARTY/VLABench` |
| `TINYASTRAL_OUT` | results root | `<repo>/runs` |
| `LIBERO_SAM3_URL` | SAM3 `/seg` server (`serving/sam3/sam3_gpu.sh`), same as `--sam3` | `http://127.0.0.1:8134` (`--sam3` overrides) |
| `MUJOCO_EGL_DEVICE_ID` | EGL render device | — |
| `ZS_THREADS` | OMP/MKL/BLAS threads per episode process | `2` |
| `HARNESS_GEN` | harness generic-improvement set if `--gen` is omitted | `default` |

## Run

```bash
export TINYASTRAL_URL=http://127.0.0.1:8000/v1
bash serving/sam3/sam3_gpu.sh 8164 1 &
cd zeroshot/vlabench
# one episode
python run_vlabench.py --track track_3_common_sense --task select_fruit --episodes 0 --sam3 http://127.0.0.1:8164 \
       --out $TINYASTRAL_OUT/zeroshot/vlabench/student
# the 500-episode paper set, 6 workers on EGL devices 0 and 1
python run_all.py --out $TINYASTRAL_OUT/zeroshot/vlabench/student --workers 6 --gpus 0,1 --sam3 http://127.0.0.1:8164
```

Paper settings (all are the defaults):

- `--thinking 1`
- `--max-tokens 4096`
- `--xml-calls 1`
- `--gen default`
- 40 tool calls per episode, no wall-clock limit

Each episode writes `<out>/<track>/<task>/eKK/` containing `result.json`, `episode.json`, `harness_info.json` and
`img/`. A crashed episode writes `error.json`, which counts as a failure.

## Score

```bash
python score_vlabench.py --out $TINYASTRAL_OUT/zeroshot/vlabench/student
```

**Official measure (paper):** the official success event, `result.json["success"]`. This is the task's termination
condition, checked after every `env.step`. The episode stops when it fires, and the planner never sees it. VLABench
defines no final-state predicate after `done`.

- **Track SR** is the arithmetic mean of the per-task SRs.
- **Overall** is the mean of the five track SRs. Only `e00`–`e09` are scored; `--max-ep 0` scores everything present.

The scorer also prints the following. They are not paper measures:

- `success_strict`: success within the official `max_episode_length`. This is close to 0 for any tool-level agent,
  because one grasp takes thousands of `env.step`s.
- the official `progress_score` and `intention_score`, which are stored in `result.json`

## Verify the request bytes

`tests/equiv_zeroshot.py --bench vlabench` runs one fixed episode against a local stub planner. The stub records the
step-0 request and then ends the episode. The script compares the request with the SHA-256 digests of the paper code's
request, stored in `tests/zeroshot_ref_requests.json`:

- the text-only view must match;
- the full bytes, which include the rendered JPEG, match when the renderer build is the same.

Run it inside the benchmark environment described above.
