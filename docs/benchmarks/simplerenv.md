# SimplerEnv (zero-shot)

SimplerEnv is one of the three zero-shot benchmarks: it was never used for harness search or training. We run the
official **Visual Matching** set, 960 episodes: Google Robot (Pick Coke Can, Move Near, Open/Close Drawer, Open Top
Drawer and Place Apple) and WidowX (four Bridge tasks). The planner gets the single-arm Guava prompt and tool set of
H\*. Each tool call is executed through the official SimplerEnv controller with `env.step`.

Code: `zeroshot/simplerenv/`

| file | role |
|---|---|
| `run_simplerenv.py` | episode driver (one or more episodes of one task key) |
| `adapter_simplerenv.py` | robot / camera adapter for the H\* tool layer. Perception uses only the RGB and depth of the policy camera, plus SAM3 |
| `episodes_simplerenv.py` | the 960 official visual-matching configurations, with env build and reset copied from the official evaluator |
| `run_all.py` | local worker pool over the full set. Resumable: finished episodes are skipped |
| `score_simplerenv.py` | official measure from the `result.json` files |

## Install

1. Clone SimplerEnv with its ManiSkill2_real2sim submodule. The paper runs used SimplerEnv `eddb5690749a1e34267807c0f7b5eede323918d7`
   and ManiSkill2_real2sim `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`.
   ```bash
   mkdir -p third_party && cd third_party
   git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv.git
   git -C SimplerEnv checkout eddb5690749a1e34267807c0f7b5eede323918d7
   git -C SimplerEnv/ManiSkill2_real2sim checkout ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
   ```
2. Create the conda environment (Python 3.10):
   ```bash
   conda create -y -n zs_simpler python=3.10
   conda activate zs_simpler
   pip install "numpy<2" sapien==2.2.2 requests pillow scipy opencv-python-headless mediapy transforms3d
   pip install -e third_party/SimplerEnv/ManiSkill2_real2sim
   pip install -e third_party/SimplerEnv
   python -c "import sapien, mani_skill2_real2sim, simpler_env; print('OK', sapien.__version__)"
   ```
   Also install `zeroshot/requirements.txt` and the H\* requirements listed at the repository root.
3. Rendering uses SAPIEN 2.2.2 and Vulkan. You need the NVIDIA Vulkan ICD (`/etc/vulkan/icd.d/nvidia_icd.json` or
   `/usr/share/vulkan/icd.d/nvidia_icd.json`) and a Vulkan loader (`libvulkan.so.1`). The drawer and apple tasks use
   ray tracing, so they need a GPU. `SIMPLER_CPU_RENDER=1` switches to software rendering (lavapipe ICD in
   `VK_ICD_FILENAMES`), but this does not work for the ray-traced scenes.

## What the code expects from the upstream package

No SimplerEnv source is bundled. The driver and adapter use only these upstream interfaces:

- **Env registration and build**
  - `mani_skill2_real2sim.envs` registers the envs.
  - `simpler_env.utils.env.env_builder.build_maniskill2_env(env_name, obs_mode="rgbd", robot=..., control_mode=..., scene_name=..., rgb_overlay_path=..., camera_cfgs={"add_segmentation": True}, ...)`
    is called with the official visual-matching settings from `episodes_simplerenv.py`. The segmentation buffer is
    enabled only because the official RGB overlay needs it; the adapter never reads it.
- **Episode reset:** `env.reset(options={"robot_init_options": ..., "obj_init_options": ...})`, as in
  `simpler_env/evaluation/maniskill2_evaluator.py`.
- **Calls on `env.unwrapped`:**
  - `get_obs()`, which returns the policy camera RGB plus the `Position` buffer of `overhead_camera` (Google Robot) or
    `3rd_view_camera` (WidowX), and the camera parameters
  - `agent` (TCP pose, finger joints, controller config)
  - `step_action(...)` / `env.step(...)` with the official controller
  - `get_info()` / the per-step `info["success"]`
  - `get_language_instruction()`, `is_final_subtask()`, `advance_to_next_subtask()`
- **Assets:** the overlay images in `ManiSkill2_real2sim/data/real_inpainting/`.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `TINYASTRAL_URL` | OpenAI-compatible planner endpoint (`serving/gpu/serve_student_vllm.sh`) | `http://127.0.0.1:8000/v1` |
| `SIMPLERENV_REPO` | SimplerEnv clone. The overlay images are read from `ManiSkill2_real2sim/data/real_inpainting/` | `$TINYASTRAL_THIRD_PARTY/SimplerEnv` |
| `TINYASTRAL_THIRD_PARTY` | directory with upstream clones | `<repo>/third_party` |
| `TINYASTRAL_OUT` | results root | `<repo>/runs` |
| `SIMPLER_SAM3_URL` | SAM3 `/seg` server (`serving/sam3/sam3_gpu.sh`), same as `--sam3` | `http://127.0.0.1:8160` |
| `SIMPLER_CPU_RENDER` | `1` = software Vulkan renderer | `0` (GPU `cuda:0` of the visible devices) |
| `HARNESS_GEN` | harness generic-improvement set if `--gen` is omitted | `default` |

The served model name stays `qwen` (`--model qwen`), so the request bodies match the paper runs byte for byte.

## Run

```bash
export TINYASTRAL_URL=http://127.0.0.1:8000/v1
bash serving/sam3/sam3_gpu.sh 8160 1 &                 # SAM3 (/seg) on GPU 1
cd zeroshot/simplerenv
# one episode
python run_simplerenv.py --key coke_h --idx 0 --sam3 http://127.0.0.1:8160 --out $TINYASTRAL_OUT/zeroshot/simplerenv/student
# full 960-episode set, 4 workers rendering on GPUs 0 and 1
python run_all.py --out $TINYASTRAL_OUT/zeroshot/simplerenv/student --workers 4 --gpus 0,1 --sam3 http://127.0.0.1:8160
```

Paper settings (all are the defaults):

- `--thinking 1`
- `--max-tokens 4096`
- `--cap 25`: tool calls per episode, counted per subtask context
- `--gen default`
- XML tool-call fallback off

Each episode writes `<out>/<key>__<idx>/` containing `result.json`, `episode.json` (full trace including the SYSTEM),
`harness_info.json` and `img/`. A crashed episode writes `error.json`. `run_all.py` retries it on the next invocation.

## Score

```bash
python score_simplerenv.py --out $TINYASTRAL_OUT/zeroshot/simplerenv/student
```

**Official measure (paper):** `success_final`, the official SimplerEnv success flag at the moment the agent ends the
episode.

- A task's success rate (SR) pools all of its variants. Pick Coke Can has 3 orientations × 100 episodes. Open/Close
  Drawer has open and close, 108 episodes each.
- **Google VM** is the arithmetic mean of three task SRs: Pick Coke Can (300), Move Near (240) and Open/Close Drawer
  (216).
- **WidowX** is the mean of the four Bridge tasks, 24 episodes each.

The scorer also prints the following. They are not paper measures:

- `success_strict`: the official flag at the official step horizon, `min(agent end, max_episode_steps)`
- the 4-task Google average, which includes the apple task

For multi-subtask episodes (apple into drawer), a `done` on a non-final subtask advances the env to the next
subtask, as the official evaluator does on predicted termination. A fresh planner context then starts on the new
instruction.

## Verify the request bytes

`tests/equiv_zeroshot.py --bench simplerenv` runs one fixed episode against a local stub planner. The stub records the
step-0 request and then ends the episode. The script compares the request with the SHA-256 digests of the paper code's
request, stored in `tests/zeroshot_ref_requests.json`:

- the text-only view must match;
- the full bytes, which include the rendered JPEG, match when the renderer build is the same.

Run it inside the benchmark environment described above.
