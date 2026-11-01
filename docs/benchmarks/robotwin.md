# RoboTwin-50 (training benchmark)

RoboTwin 2.0, 50 bimanual tasks (aloha-agilex, `demo_randomized`), 10 episodes per task = 500 episodes. Episode `k` of a
task is the `k`-th valid RoboTwin seed >= 100000 of that task; the valid seeds are listed in
`harness/data/robotwin_seedcache/` (one file per task; seeds whose scene the expert can solve, from the RoboDawn seed
protocol). The planner sees the head and both wrist cameras (images of the last 2 turns), the arm state, the result of
its previous call and, at the first turn, a short task note (`harness/memory/robotwin/<task>.json`). Budget: 30 calls and
900 s wall clock per episode. Strict success S: the planner ends with `done` and RoboTwin's `check_success()` holds at
the end. The simulator success is never shown to the planner.

Code: `harness/bimanual/planner.py` (SYSTEM prompt, tools, planner loop) on top of `harness/bimanual/planner_base.py`,
`harness/bimanual/geometry.py`, `harness/bimanual/perception*.py`, `harness/bimanual/camera_capture.py`,
`harness/adapters/adapter_robotwin.py`, `harness/bimanual/scene_runner.py` and `harness/bimanual/scene/` (scene
construction), with the shared `harness/tools.py` and `harness/geom.py`. The adapter seals every privileged simulator query
(`_seal_privileged`: grasp/place plans, object poses, boxes, joints, ...); calling one raises.

## Install

1. RoboDawn with its pinned RoboTwin submodule. Paper runs: RoboDawn `60d71ec840105339f58ecd15633d1c0e50bb90e2`,
   RoboTwin submodule `96c1feab536306b50c26af200044fcdf126e8904`.
   ```bash
   mkdir -p third_party && cd third_party
   git clone https://github.com/Hugo-AGI/RoboDawn.git && cd RoboDawn
   git checkout 60d71ec840105339f58ecd15633d1c0e50bb90e2 && git submodule update --init RoboTwin
   ```
2. Follow RoboTwin's own installation (conda env `RoboTwin`, Python 3.10, SAPIEN 3.0.0b1, mplib 0.2.1, torch 2.7.1 in
   the paper runs) and download its assets (embodiments, objects, textures) as described in the RoboTwin README.
   SAPIEN renders with Vulkan: the NVIDIA ICD must be visible (`/etc/vulkan/icd.d/nvidia_icd.json` or `VK_ICD_FILENAMES`)
   and `libvulkan.so.1` must be on the library path.
3. Into the same env: `pip install -r requirements.txt` (repository root).
4. A SAM3 server with the `/seg` API (`serving/sam3/sam3_gpu.sh` or `sam3_npu.sh`, see docs/serving.md).

## Environment

| variable | meaning | default |
|---|---|---|
| `ROBODAWN_REPO` | RoboDawn checkout (also read as `ROBODAWN_ROOT`) | `$TINYASTRAL_THIRD_PARTY/RoboDawn` |
| `PERC_SAM3` | SAM3 `/seg` server | `http://127.0.0.1:<8130 + first CUDA_VISIBLE_DEVICES index>` |
| `TINYASTRAL_URL` | planner endpoint (or `--url`) | `http://127.0.0.1:8000/v1` |
| `TINYASTRAL_OUT` | output root | `<repo>/runs` |
| `HARNESS_MEMORY=0` | disable the task note (ablation only) | on |
| `HARNESS_CAP`, `HARNESS_WALL` | call cap / wall clock (or `--cap`, `--wall`) | 30 / 900 |

## Run

```bash
conda activate RoboTwin
export ROBODAWN_REPO=$PWD/third_party/RoboDawn PERC_SAM3=http://127.0.0.1:8130 TINYASTRAL_URL=http://127.0.0.1:8000/v1
# one episode
python harness/run.py robotwin --task adjust_bottle --episode 0 --lane tinyastral4b --model qwen
# all 500 episodes (resumable; each episode runs in its own process)
python eval/run_robotwin.py --lane tinyastral4b --workers 4
python eval/score.py robotwin runs/tinyastral4b
```

Each episode writes `runs/<lane>/<task>/shard_<k>/results.json` and `episode_<k>/{episode.json, model_requests.jsonl, img/}`
(`model_requests.jsonl` = the exact request text with image digests).

## Notes

- The instruction wording is sampled by RoboTwin from the task's description templates with Python's global RNG, so the
  same episode can be phrased differently between runs; scene, objects and seed are fixed.
- Rendering is not bit-exact between runs and the measured gripper positions in the `Arms:` line can differ by 0.1 cm.
- Wall-clock limits make outcomes depend on the serving speed: the paper runs used one bf16 replica per GPU (vLLM).
