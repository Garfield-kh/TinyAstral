# LIBERO-PRO (training benchmark)

LIBERO-PRO through MotorMind's suite index and protocol (robosuite 1.4, single Panda, OSC_POSE): 300 episodes = the 30
original tasks of LIBERO-Spatial / Object / Goal (aggregate indices 0-9, 50-59, 100-109) and their 120 perturbed
variants (language, object, swap, task; indices 10-49, 60-99, 110-149), each with LIBERO initial states 0 and 1.
Budget: 450 s wall clock per episode (model, perception and simulation all counted; shown to the planner as time used).
Perception is camera-only: SAM3 masks on the agentview and frontview renders (512 px) back-projected with the rendered
depth; object names come from the planner, never from the simulator. Strict success S: the planner ends with `done` and
the BDDL success predicate holds at the end; an episode that runs out of wall clock is counted as unknown (U) and as a
failure in S/N.

Code: `harness/adapters/libero_protocol.py` (protocol), `harness/single_arm/prompts.py` (`LIBERO_SYSTEM`),
`harness/adapters/adapter_libero.py`, `harness/adapters/env_libero.py` and `harness/adapters/bp_core.py` (environment
port), `harness/single_arm/planner.py`, `harness/single_arm/flags.py` (`--gen default` = post_release, honest_done,
fail_post, inst_lock). Tool calls may also be written in the XML function-call format (`--xml-calls 1`, used for all reported runs).

What the port uses upstream: MotorMind's `adapter.libero.adapter.use_repo_libero()` and `adapter.libero.pro.register()`
(its vendored LIBERO / LIBERO-PRO BDDL and init files, not modified), LIBERO's `benchmark` / `OffScreenRenderEnv`, and
RoboDawn's `harness.core` command / environment base classes (`bp_core.py`).

## Install

1. MotorMind with LIBERO and LIBERO-PRO under `third_party/`. Paper runs:
   https://github.com/Motor-Mind/MotorMind-Code at `ad14459088b061b95c4a9f5d3bdd05998d47e673`.
2. RoboDawn (only its `harness/core` package is imported), see docs/benchmarks/robotwin.md step 1 (the RoboTwin submodule
   is not needed for LIBERO-PRO).
3. Create the env following MotorMind's README (paper runs: Python 3.10, robosuite 1.4.1, mujoco 2.3.7, bddl 1.0.1,
   torch 2.7.1), then `pip install -r requirements.txt` (repository root).
4. A SAM3 server with the `/seg` API (`serving/sam3/sam3_gpu.sh`, see docs/serving.md).

## Environment

| variable | meaning | default |
|---|---|---|
| `MOTORMIND_REPO` | MotorMind checkout | `$TINYASTRAL_THIRD_PARTY/motormind` |
| `ROBODAWN_REPO` | RoboDawn checkout | `$TINYASTRAL_THIRD_PARTY/RoboDawn` |
| `LIBERO_SAM3_URL` | SAM3 `/seg` server | `http://127.0.0.1:8134` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | LIBERO init states are loaded with `torch.load` | set by `eval/run_libero.py` |
| `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID` | headless rendering on a GPU (`osmesa` on CPU-only hosts) | egl |
| `PYTHONPATH=$MOTORMIND_REPO/third_party/LIBERO` | only if `import libero` fails (namespace package not exposed by an editable install) | - |

## Run

```bash
conda activate motormind
export MOTORMIND_REPO=$PWD/third_party/motormind ROBODAWN_REPO=$PWD/third_party/RoboDawn LIBERO_SAM3_URL=http://127.0.0.1:8134
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# one episode
python harness/run.py libero --gen default --task 57 --seed 1 --out runs/libero_tinyastral4b \
       --url $TINYASTRAL_URL --model qwen --thinking 1 --xml-calls 1
# all 300 episodes (resumable)
python eval/run_libero.py --label tinyastral4b --workers 4
python eval/score.py libero runs/libero_tinyastral4b
```

Each episode writes `t<task>_s<seed>/{result.json, episode.json, harness_info.json, img/}`. The wall-clock budget makes
results sensitive to endpoint latency; serve the planner on a dedicated GPU (the paper re-ran budget-expired episodes
once on dedicated endpoints).
