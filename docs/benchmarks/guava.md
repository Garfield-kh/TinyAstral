# Guava (training benchmark)

Guava's single-arm Panda tabletop suite (MuJoCo / robosuite), 15 tasks x 30 episodes = 450; the first 8 tasks of the list
are Guava's unseen tasks. Episode `k` of task `i` (order in `eval/episodes.py`) uses seed `42 + 30 i + k`; push tasks
alternate left/right with `k`; the camera is `sideview` for `shell_game` and `frontview` otherwise (512 x 512); call cap
30 for bin_and_tray_simple / set_table / shell_game / red_objects_in_basket and 25 for the others; 20 settle steps.
Perception, grasping (PCA top-down grasp) and arm control are Guava's own; GraspGen is not used. Strict success S: the
planner ends with `done` and Guava's task success holds at the end.

Code: `harness/adapters/guava_protocol.py` (protocol, engine construction), `harness/single_arm/prompts.py`
(`GUAVA_SYSTEM`), `harness/adapters/adapter_guava.py`, `harness/single_arm/planner.py` (planner loop, tool layer, episode
driver), `harness/single_arm/flags.py` (`--gen default` = post_release, honest_done, fail_post), `harness/tools.py`,
`harness/geom.py`.

What the adapter uses from Guava: `guava.inference.rollout_env.RolloutEngine` (built from `configs/<task>.yaml`), the
robot / controller of its env, `ctx.perception` (Guava's SAM3 client and depth unprojection,
`guava.tools.tools._unproject_pointclouds`) and the PCA grasp of `guava.tools.tools`.

## Install

1. Guava. Paper runs: https://github.com/hdacnw/guava-release at `8eeed908bef4fb63c8bfc7c4e2fa2f5684eb067f` (with its
   `third_party/robosuite` and `third_party/sam3`).
   ```bash
   mkdir -p third_party && git -C third_party clone https://github.com/hdacnw/guava-release.git guava
   git -C third_party/guava checkout 8eeed908bef4fb63c8bfc7c4e2fa2f5684eb067f
   ```
2. Create the evaluation env following Guava's README (paper runs: Python 3.11, mujoco 3.6.0, robosuite 1.5.2 from
   Guava's third_party), then `pip install -r requirements.txt` (repository root).
3. Guava's SAM3 server (`/segment` API) in its own env with Guava's `third_party/sam3`: `serving/sam3/sam3_guava.sh`
   (see docs/serving.md). It loads the SAM3 weights through Hugging Face (`facebook/sam3`, gated; accept the SAM License).

## Environment

| variable | meaning | default |
|---|---|---|
| `GUAVA_REPO` | Guava checkout; episodes run with this as working directory and on `PYTHONPATH` | `$TINYASTRAL_THIRD_PARTY/guava` |
| `GUAVA_SAM3_URL` | Guava SAM3 server (or `--sam3`) | `http://127.0.0.1:8314` |
| `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID`, `PYOPENGL_PLATFORM=egl` | headless rendering | egl (set by `eval/run_guava.py`) |
| `TINYASTRAL_URL` | planner endpoint (or `--url`) | `http://127.0.0.1:8000/v1` |

## Run

```bash
conda activate guava
export GUAVA_REPO=$PWD/third_party/guava GUAVA_SAM3_URL=http://127.0.0.1:8314 MUJOCO_EGL_DEVICE_ID=0
# one task, 5 episodes (what eval/run_guava.py runs per process)
(cd $GUAVA_REPO && PYTHONPATH=$GUAVA_REPO python $OLDPWD/harness/run.py guava --gen default --task pick_up_carrot \
     --seed-base 42 --episodes 5 --offset 0 --out $OLDPWD/runs/guava_tinyastral4b --model qwen --url $TINYASTRAL_URL --thinking 1)
# all 450 episodes (resumable)
python eval/run_guava.py --label tinyastral4b --workers 4
python eval/score.py guava runs/guava_tinyastral4b
```

Each episode writes `<task>__k<k>_s<seed>/{result.json, episode.json, harness_info.json, img/}`.
