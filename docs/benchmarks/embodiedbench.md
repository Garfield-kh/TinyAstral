# EB-Manipulation / EmbodiedBench (zero-shot)

EB-Manipulation is one of the three zero-shot benchmarks: it was never used for harness search or training. We run
all **228 official episodes**:

- `base`, `common_sense`, `complex` and `spatial`: 48 each
- `visual`: 36

The planner gets the Guava single-arm prompt of H\*, with two changes:

- the `articulate` line is removed
- a straight-line `move` tool is documented

The H\* tool layer drives the official `EBManEnv`.

- **Perception:** only the front-camera RGB, the camera's own point cloud, and SAM3. The benchmark's privileged inputs
  (ground-truth object coordinates, YOLO boxes, segmentation masks) are not used.
- **Actions:** every arm motion is one official 7-d discrete action through `EBManEnv.step`.

Code: `zeroshot/embodiedbench/`

| file | role |
|---|---|
| `run_eb.py` | episode driver (`--eval-set --indexes`) |
| `adapter_eb.py` | H\* tool-layer adapter on top of the official discrete action interface |
| `run_all.py` | local worker pool over the 228 episodes. Resumable |
| `score_eb.py` | official measure from the `result.json` files |

**Not included:** EmbodiedBench's own "native" baseline, which uses its official prompt, ground-truth coordinates and
YOLO boxes. That baseline uses Ultralytics YOLO (`yolo11n.pt`, AGPL-3.0), and neither its weights nor our runner for it
are part of this release. EB's `eb_man_utils.py` still constructs `YOLO("yolo11n.pt")` when it is imported, so
`ultralytics` must be installed. `run_eb.py` links `yolo11n.pt` into the run directory if your EmbodiedBench clone has
one, at `_assets/yolo11n.pt` or `yolo11n.pt`. Otherwise Ultralytics downloads it on first use. The harness never calls
YOLO.

## Install

Follow the upstream EB-Manipulation instructions (<https://github.com/EmbodiedBench/EmbodiedBench>, master branch; the
paper runs used a snapshot of master; no commit id was recorded). In short:

1. Clone EmbodiedBench (git-lfs) into `third_party/EmbodiedBench`. Also clone PyRep (<https://github.com/stepjam/PyRep>, master).
2. Install CoppeliaSim 4.1.0 (`CoppeliaSim_Pro_V4_1_0_Ubuntu20_04`, from the coppeliarobotics.com download area), then:
   ```bash
   export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
   export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
   export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
   ```
3. Create the conda environment. The paper environment used Python 3.9, numpy 1.26.4, scipy 1.13.1, open3d 0.18.0,
   torch 2.4.0 (CPU is enough) and ultralytics 8.3.61.
   ```bash
   conda create -y -n zs_embodiedbench python=3.9 && conda activate zs_embodiedbench
   pip install torch==2.4.0 torchvision==0.19.0
   pip install "numpy==1.26.4" "scipy==1.13.1" "cffi==1.14.2" "opencv-python-headless<4.11" "pillow==11.0.0" \
     "ultralytics==8.3.61" "gymnasium==1.0.0" "hydra-core==1.3.2" "omegaconf==2.3.0" "pydantic==2.10.4" \
     "pyquaternion==0.9.9" "transforms3d==0.4.2" natsort tqdm requests pyyaml "matplotlib==3.9.4" "open3d==0.18.0" \
     trimesh html-testRunner imageio
   cd third_party/PyRep && pip install -r requirements.txt && pip install -e . && cd -
   cd third_party/EmbodiedBench/embodiedbench/envs/eb_manipulation && pip install -e . --no-deps \
     && cp simAddOnScript_PyRep.lua $COPPELIASIM_ROOT/ && cd -
   cd third_party/EmbodiedBench && pip install -e . --no-deps && cd -
   # ultralytics pulls in opencv-python, whose Qt plugins break CoppeliaSim's xcb plugin: keep only the headless build
   pip uninstall -y opencv-python && pip install --force-reinstall --no-deps "opencv-python-headless<4.11"
   ```
   Also install `zeroshot/requirements.txt` and the H\* requirements listed at the repository root.
4. Get the dataset: `git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation`. Move its `data/` to
   `third_party/EmbodiedBench/embodiedbench/envs/eb_manipulation/data`.
5. Start a headless X server. CoppeliaSim renders on the CPU and all workers can share one display:
   ```bash
   Xvfb :91 -screen 0 1280x1024x24 +extension GLX -nolisten tcp &
   export DISPLAY=:91
   ```
6. If `import embodiedbench` or `amsolver` fails, add the clone to `PYTHONPATH`:
   ```bash
   export PYTHONPATH=$EMBODIEDBENCH_REPO:$EMBODIEDBENCH_REPO/embodiedbench/envs/eb_manipulation:$PYTHONPATH
   ```

## What the code expects from the upstream package

No EmbodiedBench source is bundled. The driver and adapter use only these upstream interfaces:

- **Env construction:** `embodiedbench.envs.eb_manipulation.EBManEnv.EBManEnv(eval_set, img_size=(500, 500), log_path, selected_indexes)`.
  It runs with the working directory set to the run folder, which contains a link `embodiedbench -> <clone>/embodiedbench`.
  EB resolves its task files relative to that folder.
- **Episode API:** `env.reset() -> (instruction, obs)`, `env.number_of_episodes`, `env.current_task_variation`.
- **Stepping:** `env.step(action)` with the official 7-d discrete action (100 position bins, 120 rotation bins). The
  per-step `info["task_success"]` is logged and never shown to the model. The env-feedback text is dropped.
- **Observation keys:**
  - `front_rgb` and `front_point_cloud` (also the shoulder / wrist variants)
  - `gripper_pose`, `gripper_open`, `gripper_joint_positions`, `misc`
  - `env.last_frame_obs`, `env.task`
- **Not used:** the privileged observation fields `object_informations`, `*_mask`, `task_low_dim_state`, and the YOLO
  boxes.

## SAM3 server

EB-Manipulation uses **Guava's SAM3 server** (module `guava.scripts.sam3_server` from the upstream Guava repo). Its
API is `POST /segment` with one `text_prompt` per request, so it is not the `/seg` server in `perception/`. Start it
with `serving/sam3/sam3_guava.sh` and pass its URL with `--sam3` or `EB_SAM3_URL`.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `TINYASTRAL_URL` | OpenAI-compatible planner endpoint | `http://127.0.0.1:8000/v1` |
| `EMBODIEDBENCH_REPO` | EmbodiedBench clone | `$TINYASTRAL_THIRD_PARTY/EmbodiedBench` |
| `TINYASTRAL_OUT` | results root | `<repo>/runs` |
| `EB_SAM3_URL` | Guava SAM3 `/segment` server, same as `--sam3` | — (required) |
| `DISPLAY` | X display for CoppeliaSim (Xvfb) | — |
| `COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH` | CoppeliaSim, as in the upstream README | — |
| `HARNESS_GEN` | harness generic-improvement set if `--gen` is omitted | `default` |

## Run

```bash
export TINYASTRAL_URL=http://127.0.0.1:8000/v1 EB_SAM3_URL=http://127.0.0.1:8830 DISPLAY=:91
bash serving/sam3/sam3_guava.sh 8830 1 &
cd zeroshot/embodiedbench
# two episodes of the base set
python run_eb.py --eval-set base --indexes 0,1 --tag student
# all 228 episodes, 8 CPU workers
python run_all.py --tag student --workers 8
```

Paper settings (all are the defaults):

- `--thinking 1`
- `--max-tokens 8192`
- `--max-calls 25`
- `--max-env-steps 60`
- `--xml-calls 1`
- `--gen default`

Results go to `<out-root>/<tag>/<eval_set>/epNNN/`, with `result.json`, `episode.json`, `harness_info.json` and
`img/`. The default `<out-root>` is `$TINYASTRAL_OUT/zeroshot/embodiedbench`.

## Score

```bash
python score_eb.py --run $TINYASTRAL_OUT/zeroshot/embodiedbench/student
```

**Official measure (paper):** `success_any`, the official per-step `task_success` latched over the episode within 60
env steps. It is pooled over all finished episodes, 228 for the full set. EB-Manipulation defines no final-state
predicate after `done`.

The scorer also prints the following. They are not paper measures:

- `success_15`: success within the official 15-step horizon
- per-subset numbers

## Verify the request bytes

`tests/equiv_zeroshot.py --bench embodiedbench` runs one fixed episode against a local stub planner. The stub records the
step-0 request and then ends the episode. The script compares the request with the SHA-256 digests of the paper code's
request, stored in `tests/zeroshot_ref_requests.json`:

- the text-only view must match;
- the full bytes, which include the rendered JPEG, match when the renderer build is the same.

Run it inside the benchmark environment described above.
