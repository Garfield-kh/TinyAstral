# Installation

TinyAstral needs three kinds of processes; they can run on one machine or on several:

| process | environment | hardware |
|---|---|---|
| planner model | vLLM (GPU) or SGLang (Huawei Ascend Atlas A2), any OpenAI-compatible server | 1 GPU / NPU per replica (bf16; 4B fits in < 24 GB) |
| SAM3 server | PyTorch + the `sam3` package (or Guava's SAM3 server for Guava / EB) | 1 GPU / NPU (shared by several episode workers) |
| episode workers | one conda env per benchmark (the upstream simulator's env + `requirements.txt`) | CPU + a GPU for rendering (SAPIEN / MuJoCo EGL) |

## 1. This repository

```bash
git clone <this repository> tinyastral && cd tinyastral
mkdir -p third_party runs weights
```

The harness has no install step: scripts put the `harness/` directories on `sys.path` themselves. Paths are configured with environment
variables (defaults in brackets):

| variable | meaning |
|---|---|
| `TINYASTRAL_URL` | planner endpoint [`http://127.0.0.1:8000/v1`] |
| `TINYASTRAL_OUT` | episode outputs [`<repo>/runs`] |
| `TINYASTRAL_THIRD_PARTY` | upstream benchmark checkouts [`<repo>/third_party`] |
| `ROBODAWN_REPO`, `GUAVA_REPO`, `MOTORMIND_REPO` | RoboDawn (RoboTwin), Guava, MotorMind (LIBERO-PRO) checkouts [`$TINYASTRAL_THIRD_PARTY/{RoboDawn,guava,motormind}`] |
| `SIMPLERENV_REPO`, `VLABENCH_REPO`, `EMBODIEDBENCH_REPO` | zero-shot benchmark checkouts (see their docs) |
| `PERC_SAM3`, `LIBERO_SAM3_URL`, `GUAVA_SAM3_URL`, `SIMPLER_SAM3_URL`, `EB_SAM3_URL` | SAM3 endpoints per benchmark (see docs/serving.md) |
| `SAM3_CKPT` | path of `sam3.pt` for `perception/sam3_srv*.py` |

## 2. Planner model

* NVIDIA GPU: vLLM 0.29.0 (the version of the paper runs) in its own env, then
  `bash serving/gpu/serve_student_vllm.sh weights/4b 8000`. Details: [docs/serving.md](docs/serving.md).
  ```bash
  conda create -y -n tinyastral-vllm python=3.11 pip    # recent conda-forge Pythons come without pip unless it is listed
  conda activate tinyastral-vllm && pip install vllm==0.29.0
  # FlashInfer JIT-compiles kernels at the first start; its nvcc must match torch's CUDA 13.0 headers (pip would pick the
  # newest nvcc). serve_student_vllm.sh points CUDA_HOME at this toolkit unless CUDA_HOME is already set.
  pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-cccl==13.0.85"
  ```
* Huawei Ascend Atlas A2: SGLang Ascend container, `serving/npu/`. Details: [docs/npu.md](docs/npu.md).
* Check that the checkpoint renders prompts like the base model: `python serving/gpu/check_chat_template.py Qwen/Qwen3.5-4B weights/4b`.

## 3. SAM3

Request access to `facebook/sam3` on Hugging Face (SAM License), download `sam3.pt`, install the `sam3` package from
https://github.com/facebookresearch/sam3, then start `serving/sam3/sam3_gpu.sh` (CUDA) or `serving/sam3/sam3_npu.sh`
(Huawei Ascend Atlas A2). Guava and EB-Manipulation use Guava's own SAM3 server (`serving/sam3/sam3_guava.sh`).

## 4. Benchmarks

Install each benchmark from its upstream repository at the commit given in its document, then
`pip install -r requirements.txt` (and `zeroshot/requirements.txt` for the zero-shot ones) into that env:

* [RoboTwin-50](docs/benchmarks/robotwin.md) (RoboDawn + RoboTwin 2.0, SAPIEN / Vulkan)
* [Guava](docs/benchmarks/guava.md) (MuJoCo / robosuite)
* [LIBERO-PRO](docs/benchmarks/libero_pro.md) (MotorMind's LIBERO / LIBERO-PRO, robosuite 1.4)
* [SimplerEnv](docs/benchmarks/simplerenv.md), [VLABench](docs/benchmarks/vlabench.md), [EB-Manipulation](docs/benchmarks/embodiedbench.md)

No benchmark source is included in this repository.

## 5. Check the installation

```bash
# per benchmark env: runs one episode against a local stub endpoint and compares the step-0 request with the paper's
python tests/equiv_train.py --bench libero        # robotwin | guava | libero
python tests/equiv_zeroshot.py --bench simplerenv # simplerenv | vlabench | embodiedbench
```

Expected: `text-only MATCH`. Then run one real episode against the served model (README, Quick start).
