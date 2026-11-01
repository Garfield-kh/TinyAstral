# Serving

A TinyAstral run needs two services next to the simulator:

```
simulator + harness (eval/, zeroshot/)
   |-- OpenAI-compatible chat endpoint (planner model)   TINYASTRAL_URL, --model qwen
   `-- SAM3 segmentation server (perception)             PERC_SAM3 / LIBERO_SAM3_URL / GUAVA_SAM3_URL / --sam3
```

All scripts bind to `127.0.0.1` by default (`HOST` / `SAM3_HOST` to change it). The SAM3 servers have no
authentication; expose them only on a trusted network.

## 1. Model weights

Student weights: Hugging Face `kehong/TinyAstral-weights`, sub-folders `4b/`, `2b/`, `0.8b/` (to be published).
Each is a standard Hugging Face directory (safetensors, config, tokenizer, `chat_template.jinja`, processor config).

```bash
huggingface-cli download kehong/TinyAstral-weights --include "4b/*" --local-dir ./weights
```

## 2. Planner model (NVIDIA GPU, vLLM)

Tested with vLLM 0.29.

```bash
# student: one GPU, bf16, 32k context, 16 concurrent sequences, <= 8 images per request
GPU=0 GPU_UTIL=0.85 bash serving/gpu/serve_student_vllm.sh ./weights/4b 8000
until curl -s http://127.0.0.1:8000/health >/dev/null; do sleep 5; done
export TINYASTRAL_URL=http://127.0.0.1:8000/v1
```

The served model name is `qwen` (`SERVED_NAME` to change). Every harness CLI defaults to `--model qwen`, so the
request bodies are identical to those of the paper runs.

Optional sanity check that a checkpoint renders prompts exactly like its base model (catches a wrong
`chat_template.jinja`, e.g. one that keeps `<think>` in history turns):

```bash
python serving/gpu/check_chat_template.py Qwen/Qwen3.5-4B ./weights/4b [--data val.jsonl --n 20]
```

### Teacher / any other planner

The harness works with any OpenAI-compatible endpoint that accepts images and `chat_template_kwargs`
(a hosted API, SGLang, vLLM). `serving/gpu/serve_teacher_vllm.sh` is the vLLM configuration we used for a large
FP8 MoE teacher on 4 x 80 GB GPUs (TP4 + expert parallel, MTP speculative decoding with 3 tokens, 65,536-token
context, prefix caching, `--reasoning-parser qwen3`). Teacher weights are not part of this release.

```bash
GPUS=0,1,2,3 bash serving/gpu/serve_teacher_vllm.sh /path/to/teacher 8001
```

## 3. What the harness sends

```json
{"model": "qwen", "messages": [...], "temperature": 0.0, "max_tokens": 4096,
 "chat_template_kwargs": {"enable_thinking": true}}
```

* `--url` (default `$TINYASTRAL_URL`), `--model` (default `qwen`), `--thinking 1|0` -> `enable_thinking`. The students
  were evaluated with `--thinking 1`. A retry after an unparsable answer uses temperature 0.6.
* The reply is read from `message.content` (and `reasoning_content` if the server splits it). The students emit
  `...reasoning...</think>\n\n<tool_call>{...}</tool_call>`; the opening `<think>` comes from the chat template.
  Serve students **without** a reasoning parser.
* History handling is fixed by the harness and matches the training data: assistant turns in the history keep
  only the tool call (text after `</think>`), and only the last K user turns keep their images
  (K = 3 turns x 1 image for LIBERO-PRO / Guava, K = 2 turns x 3 cameras for RoboTwin). Do not change K.

## 4. SAM3 perception

Two different SAM3 servers are in use, with different HTTP APIs:

| Server | API | Launcher |
|---|---|---|
| `perception/sam3_srv.py` (CUDA), `perception/sam3_srv_npu.py` (Huawei Ascend Atlas A2) | `POST /seg {"image_b64", "prompts": [...], "top_k", "min_score"}` -> `{"results": {prompt: [{"score", "box", "mask"}]}}` | `serving/sam3/sam3_gpu.sh`, `serving/sam3/sam3_npu.sh` |
| Guava's `guava.scripts.sam3_server` (upstream Guava repo) | `POST /segment {"image_b64", "text_prompt", "top_k"}` -> `{"masks": [{"mask", "score"}]}` | `serving/sam3/sam3_guava.sh` |

Which benchmark talks to which:

| Benchmark | Server | How the URL is set | Default |
|---|---|---|---|
| RoboTwin | `/seg` (sam3_srv) | env `PERC_SAM3` | `http://127.0.0.1:<8130 + first CUDA_VISIBLE_DEVICES index>` (8130 if unset) |
| LIBERO-PRO | `/seg` (sam3_srv) | env `LIBERO_SAM3_URL` | `http://127.0.0.1:8134` |
| Guava | `/segment` (Guava server) | `--sam3` or env `GUAVA_SAM3_URL` | `http://127.0.0.1:8314` |
| SimplerEnv (zero-shot) | `/seg` (sam3_srv) | `--sam3` | `http://127.0.0.1:8160` |
| VLABench (zero-shot) | `/seg` (sam3_srv) | `--sam3` (exported as `LIBERO_SAM3_URL`) | none |
| EB-Manipulation (zero-shot) | `/segment` (Guava server) | `--sam3` (required) | none |

### Installing SAM3

1. Request access to `facebook/sam3` on Hugging Face (SAM License) and download `sam3.pt`.
2. Install the SAM3 package from https://github.com/facebookresearch/sam3 (`pip install -e .` in a checkout; we
   used the copy that Guava pins as a submodule, version 0.1.0) plus `fastapi uvicorn pillow`.
3. Start one server per GPU as needed:

```bash
export SAM3_CKPT=/path/to/sam3.pt
bash serving/sam3/sam3_gpu.sh 8130 0          # RoboTwin episodes on GPU 0 (PERC_SAM3 default when CUDA_VISIBLE_DEVICES=0)
bash serving/sam3/sam3_gpu.sh 8134 0          # LIBERO-PRO
bash serving/sam3/sam3_guava.sh 8314 0        # Guava / EB-Manipulation (run in Guava's environment)
curl -s http://127.0.0.1:8130/health
```

`sam3_srv.py` loads the checkpoint offline (`--ckpt` / `$SAM3_CKPT`). Guava's server builds SAM3 with upstream
defaults and fetches `facebook/sam3` from the hub (log in, or pre-populate `HF_HOME` and set `HF_HUB_OFFLINE=1`).
One server serializes requests with a lock; in our runs one SAM3 server handled 4 (LIBERO-PRO) to 8 (Guava)
concurrent episodes.

## 5. Huawei Ascend Atlas A2

SGLang serving and the NPU SAM3 server are described in [npu.md](npu.md).
