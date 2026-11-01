# Huawei Ascend Atlas A2

The students and the SAM3 perception server run on Huawei Ascend Atlas A2: the planner model with SGLang
(Ascend backend) in a container, SAM3 with `torch_npu` in a conda environment. The harness and the simulator
are unchanged; they only see the two HTTP endpoints.

Scripts: `serving/npu/`. Environment variables follow one convention:

| Variable | Default | Meaning |
|---|---|---|
| `TA_MODEL` | (required) | model directory, e.g. `weights/4b` from `kehong/TinyAstral-weights` |
| `TA_PORT` | `8000` | model service port |
| `TA_SERVED_NAME` | `qwen` | served model name (harness default `--model qwen`) |
| `TA_CONTAINER` | `tinyastral` | container name |
| `TA_IMAGE` | `quay.io/ascend/sglang:main-cann9.0.0-910b` | SGLang Ascend image |
| `TA_NPU_DEV` | `/dev/davinci0` | NPU device mapped into the container |
| `TA_WORK` | `$HOME` | directory bind-mounted into the container at the same path (put `TA_MODEL` under it) |
| `TA_CONDA`, `TA_SAM3_ENV` | `$HOME/miniconda3`, `ta_sam3` | conda install and SAM3 env name |
| `SAM3_CKPT` | (required) | path to `sam3.pt` |
| `TINYASTRAL_URL` | `http://127.0.0.1:8000/v1` | model endpoint used by the harness |

## 1. Model service (SGLang)

```bash
# once (and after a node reboot): container with one NPU, port published on 127.0.0.1
bash serving/npu/npu_create_container.sh /dev/davinci0 8000

# start the server (run on the host; the script calls docker exec)
TA_MODEL=$HOME/weights/4b bash serving/npu/npu_start_server.sh 8000
until curl -s http://127.0.0.1:8000/health >/dev/null; do sleep 5; done   # weight loading + graph capture
export TINYASTRAL_URL=http://127.0.0.1:8000/v1
```

Server settings: bf16, context 32768, 16 concurrent requests, `--mem-fraction-static 0.6`. Compared with a GPU
SGLang launch only three flags differ: `--device npu --attention-backend ascend --mm-attention-backend ascend_attn`.

Stop:

```bash
bash serving/npu/npu_stop_server.sh 8000      # prints the exact PIDs; kill them one by one with docker exec
npu-smi info | sed -n '/Process id/,$p'       # verify: scheduler / TP workers can live in other process groups
```

The stop script never kills by pattern on purpose (on a shared node a pattern can match your own session).

Connectivity check:

```bash
curl -s $TINYASTRAL_URL/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with the single word: ready"}],
       "temperature":0.0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 2. SAM3 on Ascend

SAM3 needs three small source changes under `sam3/model/` of a SAM3 checkout before it runs on NPU:

| File | Change |
|---|---|
| `decoder.py` | a device constant `_DEV` (`npu` if `torch.npu` is available, else `cuda`); use it for `device=` and `autocast(device_type=)`; on NPU skip the `sdpa_kernel(FLASH_ATTENTION)` backend selection and let `torch_npu` pick its fused kernel |
| `position_encoding.py` | the same `_DEV`, and `torch.zeros(..., device=_DEV)` |
| `sam3_tracker_utils.py` | import `edt_triton` lazily (triton is only needed by the video tracking path) |

Then build the environment (torch 2.10 + torch_npu 2.10, Python 3.11) and start the server:

```bash
SAM3_SRC=$HOME/sam3_src bash serving/npu/mkenv_sam3.sh
SAM3_CKPT=$HOME/models/sam3/sam3.pt ASCEND_RT_VISIBLE_DEVICES=1 bash serving/sam3/sam3_npu.sh 8134
curl -s http://127.0.0.1:8134/health
```

`serving/npu/sam3_npu_env.sh` sources the CANN toolkit and the conda env in the required order (the launcher
does this for you). The NPU server has the same `/seg` API as the CUDA one and also returns server-side timings.
Point LIBERO-PRO at it with `LIBERO_SAM3_URL=http://127.0.0.1:8134` (see [serving.md](serving.md) for the other
benchmarks). If the harness runs inside a container, `127.0.0.1` there is not the host: use an address the
container can reach.

## 3. Service smoke test and benchmark

Both tools replay recorded harness requests from a JSONL file, one request per line:

```json
{"ep": "t0_s0", "k": 3, "rec_prompt_tokens": 1913, "rec_completion_tokens": 210,
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                                           {"type": "text", "text": "..."}]}]}
```

The `steps[]` of an `episode.json` written by the harness contain the material to build this file.

```bash
# streams a few image-carrying requests; reports TTFT / decode rate and checks for a parsable <tool_call>
python3 serving/npu/smoke_service.py $TINYASTRAL_URL qwen reqs.jsonl --n 6

# fixed-length replay (recorded completion length, ignore_eos) for TTFT / decode / throughput comparisons
python3 serving/npu/bench_llm.py $TINYASTRAL_URL qwen reqs.jsonl out.jsonl --conc 4
```

Both work against any OpenAI-compatible server (GPU vLLM/SGLang as well).
