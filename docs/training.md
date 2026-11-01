# Training the students

The students are full fine-tunes of Qwen3.5-4B / 2B / 0.8B with [ms-swift](https://github.com/modelscope/ms-swift)
on decisions recorded from the teacher harness plus relabeled real-robot data.

## 1. Environment

Python 3.11, CUDA 12.8. Versions used for the released students are pinned in `train/requirements-train.txt`
(ms-swift 4.5.3, transformers 5.3.0, deepspeed 0.19.7, torch 2.11.0, flash-attn 2.8.3).

```bash
conda create -n tinyastral-train python=3.11 -y && conda activate tinyastral-train
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r train/requirements-train.txt
pip install flash-attn==2.8.3.post1 --no-build-isolation
# optional, faster linear-attention kernels:
pip install flash-linear-attention==0.5.2 fla-core==0.5.2
```

## 2. Data

Released dataset: Hugging Face `kehong/TinyAstral-SFT` (to be published), `train.jsonl` (83,753 rows) and
`val.jsonl` (4,443 rows), in the ms-swift multimodal chat format. Composition of the training split: about 41.9k
simulator decisions (RoboTwin, Guava, LIBERO-PRO teacher episodes, plus recovery branches) and about 41.9k
real-robot rows (judge 27,918 / plan 4,187 / next action 9,771) built from DROID, Fractal, Bridge V2, Berkeley UR5
and AgiBot World. Check the licenses of these source datasets before redistribution (AgiBot World is CC BY-NC-SA 4.0).

### Row format

```json
{"id": "L3|libero_90_0__s30|12",
 "messages": [{"role": "system",    "content": "<the harness SYSTEM prompt>"},
              {"role": "user",      "content": "Task: ...\nGripper: ...\nTime used: 0/450 s. Calls used: 0."},
              {"role": "assistant", "content": "\n\n<tool_call>\n{\"name\": \"locate\", \"arguments\": {...}}\n</tool_call>"},
              ...
              {"role": "user",      "content": "<image>TOOL RESULT: ..."},
              {"role": "assistant", "content": "<think>\n...reasoning...\n</think>\n\n<tool_call>\n{...}\n</tool_call>"}],
 "images": ["img/.../010_08.jpg", ...]}
```

Rules (these must hold or the student sees a different distribution at inference time):

* **One row = one decision**, not one episode. An episode with N planner calls yields N rows, each truncated at a
  different step. Loss is computed on the **last assistant turn only** (`--loss_scale last_round`).
* The last assistant turn carries the reasoning: `<think>\n...\n</think>\n\n` followed by the tool call. At inference
  the opening `<think>\n` is emitted by the chat template (`enable_thinking=true`) and the model generates the rest.
* **History assistant turns contain no reasoning**, only what follows `</think>` (`"\n\n<tool_call>...</tool_call>"`).
  The harness strips history the same way at inference.
* **Only the last K user turns keep images** (`<image>` placeholders, one per entry in `images`, in order).
  K = 3 turns x 1 image for LIBERO-PRO / Guava, K = 2 turns x 3 cameras (head, left wrist, right wrist) for
  RoboTwin. Older user turns keep their text only. If you change K in the harness you must re-export the data.
* Real-robot rows are single-turn (system, user with one image, assistant).
* The system prompt is the benchmark's harness SYSTEM text, byte-identical to what the harness sends.

## 3. Train

```bash
# 2B or 0.8B: one node, 8 GPUs (GA 4 -> global batch 32)
bash train/train.sh Qwen/Qwen3.5-2B /path/to/TinyAstral-SFT ./ckpt_2b

# 4B as released: 2 nodes x 8 GPUs (GA 2 -> global batch 32)
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0 ip> MASTER_PORT=29621 bash train/train.sh Qwen/Qwen3.5-4B /path/to/TinyAstral-SFT ./ckpt_4b   # node 0
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0 ip> MASTER_PORT=29621 bash train/train.sh Qwen/Qwen3.5-4B /path/to/TinyAstral-SFT ./ckpt_4b   # node 1
```

Recipe (identical for all three sizes):

| | |
|---|---|
| tuning | full fine-tuning, vision tower and aligner frozen |
| precision / parallelism | bf16, DeepSpeed ZeRO-2 (no offload), flash-attention, gradient checkpointing |
| optimizer | AdamW, lr 1e-5, cosine, warmup 0.05, weight decay 0.01, grad clip 1.0 |
| epochs | 2 (the released students are the end-of-epoch-2 checkpoints) |
| sequence | max_length 12288; longer rows are dropped (`--truncation_strategy delete`) |
| batch | per-device 1, global 32 (4B: 16 GPUs x GA 2; 2B and 0.8B: 8 GPUs x GA 4) |
| loss | last assistant turn (`last_round`; ms-swift appends `+ignore_empty_think` automatically for Qwen3.5) |
| images | `IMAGE_MAX_TOKEN_NUM=512`, `MAX_PIXELS=262144` |
| seed | 42 |

`train.sh` derives the gradient accumulation from the world size to keep the global batch at 32
(`GLOBAL_BATCH`, `BS`, `GA` override it). Roughly 5.2k optimizer steps for the full set.

## 4. After training

Serve the last checkpoint with `serving/gpu/serve_student_vllm.sh` and run
`serving/gpu/check_chat_template.py <base> <checkpoint>`; it must print `IDENTICAL`.
For a public upload keep only the safetensors, config, tokenizer, `chat_template.jinja`, processor and generation
configs (not optimizer states, `args.json` or `training_args.bin`, which contain local paths).

## 5. Data generation

Training rows come from teacher-harness episodes (simulators) and teacher relabeling of real-robot frames. The
export scripts are tied to our cluster layout and are not part of this release; the format above is the contract.
