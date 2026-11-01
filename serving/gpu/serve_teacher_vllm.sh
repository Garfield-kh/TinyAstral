#!/bin/bash
# Serve a large teacher / planner model with vLLM (tensor + expert parallel, MTP speculative decoding).
#
#   bash serving/gpu/serve_teacher_vllm.sh MODEL_DIR [PORT] [extra vllm args...]
#
# The harness only needs an OpenAI-compatible /v1/chat/completions endpoint that accepts images and
# chat_template_kwargs.enable_thinking.  The teacher can be ANY such endpoint (a hosted API, SGLang,
# another vLLM setup); this script is just the configuration we used for an FP8 MoE teacher on
# 4 x 80 GB GPUs without native FP8 (the MoE layers then run as W8A16 Marlin kernels, which requires
# --enable-expert-parallel).  Teacher weights are not distributed with this repository; obtain them from
# the model provider under its own license.
#
# Env:  GPUS (default 0,1,2,3; TP size = number of GPUs), GPU_UTIL (default 0.85), MAX_SEQS (default 32),
#       NSPEC (MTP speculative tokens, default 3; set MTP=0 to disable), CTX (default 65536),
#       MAX_IMAGES (default 32), SERVED_NAME (default qwen), HOST (default 127.0.0.1).
# Point the harness at it with --url http://HOST:PORT/v1 --model qwen.
set -eu
MODEL=${1:?usage: serve_teacher_vllm.sh MODEL_DIR [PORT] [extra args]}
PORT=${2:-8001}
shift $(( $# >= 2 ? 2 : 1 ))
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
NG=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
SPEC=()
[ "${MTP:-1}" = 1 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NSPEC:-3}}")
exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-qwen}" \
  --tensor-parallel-size "$NG" --enable-expert-parallel \
  --max-model-len "${CTX:-65536}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --max-num-seqs "${MAX_SEQS:-32}" \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES:-32}}" \
  --trust-remote-code \
  --host "${HOST:-127.0.0.1}" --port "$PORT" \
  "${SPEC[@]}" "$@"
