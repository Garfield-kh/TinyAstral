#!/bin/bash
# Serve a TinyAstral student (4B / 2B / 0.8B) with vLLM on one NVIDIA GPU.
#
#   bash serving/gpu/serve_student_vllm.sh MODEL_DIR [PORT] [extra vllm args...]
#
#   MODEL_DIR  local Hugging Face directory, e.g. a download of kehong/TinyAstral-weights/4b
#   PORT       default 8000  (harness default: TINYASTRAL_URL=http://127.0.0.1:8000/v1)
#
# Env:  GPU (CUDA device index, default 0), GPU_UTIL (gpu-memory-utilization, default 0.85),
#       SERVED_NAME (default qwen: the harness CLIs default to --model qwen), HOST (default 127.0.0.1).
#
# These are the serving settings used for the paper's student evaluations: bf16, 32k context,
# 16 concurrent sequences, up to 8 images per request.  The harness itself never sends more than
# K recent images (K=3 for LIBERO/Guava, 2 turns x 3 cameras for RoboTwin), so 8 is enough.
# No reasoning parser: the harness reads the raw completion and splits at </think> itself.
# On a GPU shared with other jobs, vLLM's memory profiling can count their allocations against this server and
# fail with "No available memory for the cache blocks"; pass a fixed KV-cache size instead, e.g.
#   bash serving/gpu/serve_student_vllm.sh MODEL_DIR 8000 --kv-cache-memory-bytes 12G
set -eu
MODEL=${1:?usage: serve_student_vllm.sh MODEL_DIR [PORT]}
PORT=${2:-8000}
shift $(( $# >= 2 ? 2 : 1 ))
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# vLLM 0.29 JIT-compiles FlashInfer kernels with nvcc at the first start. Its torch is built with CUDA 13.0, so an older
# system toolkit (/usr/local/cuda 12.x) fails ("nvcc fatal: Unknown option '--compress-mode=size'"). Unless CUDA_HOME is
# set, use the pip CUDA 13.0 toolkit next to vLLM (site-packages/nvidia/cu13, nvcc pinned in INSTALL.md) through a small
# CUDA_HOME view with the lib64/ and libcudart.so names the JIT build links against.
if [ -z "${CUDA_HOME:-}" ]; then
  _CU=$("$(dirname "$(command -v vllm)")/python" -c 'import nvidia, os; print(os.path.join(list(nvidia.__path__)[0], "cu13"))' 2>/dev/null || true)
  if [ -n "$_CU" ] && [ -x "$_CU/bin/nvcc" ]; then
    _SH=${TINYASTRAL_CUDA_HOME_VIEW:-${XDG_CACHE_HOME:-$HOME/.cache}/tinyastral/cuda13}
    mkdir -p "$_SH/lib64"
    for d in bin include nvvm; do ln -sfn "$_CU/$d" "$_SH/$d"; done
    for f in "$_CU"/lib/*; do ln -sfn "$f" "$_SH/lib64/$(basename "$f")"; done
    ln -sfn libcudart.so.13 "$_SH/lib64/libcudart.so"
    export CUDA_HOME=$_SH PATH=$_SH/bin:$PATH
  fi
fi
exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-qwen}" \
  --host "${HOST:-127.0.0.1}" --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --limit-mm-per-prompt '{"image":8}' \
  "$@"
