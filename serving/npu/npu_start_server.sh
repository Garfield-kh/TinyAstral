#!/bin/bash
# Serve a TinyAstral student with SGLang on Huawei Ascend Atlas A2.
#
#   TA_MODEL=/path/to/tinyastral/4b bash npu_start_server.sh [PORT]        default port 8000
#
# Run this on the HOST; the script issues the `docker exec` itself. The container has one NPU
# mapped in (npu_create_container.sh), so ASCEND_RT_VISIBLE_DEVICES is 0 inside it.
# TA_MODEL must be a path visible inside the container (npu_create_container.sh bind-mounts TA_WORK).
# env: TA_CONTAINER (tinyastral), TA_SERVED_NAME (qwen, matches the harness default --model qwen),
#      TA_LOG (server log, default $HOME/sglang_<PORT>.log)
set -u
PORT=${1:-${TA_PORT:-8000}}
CTR=${TA_CONTAINER:-tinyastral}
MODEL=${TA_MODEL:?set TA_MODEL to the model directory}
NAME=${TA_SERVED_NAME:-qwen}
LOG=${TA_LOG:-$HOME/sglang_${PORT}.log}

if ! docker ps --format '{{.Names}}' | grep -qx "${CTR}"; then
  echo "container ${CTR} is not running -- run npu_create_container.sh first"; exit 1
fi
if docker exec ${CTR} bash -lc "curl -s -m 3 localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "something is already serving :${PORT} inside ${CTR}; abort"; exit 1
fi

# setsid + </dev/null detach the server from the calling shell's stdio.
setsid nohup docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 ${CTR} \
  python3 -m sglang.launch_server \
    --model-path ${MODEL} \
    --served-model-name ${NAME} \
    --host 0.0.0.0 --port ${PORT} \
    --device npu --attention-backend ascend --mm-attention-backend ascend_attn \
    --dtype bfloat16 --context-length 32768 \
    --max-running-requests 16 --mem-fraction-static 0.6 \
  > ${LOG} 2>&1 </dev/null &

echo "launched in ${CTR}, port ${PORT}; log: ${LOG}"
echo "Wait for weight loading and NPU graph capture, then check the health endpoint:"
echo "  until curl -s http://127.0.0.1:${PORT}/health >/dev/null; do sleep 5; done"
