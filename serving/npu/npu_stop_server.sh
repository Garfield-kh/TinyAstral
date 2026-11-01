#!/bin/bash
# List the exact server PIDs so they can be stopped one by one.  This script does not kill anything.
# Do not use pattern-based killing (pkill -f) on a shared node: the pattern can match your own session.
# usage: bash npu_stop_server.sh [PORT]   (default 8000)
set -u
PORT=${1:-${TA_PORT:-8000}}
CTR=${TA_CONTAINER:-tinyastral}
echo "--- launch_server pids inside ${CTR} ---"
docker exec ${CTR} bash -lc "ps -ef | grep launch_server | grep -- '--port ${PORT}' | grep -v grep" || true
echo
echo "Kill them one by one:   docker exec ${CTR} kill <PID>"
echo "SGLang leaves scheduler / TP workers in other process groups, so verify afterwards:"
echo "  npu-smi info | sed -n '/Process id/,\$p'      # the card must show no sglang process"
