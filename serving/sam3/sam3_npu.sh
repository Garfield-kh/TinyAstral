#!/bin/bash
# SAM3 segmentation server (/seg API, perception/sam3_srv_npu.py) on Huawei Ascend Atlas A2.
#
#   SAM3_CKPT=/path/to/sam3.pt bash serving/sam3/sam3_npu.sh [PORT]      default 8134
#
# Sources serving/npu/sam3_npu_env.sh (CANN toolkit + conda env). Pick the card with ASCEND_RT_VISIBLE_DEVICES.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=${1:-8134}
: "${SAM3_CKPT:?set SAM3_CKPT to the downloaded sam3.pt}"
set +u; source "$ROOT/serving/npu/sam3_npu_env.sh"; set -u
exec python "$ROOT/perception/sam3_srv_npu.py" --port "$PORT" --ckpt "$SAM3_CKPT" --device npu --host "${SAM3_HOST:-127.0.0.1}"
