#!/bin/bash
# SAM3 segmentation server (/seg API, perception/sam3_srv.py) on one NVIDIA GPU.
# Used by RoboTwin (PERC_SAM3), LIBERO-PRO (LIBERO_SAM3_URL), SimplerEnv (--sam3) and VLABench (--sam3).
#
#   SAM3_CKPT=/path/to/sam3.pt bash serving/sam3/sam3_gpu.sh [PORT] [GPU]      defaults: 8130, 0
#
# Run inside an environment where `import sam3` works (see docs/serving.md).
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=${1:-8130}
export CUDA_VISIBLE_DEVICES=${2:-${GPU:-0}}
: "${SAM3_CKPT:?set SAM3_CKPT to the downloaded sam3.pt}"
exec python "$ROOT/perception/sam3_srv.py" --port "$PORT" --ckpt "$SAM3_CKPT" --host "${SAM3_HOST:-127.0.0.1}"
