#!/bin/bash
# Guava's own SAM3 server (/segment API, module guava.scripts.sam3_server from the upstream Guava repo).
# Used by the Guava benchmark (GUAVA_SAM3_URL / harness/run.py guava --sam3) and by EmbodiedBench (--sam3).
# It is a different API from perception/sam3_srv.py (/segment with one text_prompt per request), so it cannot
# be replaced by our /seg server.
#
#   bash serving/sam3/sam3_guava.sh [PORT] [GPU]        defaults: 8314, 0
#
# Run inside Guava's SAM3 environment (Guava README). The server builds SAM3 with the upstream defaults, i.e.
# it fetches facebook/sam3 from the Hugging Face hub (gated: log in with an account that accepted the
# SAM License, or pre-populate HF_HOME and set HF_HUB_OFFLINE=1).
set -eu
TP=${TINYASTRAL_THIRD_PARTY:-$(cd "$(dirname "$0")/../.." && pwd)/third_party}
GUAVA=${GUAVA_REPO:-$TP/guava}
PORT=${1:-8314}
export CUDA_VISIBLE_DEVICES=${2:-${GPU:-0}}
export PYTHONPATH="$GUAVA${PYTHONPATH:+:$PYTHONPATH}"
cd "$GUAVA"
exec python -u -m guava.scripts.sam3_server --host "${SAM3_HOST:-127.0.0.1}" --port "$PORT" --device cuda
