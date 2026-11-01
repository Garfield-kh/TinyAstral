#!/bin/bash
# Create the SAM3 perception conda env for Huawei Ascend Atlas A2 (torch_npu).
# SAM3_SRC must point at a SAM3 source checkout; apply the source changes described in docs/npu.md ("SAM3 on Ascend") first.
set -euo pipefail
CONDA=${TA_CONDA:-$HOME/miniconda3}
ENV=${TA_SAM3_ENV:-ta_sam3}
SRC=${SAM3_SRC:-$HOME/sam3_src}
source $CONDA/etc/profile.d/conda.sh
conda create -y -n $ENV -c conda-forge --override-channels python=3.11 2>&1 | tail -2
conda activate $ENV; unset PYTHONPATH
pip install --no-cache-dir torch==2.10.0 torch_npu==2.10.0 torchvision==0.25.0 2>&1 | tail -1
pip install --no-cache-dir "numpy<2" timm==1.0.26 ftfy==6.1.1 iopath==0.1.10 regex pycocotools einops tqdm huggingface_hub fastapi uvicorn opencv-python-headless scipy pyyaml decorator attrs psutil 2>&1 | tail -1
cd $SRC && pip install -e . --no-deps 2>&1 | tail -1
python -c "import torch,torch_npu,timm;print(\"ENV_OK\",torch.__version__,torch_npu.__version__,timm.__version__)"
