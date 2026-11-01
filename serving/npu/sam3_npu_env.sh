#!/bin/bash
# Source this before launching the SAM3 service on Huawei Ascend Atlas A2: CANN toolkit + the conda env, in that order.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CANN_LD="$LD_LIBRARY_PATH"
source ${TA_CONDA:-$HOME/miniconda3}/etc/profile.d/conda.sh
conda activate ${TA_SAM3_ENV:-ta_sam3}; unset PYTHONPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CANN_LD:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver"
