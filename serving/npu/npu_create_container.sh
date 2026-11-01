#!/bin/bash
# Create the inference container on a Huawei Ascend Atlas A2 node. Needed once, and again after the node reboots.
#
# The container gets exactly ONE NPU mapped in (HOST_DEVICE), so inside the container
# ASCEND_RT_VISIBLE_DEVICES is always 0, whichever physical card was mapped.
#
# usage: bash npu_create_container.sh [HOST_DEVICE] [PORT]      defaults: /dev/davinci0, 8000
# env:   TA_CONTAINER (default tinyastral), TA_IMAGE (SGLang Ascend image), TA_WORK (dir bind-mounted at the same path, default $HOME)
set -eu
DEV=${1:-${TA_NPU_DEV:-/dev/davinci0}}
PORT=${2:-${TA_PORT:-8000}}
NAME=${TA_CONTAINER:-tinyastral}
IMG=${TA_IMAGE:-quay.io/ascend/sglang:main-cann9.0.0-910b}
WORK=${TA_WORK:-$HOME}

docker run -d --name ${NAME} \
  --device ${DEV} --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /var/queue_schedule:/var/queue_schedule \
  -v ${WORK}:${WORK} \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -p 127.0.0.1:${PORT}:${PORT} \
  --shm-size 32g \
  ${IMG} /bin/bash -c 'sleep infinity'

# The image does not ship libOSMesa, which MuJoCo's software renderer needs (only if simulations run in this container).
docker exec ${NAME} bash -lc 'apt-get update -qq && apt-get install -y -qq libosmesa6 libosmesa6-dev'

docker exec -e ASCEND_RT_VISIBLE_DEVICES=0 ${NAME} \
  python3 -c "import torch, torch_npu; print('NPUs visible:', torch.npu.device_count(), torch.npu.get_device_name(0))"
echo "created ${NAME} with ${DEV} (= local index 0), host port 127.0.0.1:${PORT}"
