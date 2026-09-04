#!/usr/bin/env bash
set -euo pipefail
RANK="${1:?rank 0|1}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-v4f-2dgx}"
MODEL_HOST="${MODEL_HOST:-/home/anemll/models/DeepSeek-V4-Flash-Vision-Exp}"
OVERLAY="${OVERLAY:-/tmp/sglang-vision-thin-overlay}"
DIST_ADDR="${DIST_ADDR:-10.200.0.1:25000}"
PORT="${PORT:-30000}"
NAME="sglang-vision-rank${RANK}"

docker rm -f "$NAME" 2>/dev/null || true
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true

docker run -d --name "$NAME" --gpus all \
  --network host --ipc=host \
  --ulimit memlock=-1:-1 \
  --cap-add IPC_LOCK \
  --device /dev/infiniband \
  --shm-size 32g \
  -v "${MODEL_HOST}:/models/DeepSeek-V4-Flash-Vision-Exp:ro" \
  -v "${OVERLAY}/models/deepseek_v4.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py:ro" \
  -v "${OVERLAY}/models/deepseek_v4_vl.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_vl.py:ro" \
  -v "${OVERLAY}/models/deepseek_v4_vit.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_vit.py:ro" \
  -v "${OVERLAY}/multimodal/processors/deepseek_v4_vl.py:/sgl-workspace/sglang/python/sglang/srt/multimodal/processors/deepseek_v4_vl.py:ro" \
  -e SGLANG_SM120_FLASHMLA_BACKEND=b12x \
  -e B12X_MLA_SM120_DSV4_H16_NATIVE=1 \
  -e SGLANG_OPT_FUSE_MHC_POST_PRE=1 \
  -e SGLANG_OPT_FP8_WO_A_GEMM=1 \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  -e SGLANG_B12X_MAX_TOKENS=8192 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_NET=IB \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" \
  sglang serve \
    --trust-remote-code \
    --model-path /models/DeepSeek-V4-Flash-Vision-Exp \
    --tp 2 \
    --nnodes 2 \
    --node-rank "$RANK" \
    --dist-init-addr "$DIST_ADDR" \
    --moe-runner-backend b12x \
    --disable-shared-experts-fusion \
    --weight-loader-drop-cache-after-load \
    --chunked-prefill-size 8192 \
    --context-length 327680 \
    --mem-fraction-static 0.80 \
    --swa-full-tokens-ratio 0.2 \
    --cuda-graph-max-bs-decode 8 \
    --disable-decode-cuda-graph \
    --max-running-requests 8 \
    --host 0.0.0.0 \
    --port "$PORT"

echo "started $NAME on $(hostname)"
