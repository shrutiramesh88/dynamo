#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ -z ${MODEL_PATH} ]]; then
    echo "ERROR: MODEL_PATH was not set."
    echo "ERROR: MODEL_PATH must be set to either the HuggingFace ID or locally " \
         "downloaded path to the model weights. Since Deepseek R1 is large, it is " \
         "recommended to pre-download them to a shared location and provide the path."
    exit 1
fi

if [[ -z ${SERVED_MODEL_NAME} ]]; then
    echo "WARNING: SERVED_MODEL_NAME was not set. It will be derived from MODEL_PATH."
fi



#EXTRA_ARGS=""
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -n ${DISAGGREGATION_MODE} ]]; then
  EXTRA_ARGS+="--disaggregation-mode ${DISAGGREGATION_MODE} "
  # FIX: Map DISAGGREGATION_MODE to the required KV transfer roles
  if [[ "${DISAGGREGATION_MODE}" == "prefill" ]]; then
     EXTRA_ARGS+='--kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_producer"} '
  elif [[ "${DISAGGREGATION_MODE}" == "decode" ]]; then
     EXTRA_ARGS+='--kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_consumer"} '
  fi
fi

# Only publish KV events if using KV-aware routing (not needed for round-robin)
if [[ -n ${PUBLISH_KV_EVENTS} ]] && [[ ${PUBLISH_KV_EVENTS} == "true" ]]; then
  EXTRA_ARGS+="--publish-events-and-metrics "
fi

if [[ -n ${MODALITY} ]]; then
  EXTRA_ARGS+="--modality ${MODALITY} "
fi


#pip install nvidia-modelopt[hf]
export HOME="/tmp"
export XDG_CACHE_HOME="/tmp"
export PIP_CACHE_DIR="/tmp/pip_cache"
export FLASHINFER_WORKSPACE_DIR="/tmp/flashinfer_jit"
export FLASHINFER_CACHE_DIR="/tmp/flashinfer_cache"
export HF_HOME="/tmp/huggingface"
export TRITON_CACHE_DIR="/tmp/dynamo_triton"
#export TRTLLM_HANG_DETECTION_TIMEOUT=1200

export UCX_RCACHE_MAX_UNRELEASED=1024

# PYTORCH_CUDA_ALLOC_CONF is unset — the cumem allocator (--enable-cumem-allocator)
# manages expandable segments internally and is compatible with NixlConnector KV transfer.
# Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True externally would conflict.
#unset PYTORCH_CUDA_ALLOC_CONF

#export WORLD_SIZE=4
#export TRTLLM_MAX_WORKSPACE_SIZE=4294967296  # Set to 4GB

# Use HEAD_NODE_IP from the main script env if set, otherwise derive from head node hostname
if [[ -z "${HEAD_NODE_IP}" ]]; then
  HEAD_NODE_IP="10.140.0.16"
fi
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

#export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

#export VLLM_USE_V1=0
#export VLLM_USE_DEEP_GEMM=1		
#export VLLM_ALL2ALL_BACKEND="pplx"		

#trtllm-llmapi-launch \
#dynamo-vllm-launch \
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export EXTRA_ARGS="--enable-cumem-allocator ${EXTRA_ARGS}"
#unset PYTORCH_CUDA_ALLOC_CONF

# cumem allocator required for NixlConnector KV transfer.
# expandable_segments set below to reduce native allocator fragmentation.
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:16,max_split_size_mb:256"

#export EXTRA_ARGS="--gpu-memory-utilization 0.70 ${EXTRA_ARGS}"
#export VLLM_EXECUTOR_CLASS="gpu_executor"
#export VLLM_MULTIPROC_EXECUTOR=0


#export VLLM_DISABLE_TORCHAO=1
export VLLM_USE_V1=1
export EXTRA_ARGS="--quantization compressed-tensors --enable-cumem-allocator --enforce-eager --gpu-memory-utilization 0.90 --max-model-len 2048 --max-num-seqs 16 --max-num-batched-tokens 2048 ${EXTRA_ARGS}"

#export VLLM_DISABLE_TORCHAO=1

# FIX: Override Kimi-K3 model.py with the KimiDecoderLayer.load_weights fix
# for shared_experts gate_proj/up_proj fusion. Without this, AutoWeightsLoader
# recurses into KimiMoE -> KimiMLP and fails because KimiMLP has fused
# gate_up_proj but the checkpoint has separate gate_proj/up_proj.
# We use a container mount to overlay the patched file over the original
# since /usr/local/ inside the container is read-only.
FIXED_MODEL="/mnt/examples/basics/multinode/trtllm/kimi_k3_model_fixed.py"
if [[ ! -f "${FIXED_MODEL}" ]]; then
  echo "ERROR: ${FIXED_MODEL} not found. Please ensure the patched model file exists."
  echo "See the pre-built file at the same path in the repo."
  exit 1
fi

# Mount the fixed model.py over the original container path using bind mount.
# This is needed because:
#   1. /usr/local/ inside the container is part of the squashfs image (read-only)
#   2. The multiproc workers spawn their own subprocesses that don't inherit PYTHONPATH
#   3. A bind mount at the directory level makes the fix visible to ALL processes
TARGET_DIR="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia"
PATCH_DIR="/tmp/vllm_models_patch/kimi_k3/nvidia"
if [[ ! -f "${PATCH_DIR}/model.py" ]]; then
  mkdir -p "${PATCH_DIR}"
  cp "${TARGET_DIR}/__init__.py" "${PATCH_DIR}/__init__.py" 2>/dev/null || true
  cp "${FIXED_MODEL}" "${PATCH_DIR}/model.py"
fi
if ! mountpoint -q "${TARGET_DIR}" 2>/dev/null; then
  mount --bind "${PATCH_DIR}" "${TARGET_DIR}"
fi

python3 -m dynamo.vllm \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    ${EXTRA_ARGS}
