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



if [[ -z ${ENGINE_CONFIG} ]]; then
    echo "ERROR: ENGINE_CONFIG was not set."
    echo "ERROR: ENGINE_CONFIG must be set to a valid Dynamo+TRTLLM engine config file."
    exit 1
fi

EXTRA_ARGS=""
if [[ -n ${DISAGGREGATION_MODE} ]]; then
  EXTRA_ARGS+="--disaggregation-mode ${DISAGGREGATION_MODE} "
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
export FLASHINFER_CACHE_DIR="/tmp/flashinfer"
export HF_HOME="/tmp/huggingface"
export TRITON_CACHE_DIR="/tmp/dynamo_triton"
export TRTLLM_HANG_DETECTION_TIMEOUT=1200

#export WORLD_SIZE=4
#export TRTLLM_MAX_WORKSPACE_SIZE=4294967296  # Set to 4GB

HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
#export ETCD_ENDPOINTS="$HEAD_NODE_IP:2379"
#export NATS_SERVER="nats://$HEAD_NODE_IP:4222"
export ETCD_ENDPOINTS="10.140.0.16:2379"
export NATS_SERVER="nats://10.140.0.16:4222"


trtllm-llmapi-launch \
  python3 -m dynamo.trtllm \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --extra-engine-args "${ENGINE_CONFIG}" \
    ${EXTRA_ARGS}
