#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start NATS
nats-server -js -a 0.0.0.0 -sd /tmp/nats-jetstream -V &

export NCCL_DEBUG=INFO

# Start etcd
# start etcd with head node IP
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://$HEAD_NODE_IP:2379 --data-dir /tmp/etcd &

export ETCD_ENDPOINTS="$HEAD_NODE_IP:2379"
export NATS_SERVER="nats://$HEAD_NODE_IP:4222"

# Wait for NATS/etcd to startup
sleep 10 #3

# Start OpenAI Frontend which will dynamically discover workers when they startup
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
# NOTE: This is a blocking call.

export HOME="/tmp"
export XDG_CACHE_HOME="/tmp"
export PIP_CACHE_DIR="/tmp/pip_cache"
export FLASHINFER_WORKSPACE_DIR="/tmp/flashinfer_jit"
export FLASHINFER_CACHE_DIR="/tmp/flashinfer_cache"
export HF_HOME="/tmp/huggingface"
export TRITON_CACHE_DIR="/tmp/dynamo_triton"
#export TRTLLM_MAX_WORKSPACE_SIZE=4294967296  # Set to 4GB
#export TRTLLM_HANG_DETECTION_TIMEOUT=1200

#export TRTLLM_WORKSPACE_SIZE=2147483648
#export TRT_LLM_MAX_WORKSPACE_SIZE=2147483648
#export PYTORCH_ALLOC_CONF="expandable_segments:True"

#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1200000
export NCCL_COMM_ID=$HEAD_NODE_IP:12345

NUMEXPR_MAX_THREADS=64 PIP_CACHE_DIR="/tmp/pip_cache" FLASHINFER_WORKSPACE_DIR="/tmp/flashinfer_jit" FLASHINFER_CACHE_DIR="/tmp/flashinfer"  HF_HOME="/tmp/huggingface" TRITON_CACHE_DIR="/tmp/dynamo_triton" python3 -m dynamo.frontend
