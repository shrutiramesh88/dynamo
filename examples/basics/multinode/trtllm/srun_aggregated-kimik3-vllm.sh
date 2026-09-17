#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This is one of the only variables that must be set currently, most of the rest may
# just work out of the box if following the steps in the README.
IMAGE="${IMAGE:-"/hpelustre/shruti/qwen-experiments/ai-dynamo-vllm-runtime-1.4.0-kimi-k3.sqsh"}"

# Set to mount current host directory to /mnt inside the container as an example,
# but you may freely customize the mounts based on your cluster. A common practice
# is to mount paths to NFS storage for common scripts, model weights, etc.
# NOTE: This can be a comma separated list of multiple mounts as well.
DEFAULT_MOUNT="${PWD}/../../../..:/mnt,/hpelustre:/hpelustre"
MOUNTS="${MOUNTS:-${DEFAULT_MOUNT}}"

# Example values, assuming 4 nodes with 4 GPUs on each node, such as 4xGB200 nodes.
# For 8xH100 nodes as an example, you may set this to 2 nodes x 8 gpus/node instead.
NUM_NODES=${NUM_NODES:-2}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

#export ENGINE_CONFIG="${ENGINE_CONFIG:-/mnt/examples/basics/multinode/trtllm/aggregated-llama3.yaml}"

# Automate settings of certain variables for convenience, but you are free
# to manually set these for more control as well.
ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"
export HEAD_NODE="${SLURMD_NODENAME}"
export HEAD_NODE_IP="$(hostname -I | awk '{print $1}')"
#export HEAD_NODE_IP="$(hostname -i)"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

if [[ -z ${IMAGE} ]]; then
  echo "ERROR: You need to set the IMAGE environment variable to the " \
       "Dynamo+TRTLLM docker image or .sqsh file from 'enroot import' " \
       "See how to build one from source here: " \
       "https://github.com/ai-dynamo/dynamo/tree/main/docs/backends/trtllm/README.md#build-container"
  exit 1
fi

export XDG_CACHE_HOME="/tmp"
export PIP_CACHE_DIR="/tmp/pip_cache"
export FLASHINFER_WORKSPACE_DIR="/tmp/flashinfer_jit"
export FLASHINFER_CACHE_DIR="/tmp/flashinfer"
export HF_HOME="/tmp/huggingface"
export TRITON_CACHE_DIR="/tmp/dynamo_triton"
export HF_HUB_OFFLINE=1

mkdir -p /tmp/flashinfer_jit /tmp/flashinfer_cache /tmp/huggingface /tmp/dynamo_triton /tmp/pip_cache

#export MODEL_PATH="/hpelustre/shruti/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
#export SERVED_MODEL_NAME="Qwen/Qwen3-0.6B"
export MODEL_PATH="/hpelustre/shruti/.cache/huggingface/hub/models--RedHatAI--Kimi-K3-NVFP4/snapshots/71eed69448bc5ee7f83c4c08a61d6598a25b6ae4"
export SERVED_MODEL_NAME="RedHatAI/Kimi-K3-NVFP4"

# NOTE: Output streamed to stdout for ease of understanding the example, but
# in practice you would probably set `srun --output ... --error ...` to pipe
# the stdout/stderr to files.
echo "Launching frontend services in background."
srun \
  --mpi pmix \
  --overlap \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "${ACCOUNT}-dynamo.trtllm" \
  --nodelist "${HEAD_NODE}" \
  --nodes 1 \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/examples/basics/multinode/trtllm/start_frontend_services_vllm.sh &

# NOTE: Output streamed to stdout for ease of understanding the example, but
# in practice you would probably set `srun --output ... --error ...` to pipe
# the stdout/stderr to files.
echo "Launching multi-node worker in background."
DISAGGREGATION_MODE="agg" \
srun \
  --mpi pmix \
  --oversubscribe \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-writable \
  --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE,DISAGGREGATION_MODE,MODEL_PATH,SERVED_MODEL_NAME,PYTORCH_CUDA_ALLOC_CONF \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "${ACCOUNT}-dynamo.trtllm" \
  --nodes "${NUM_NODES}" \
  --ntasks-per-node 1 \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/examples/basics/multinode/trtllm/start_worker_vllm_agg.sh &
