#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This is one of the only variables that must be set currently, most of the rest may
# just work out of the box if following the steps in the README
IMAGE="${IMAGE:-"/hpelustre/shruti/qwen-experiments/ai-dynamo-tensorrtllm-runtime-1.3.1.sqsh"}"

# Set to mount current host directory to /mnt inside the container as an example,
# but you may freely customize the mounts based on your cluster. A common practice
# is to mount paths to NFS storage for common scripts, model weights, etc.
# NOTE: This can be a comma separated list of multiple mounts as well.
DEFAULT_MOUNT="${PWD}/../../../..:/mnt,/hpelustre:/hpelustre"
MOUNTS="${MOUNTS:-${DEFAULT_MOUNT}}"

NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

NUM_PREFILL_NODES=${NUM_PREFILL_NODES:-1}
NUM_PREFILL_WORKERS=${NUM_PREFILL_WORKERS:-1}
PREFILL_ENGINE_CONFIG="${PREFILL_ENGINE_CONFIG:-/mnt/examples/basics/multinode/trtllm/prefill-kimik3.yaml}"

NUM_DECODE_NODES=${NUM_DECODE_NODES:-1}
NUM_DECODE_WORKERS=${NUM_DECODE_WORKERS:-1}
DECODE_ENGINE_CONFIG="${DECODE_ENGINE_CONFIG:-/mnt/examples/basics/multinode/trtllm/decode-kimik3.yaml}"

# Automate settings of certain variables for convenience, but you are free
# to manually set these for more control as well.
ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"
#export HEAD_NODE="${SLURMD_NODENAME}"
#export HEAD_NODE_IP="$(hostname -i)"
#export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
#export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

# Uses the primary network card interface IP instead of local loopback (127.0.0.1)
export HEAD_NODE="${SLURMD_NODENAME}"
export HEAD_NODE_IP="$(hostname -I | awk '{print $1}')"
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

export MODEL_PATH="/hpelustre/shruti/.cache/huggingface/hub/models--nvidia--GLM-5.1-NVFP4/snapshots/9116eb1c3ef24dd38dfe61e2eb71399ddd8070c8"
export SERVED_MODEL_NAME="nvidia/GLM-5.1-NVFP4"
#export MODEL_PATH="/hpelustre/shruti/.cache/huggingface/hub/models--RedHatAI--Kimi-K3-NVFP4/snapshots/71eed69448bc5ee7f83c4c08a61d6598a25b6ae4"
#export SERVED_MODEL_NAME="RedHatAI/Kimi-K3-NVFP4"

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
  /mnt/examples/basics/multinode/trtllm/start_frontend_services.sh &

sleep 5

# NOTE: Output streamed to stdout for ease of understanding the example, but
# in practice you would probably set `srun --output ... --error ...` to pipe
# the stdout/stderr to files.
for ((i=1; i<=${NUM_PREFILL_WORKERS}; i++)); do
  echo "Launching multi-node prefill worker in background."
  (
  unset SLURM_PROCID SLURM_NTASKS SLURM_LOCALID
  export DISAGGREGATION_MODE=prefill 
  export ENGINE_CONFIG=${PREFILL_ENGINE_CONFIG} 
  srun \
    --mpi pmix \
    --oversubscribe \
    --container-image "${IMAGE}" \
    --container-mounts "${MOUNTS}" \
    --container-env DISAGGREGATION_MODE,ENGINE_CONFIG \
    --verbose \
    --label \
    -A "${ACCOUNT}" \
    -J "${ACCOUNT}-dynamo.trtllm" \
    --nodes "${NUM_PREFILL_NODES}" \
    --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
    --jobid "${SLURM_JOB_ID}" \
    /mnt/examples/basics/multinode/trtllm/start_trtllm_worker.sh ) &
done


echo "Waiting for prefill cluster to stabilize..."
sleep 20


for ((i=1; i<=${NUM_DECODE_WORKERS}; i++)); do
  echo "Launching multi-node decode worker in background."
  (
  unset SLURM_PROCID SLURM_NTASKS SLURM_LOCALID
  export DISAGGREGATION_MODE=decode 
  export ENGINE_CONFIG=${DECODE_ENGINE_CONFIG} 
  srun \
    --mpi pmix \
    --oversubscribe \
    --container-image "${IMAGE}" \
    --container-mounts "${MOUNTS}" \
    --container-env DISAGGREGATION_MODE,ENGINE_CONFIG \
    --verbose \
    --label \
    -A "${ACCOUNT}" \
    -J "${ACCOUNT}-dynamo.trtllm" \
    --nodes "${NUM_DECODE_NODES}" \
    --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
    --jobid "${SLURM_JOB_ID}" \
    /mnt/examples/basics/multinode/trtllm/start_trtllm_worker.sh ) &
done
