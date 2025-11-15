#!/bin/bash
# BF16 Baseline Training Script
# This provides a baseline to compare against FP8 performance

set -e

echo "=========================================="
echo "BF16 Baseline Training"
echo "=========================================="

# Network configuration for multi-GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export TQDM_POSITION=-1

# NCCL settings (adjust interface name as needed)
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
# export GLOO_SOCKET_IFNAME=enp50s0  # Uncomment and adjust for your network interface
# export NCCL_SOCKET_IFNAME=enp50s0
export NCCL_P2P_LEVEL=NVL

# GPU configuration
GPU_NUM=${GPU_NUM:-$(nvidia-smi -L | wc -l)}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "GPU number: $GPU_NUM"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

cd /workspace/LLaMA-Factory

# Run training
echo "Starting BF16 training..."
torchrun --nproc_per_node=$GPU_NUM \
    --nnodes=$WORLD_SIZE \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    /workspace/configs/llama3_bf16_baseline_sft.yaml

echo "BF16 training completed!"
