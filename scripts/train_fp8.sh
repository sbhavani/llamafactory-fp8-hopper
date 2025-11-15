#!/bin/bash
# FP8 Training Script with Transformer Engine for Hopper
# This script sets up the environment for optimal FP8 performance

set -e

echo "=========================================="
echo "FP8 Training with Transformer Engine"
echo "=========================================="

# Network configuration for multi-GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export TQDM_POSITION=-1

# NCCL settings (adjust interface name as needed)
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
# export GLOO_SOCKET_IFNAME=enp50s0  # Uncomment and adjust for your network interface
# export NCCL_SOCKET_IFNAME=enp50s0
export NCCL_P2P_LEVEL=NVL

# Critical: Enable communication-computation overlap for Hopper
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Transformer Engine FP8 settings
export NVTE_APPLY_QK_LAYER_SCALING=1
export NVTE_FP8_DPA_BWD=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NVTE_FP8_ALLREDUCE=1  # Hopper-specific optimization

# Enable debug output to verify FP8 is working
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=1

# Force Accelerate to use Transformer Engine backend
# This must be set BEFORE trainer initialization
export ACCELERATE_USE_FP8=1
export FP8_BACKEND=te

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

# Check if transformer-engine is installed and working
echo "Checking Transformer Engine installation..."
python -c "import transformer_engine; print(f'Transformer Engine version: {transformer_engine.__version__}')" || {
    echo "WARNING: Transformer Engine not found or not working!"
    exit 1
}

# Config file (use argument or default to Qwen)
CONFIG_FILE=${1:-/workspace/configs/qwen_fp8_deepspeed_sft.yaml}

echo "Using config: $CONFIG_FILE"

# Run training
echo "Starting FP8 training..."
torchrun --nproc_per_node=$GPU_NUM \
    --nnodes=$WORLD_SIZE \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    $CONFIG_FILE

echo "FP8 training completed!"
