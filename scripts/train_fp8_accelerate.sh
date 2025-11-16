#!/bin/bash
set -e

# FP8 Training with Transformer Engine using Accelerate CLI
# This is the recommended approach as it uses accelerate's native initialization

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Enable debug output to verify FP8 is working
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=1

# Force Accelerate to use Transformer Engine backend via environment variables
# These env vars are read by Accelerate during initialization
export ACCELERATE_FP8_BACKEND=TE  # THIS IS THE KEY!
export ACCELERATE_FP8_FORMAT=HYBRID
export ACCELERATE_FP8_AMAX_COMPUTE_ALGO=max
export ACCELERATE_FP8_AMAX_HISTORY_LEN=16

# GPU configuration
GPU_NUM=${GPU_NUM:-$(nvidia-smi -L | wc -l)}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Config file (use argument or default to Qwen)
CONFIG_FILE=${1:-/workspace/configs/qwen_fp8_accelerate_only.yaml}

echo "==================================="
echo "FP8 Training with Transformer Engine"
echo "Using: accelerate launch (recommended)"
echo "==================================="
echo "Using config: $CONFIG_FILE"
echo "GPU number: $GPU_NUM"
echo "World size: $WORLD_SIZE"

cd /workspace/LLaMA-Factory

# Check if transformer-engine is installed and working
echo "Checking Transformer Engine installation..."
python -c "import transformer_engine; print(f'Transformer Engine version: {transformer_engine.__version__}')" || {
    echo "WARNING: Transformer Engine not found or not working!"
    exit 1
}

# Run training using accelerate launch
echo "Starting FP8 training..."
accelerate launch \
    --mixed_precision=fp8 \
    --num_processes=$GPU_NUM \
    --num_machines=$WORLD_SIZE \
    --machine_rank=$RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    src/train.py $CONFIG_FILE

echo "FP8 training completed!"
