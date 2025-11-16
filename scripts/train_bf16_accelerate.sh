#!/bin/bash
set -e

# BF16 Baseline Training using Accelerate CLI (for fair comparison with FP8)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# GPU configuration
GPU_NUM=${GPU_NUM:-$(nvidia-smi -L | wc -l)}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Config file (use argument or default to Qwen)
CONFIG_FILE=${1:-/workspace/configs/qwen_bf16_baseline_no_deepspeed.yaml}

echo "==================================="
echo "BF16 Baseline Training"
echo "Using: accelerate launch (for fair comparison)"
echo "==================================="
echo "Using config: $CONFIG_FILE"
echo "GPU number: $GPU_NUM"
echo "World size: $WORLD_SIZE"

cd /workspace/LLaMA-Factory

# Run training using accelerate launch
echo "Starting BF16 training..."
accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=$GPU_NUM \
    --num_machines=$WORLD_SIZE \
    --machine_rank=$RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    src/train.py $CONFIG_FILE

echo "BF16 training completed!"
