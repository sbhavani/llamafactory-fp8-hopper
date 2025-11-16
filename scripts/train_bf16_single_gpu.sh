#!/bin/bash
set -e

# BF16 Training on Single GPU

export CUDA_VISIBLE_DEVICES=0  # Single GPU only!

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Config file
CONFIG_FILE=${1:-/workspace/configs/qwen_3b_bf16_single_gpu.yaml}

echo "==========================================="
echo "BF16 Training - Single GPU (Baseline)"
echo "==========================================="
echo "Using config: $CONFIG_FILE"

cd /workspace/LLaMA-Factory

# Run on single GPU
echo "Starting BF16 training on single GPU..."
python src/train.py $CONFIG_FILE

echo "BF16 training completed!"
