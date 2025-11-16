#!/bin/bash
set -e

# FP8 Training on Single GPU (no DDP overhead)

export CUDA_VISIBLE_DEVICES=0  # Single GPU only!

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Enable debug output
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=1

# Force Accelerate to use Transformer Engine backend
export ACCELERATE_FP8_BACKEND=TE
export ACCELERATE_FP8_FORMAT=HYBRID
export ACCELERATE_FP8_AMAX_COMPUTE_ALGO=max
export ACCELERATE_FP8_AMAX_HISTORY_LEN=16

# Config file
CONFIG_FILE=${1:-/workspace/configs/qwen_3b_fp8_single_gpu.yaml}

echo "==========================================="
echo "FP8 Training - Single GPU (Clean Benchmark)"
echo "==========================================="
echo "Using config: $CONFIG_FILE"

cd /workspace/LLaMA-Factory

# Check Transformer Engine
echo "Checking Transformer Engine..."
python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')" || {
    echo "WARNING: Transformer Engine not found!"
    exit 1
}

# Run on single GPU (no torchrun needed!)
echo "Starting FP8 training on single GPU..."
python src/train.py $CONFIG_FILE

echo "FP8 training completed!"
