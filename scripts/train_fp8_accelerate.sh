#!/bin/bash
set -e

# FP8 Training with Transformer Engine using Accelerate CLI
# This is cleaner than torchrun as it uses accelerate's native config

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Enable debug output to verify FP8 is working
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=1

# Force Accelerate to use Transformer Engine backend via environment variables
export ACCELERATE_FP8_BACKEND=TE  # THIS IS THE KEY!
export ACCELERATE_FP8_FORMAT=HYBRID
export ACCELERATE_FP8_AMAX_COMPUTE_ALGO=max
export ACCELERATE_FP8_AMAX_HISTORY_LEN=16

# Config file (use argument or default to Qwen)
CONFIG_FILE=${1:-/workspace/configs/qwen_fp8_accelerate_only.yaml}

echo "==================================="
echo "FP8 Training with Transformer Engine (using accelerate launch)"
echo "==================================="
echo "Using config: $CONFIG_FILE"

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
    --num_processes=2 \
    src/train.py $CONFIG_FILE
