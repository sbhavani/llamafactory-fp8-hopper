#!/bin/bash

# FP8 Training with LLaMA-Factory (User-Friendly Way)
# This uses LLaMA-Factory's YAML configs and CLI

set -e

# Get config file (default to Qwen 7B FP8 config)
CONFIG_FILE=${1:-"/workspace/configs/qwen_7b_fp8_full_benchmark.yaml"}

# Detect base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "/workspace/configs" ]]; then
    BASE_DIR="/workspace"
elif [[ -d "$SCRIPT_DIR/../configs" ]]; then
    BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    BASE_DIR="$HOME/llamafactory-fp8/LLaMA-Factory"
fi

echo "=================================================="
echo "FP8 Training with LLaMA-Factory CLI"
echo "=================================================="
echo "LLaMA-Factory Config: $CONFIG_FILE"
echo "Accelerate Config: $BASE_DIR/configs/accelerate_fp8.yaml"
echo "Base Directory: $BASE_DIR"
echo ""

# Apply FP8 trainer patch if needed
if [[ -f "$BASE_DIR/scripts/patch_trainer_fp8.py" ]]; then
    echo "Checking/applying FP8 trainer patch..."
    python "$BASE_DIR/scripts/patch_trainer_fp8.py" "$BASE_DIR/LLaMA-Factory" 2>/dev/null || \
    python "$BASE_DIR/scripts/patch_trainer_fp8.py" "/workspace/LLaMA-Factory" 2>/dev/null || \
    echo "Note: Could not auto-patch trainer (may already be patched or path differs)"
    echo ""
fi

# Force FP8 mixed precision (critical fix!)
export ACCELERATE_MIXED_PRECISION=fp8
export ACCELERATE_FP8_BACKEND=TE

# Set environment variables for optimal FP8 performance
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2

# Set HuggingFace cache to /tmp to avoid disk quota issues
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/huggingface/transformers
export HF_DATASETS_CACHE=/tmp/huggingface/datasets
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

echo "Environment:"
echo "  ACCELERATE_MIXED_PRECISION=$ACCELERATE_MIXED_PRECISION"
echo "  ACCELERATE_FP8_BACKEND=$ACCELERATE_FP8_BACKEND"
echo ""
echo "Starting training with LLaMA-Factory CLI..."
echo ""

# Use accelerate launch with LLaMA-Factory CLI
# --mixed_precision fp8 ensures FP8 is used even if config parsing fails
accelerate launch \
  --config_file "$BASE_DIR/configs/accelerate_fp8.yaml" \
  --mixed_precision fp8 \
  $(which llamafactory-cli) train \
  "$CONFIG_FILE" 2>&1 | tee /tmp/fp8_llamafactory.log

echo ""
echo "Training complete! Logs saved to /tmp/fp8_llamafactory.log"
echo ""
