#!/bin/bash

# FP8 Training with LLaMA-Factory (User-Friendly Way)
# This uses LLaMA-Factory's YAML configs and CLI

set -e

# Get config file (default to Qwen 7B FP8 config)
CONFIG_FILE=${1:-"/workspace/configs/qwen_7b_fp8_full_benchmark.yaml"}

echo "=================================================="
echo "FP8 Training with LLaMA-Factory CLI"
echo "=================================================="
echo "LLaMA-Factory Config: $CONFIG_FILE"
echo "Accelerate Config: /workspace/configs/accelerate_fp8.yaml"
echo ""

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

echo "Starting training with LLaMA-Factory CLI..."
echo ""

# Use accelerate launch with LLaMA-Factory CLI
# This is the PROPER way for LLaMA-Factory users to use FP8!
accelerate launch \
  --config_file /workspace/configs/accelerate_fp8.yaml \
  $(which llamafactory-cli) train \
  $CONFIG_FILE 2>&1 | tee /tmp/fp8_llamafactory.log

echo ""
echo "Training complete! Logs saved to /tmp/fp8_llamafactory.log"
echo ""
