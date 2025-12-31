#!/bin/bash
# FP8 training with LLaMA-Factory (requires H100/H200/B200)
set -e

CONFIG=${1:-"/workspace/configs/qwen_7b_fp8_full_benchmark.yaml"}

export ACCELERATE_MIXED_PRECISION=fp8
export HF_HOME=/tmp/huggingface
mkdir -p $HF_HOME

accelerate launch \
  --config_file /workspace/configs/accelerate_fp8.yaml \
  --mixed_precision fp8 \
  $(which llamafactory-cli) train "$CONFIG" 2>&1 | tee /tmp/fp8.log
