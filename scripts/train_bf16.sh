#!/bin/bash
# BF16 training with LLaMA-Factory
set -e

CONFIG=${1:-"/workspace/configs/qwen_7b_bf16_full_benchmark.yaml"}

export HF_HOME=/tmp/huggingface
mkdir -p $HF_HOME

accelerate launch \
  --config_file /workspace/configs/accelerate_bf16.yaml \
  $(which llamafactory-cli) train "$CONFIG" 2>&1 | tee /tmp/bf16.log
