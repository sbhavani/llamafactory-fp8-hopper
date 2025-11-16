#!/bin/bash

# Proper BF16 Training Script using Accelerate Config
# This is the OFFICIAL way to use mixed precision with HuggingFace Trainer

set -e

# Get config file (default to Qwen 7B benchmark)
CONFIG_FILE=${1:-"/workspace/configs/qwen_7b_bf16_full_benchmark.yaml"}

echo "=================================================="
echo "BF16 Training with Accelerate Config (PROPER WAY)"
echo "=================================================="
echo "Config: $CONFIG_FILE"
echo "Accelerate Config: /workspace/configs/accelerate_bf16.yaml"
echo ""

# Set environment variables for optimal performance
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Set HuggingFace cache to /tmp to avoid disk quota issues
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/huggingface/transformers
export HF_DATASETS_CACHE=/tmp/huggingface/datasets

# Create cache directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

echo "Starting training with accelerate launch..."
echo ""

# Use accelerate launch with explicit config
# This is the OFFICIAL way to use mixed precision with Trainer!
accelerate launch \
  --config_file /workspace/configs/accelerate_bf16.yaml \
  /workspace/LLaMA-Factory/src/train.py \
  --stage sft \
  --do_train true \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --preprocessing_num_workers 16 \
  --finetuning_type full \
  --template qwen \
  --flash_attn auto \
  --dataset_dir /workspace/LLaMA-Factory/data \
  --dataset alpaca_en_demo \
  --cutoff_len 1024 \
  --learning_rate 5e-05 \
  --num_train_epochs 1 \
  --max_samples 1000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --logging_steps 5 \
  --save_steps 1000 \
  --output_dir /workspace/checkpoints/qwen-7b-bf16-proper \
  --overwrite_output_dir true \
  --ddp_timeout 180000000 \
  --plot_loss true 2>&1 | tee /tmp/bf16_proper.log

echo ""
echo "Training complete! Logs saved to /tmp/bf16_proper.log"
echo ""
