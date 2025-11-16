# FP8 Training Guide for LLaMA-Factory Users

This guide shows how **existing LLaMA-Factory users** can enable FP8 training with minimal changes to their workflow.

## TL;DR - What Changed?

**Before (BF16 training):**
```bash
llamafactory-cli train configs/my_model.yaml
```

**After (FP8 training):**
```bash
accelerate launch --config_file configs/accelerate_fp8.yaml \
  llamafactory-cli train configs/my_model.yaml
```

That's it! Your existing LLaMA-Factory configs work as-is. 🎉

## Detailed Setup

### 1. Your Existing LLaMA-Factory Config

You don't need to change your existing YAML configs! For example, your `configs/qwen_7b_sft.yaml`:

```yaml
# Your existing LLaMA-Factory config - NO CHANGES NEEDED
model_name_or_path: Qwen/Qwen2.5-7B
stage: sft
do_train: true
finetuning_type: full
dataset: alpaca_en_demo
template: qwen
cutoff_len: 1024
learning_rate: 5.0e-05
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
bf16: true
output_dir: saves/qwen-7b-sft
```

**Important:** Keep `bf16: true` even for FP8 training! FP8 uses BF16 for non-quantized operations.

### 2. Add Accelerate Config (One-Time Setup)

Create `configs/accelerate_fp8.yaml` (this repo already has it):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: fp8
fp8_config:
  backend: TE
  fp8_format: HYBRID
  amax_compute_algo: max
  amax_history_len: 1024
num_processes: 1
```

### 3. Enable FP8 in LLaMA-Factory Config

Add ONE line to your LLaMA-Factory config:

```yaml
# ... your existing config ...
bf16: true
fp8: true          # <-- ADD THIS LINE
fp8_backend: te    # <-- OPTIONAL: specify Transformer Engine
output_dir: saves/qwen-7b-fp8-sft
```

### 4. Launch Training

**Option A: Simple wrapper script (Recommended)**

```bash
# FP8 training
bash scripts/train_fp8_llamafactory.sh configs/qwen_7b_sft.yaml

# BF16 baseline
bash scripts/train_bf16_llamafactory.sh configs/qwen_7b_sft.yaml
```

**Option B: Direct command**

```bash
# FP8 training
accelerate launch \
  --config_file configs/accelerate_fp8.yaml \
  llamafactory-cli train \
  configs/qwen_7b_sft.yaml

# BF16 baseline
accelerate launch \
  --config_file configs/accelerate_bf16.yaml \
  llamafactory-cli train \
  configs/qwen_7b_sft.yaml
```

**Option C: Using accelerate default config**

One-time setup:
```bash
accelerate config  # Answer prompts, select FP8 when asked about mixed precision
```

Then just:
```bash
accelerate launch llamafactory-cli train configs/qwen_7b_sft.yaml
```

## Complete Example

### 1. Create Your Training Config

`configs/my_qwen_fp8.yaml`:
```yaml
### Model
model_name_or_path: Qwen/Qwen2.5-7B

### Method
stage: sft
do_train: true
finetuning_type: full
fp8: true              # Enable FP8
fp8_backend: te        # Use Transformer Engine

### Dataset
dataset: alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### Output
output_dir: saves/qwen-7b-fp8
logging_steps: 5
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### Training Hyperparameters
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-05
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

### 2. Train with FP8

```bash
# Method 1: Using wrapper script
bash scripts/train_fp8_llamafactory.sh configs/my_qwen_fp8.yaml

# Method 2: Direct command
accelerate launch \
  --config_file configs/accelerate_fp8.yaml \
  llamafactory-cli train \
  configs/my_qwen_fp8.yaml
```

### 3. Verify FP8 is Working

Check the logs for:

✅ **Accelerate config detected:**
```
Accelerate already configured via config/env - skipping FP8 environment setup
```

✅ **FP8 training enabled:**
```
FP8 training enabled with te backend.
```

✅ **TE layers created:**
```
Using Transformer Engine FP8 backend
```

✅ **Training progress:**
```
{'loss': 2.345, 'learning_rate': 5e-05, 'epoch': 0.1}
{'loss': 2.123, 'learning_rate': 4.8e-05, 'epoch': 0.2}
```

## Migration Guide: Existing Users

If you're already using LLaMA-Factory, here's what you need to change:

### Before (Your Current Workflow)

```bash
# 1. Create config
cat > configs/my_model.yaml << EOF
model_name_or_path: Qwen/Qwen2.5-7B
stage: sft
do_train: true
finetuning_type: full
dataset: alpaca_en_demo
bf16: true
output_dir: saves/my-model
EOF

# 2. Train
llamafactory-cli train configs/my_model.yaml
```

### After (FP8-Enabled Workflow)

```bash
# 1. Add FP8 to your existing config
cat >> configs/my_model.yaml << EOF
fp8: true
fp8_backend: te
EOF

# 2. Train with accelerate launch (wrap your existing command)
accelerate launch \
  --config_file configs/accelerate_fp8.yaml \
  llamafactory-cli train configs/my_model.yaml
```

**That's it!** Two small changes:
1. Add `fp8: true` to your YAML config
2. Wrap your command with `accelerate launch --config_file ...`

## How It Works Behind the Scenes

```
┌─────────────────────────────────────────────────────────┐
│  User runs:                                             │
│  accelerate launch --config_file accelerate_fp8.yaml \  │
│    llamafactory-cli train configs/my_model.yaml        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Accelerate:                                            │
│  - Reads accelerate_fp8.yaml                            │
│  - Sets ACCELERATE_MIXED_PRECISION=fp8                  │
│  - Configures TE backend                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  LLaMA-Factory:                                         │
│  - Reads configs/my_model.yaml                          │
│  - Detects Accelerate config (via env var)              │
│  - Skips its own FP8 setup (no conflict!)               │
│  - Passes control to Trainer                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Trainer:                                               │
│  - Creates Accelerator (detects FP8 from env)           │
│  - prepare() converts nn.Linear → te.Linear             │
│  - Training uses FP8 automatically!                     │
└─────────────────────────────────────────────────────────┘
```

## FAQ

### Q: Do I need to change my existing LLaMA-Factory configs?

**A:** Minimal changes:
1. Add `fp8: true` to enable FP8
2. Add `fp8_backend: te` (optional, to specify Transformer Engine)
3. Keep everything else the same (including `bf16: true`)

### Q: What if I don't want to use `accelerate launch`?

**A:** You can use `llamafactory-cli` directly IF you set up Accelerate config first:

```bash
# One-time setup
accelerate config  # Select FP8 when prompted

# Then use LLaMA-Factory normally
llamafactory-cli train configs/my_model.yaml
```

But `accelerate launch --config_file` is more explicit and reproducible.

### Q: Does this work with DeepSpeed?

**A:** Yes! Just add DeepSpeed config to your Accelerate config:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: fp8
fp8_config:
  backend: TE
  fp8_format: HYBRID
deepspeed_config:
  deepspeed_config_file: configs/ds_z3_config.json
```

Then use the same workflow.

### Q: Does this work with LoRA/QLoRA?

**A:** Yes! Just set `finetuning_type: lora` in your LLaMA-Factory config:

```yaml
finetuning_type: lora
fp8: true
fp8_backend: te
lora_rank: 8
lora_alpha: 16
lora_target: all
```

### Q: Can I use this in production?

**A:** Yes! This uses:
- ✅ Official Accelerate workflow (documented)
- ✅ Stable Transformer Engine backend
- ✅ Standard LLaMA-Factory configs
- ✅ No monkey-patches or hacks

### Q: What if I want to go back to BF16?

**A:** Just remove `fp8: true` from your config and use `accelerate_bf16.yaml`:

```bash
accelerate launch \
  --config_file configs/accelerate_bf16.yaml \
  llamafactory-cli train configs/my_model.yaml
```

## Performance Expectations

For **7B+ parameter models** on H100/B200:

| Metric | BF16 | FP8 | Improvement |
|--------|------|-----|-------------|
| Training Speed | 1.00x | 1.3-1.5x | 30-50% faster |
| Memory Usage | 100% | 70-80% | 20-30% reduction |
| Model Accuracy | Baseline | Equivalent | No degradation |

For **smaller models (< 7B)**:
- FP8 overhead may outweigh benefits
- Stick with BF16 for < 7B models

## Troubleshooting

### Issue: "fp8: true not recognized"

**Solution:** Make sure you're using the updated LLaMA-Factory fork:

```bash
# Check if fp8 argument exists
grep "fp8" /workspace/LLaMA-Factory/src/llamafactory/hparams/model_args.py

# If not found, rebuild Docker image
docker build -t llamafactory-fp8:latest .
```

### Issue: FP8 slower than BF16

**Check model size:**
```bash
# Count parameters
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B')
print(f'Parameters: {config.num_hidden_layers * config.hidden_size * 4 / 1e9:.1f}B')
"
```

If < 7B, use BF16 instead.

### Issue: "Accelerate already configured" but no speedup

**Check if TE layers are created:**
```bash
grep -i "transformer.*engine\|te.*linear" /tmp/fp8_llamafactory.log
```

If no TE layers found, check:
1. Is `fp8_backend: te` in your LLaMA-Factory config?
2. Is `backend: TE` in your Accelerate config?
3. Is Transformer Engine installed? (`pip show transformer-engine`)

## Summary

For LLaMA-Factory users, enabling FP8 is simple:

1. ✅ Add `fp8: true` to your existing YAML config
2. ✅ Wrap your command with `accelerate launch --config_file ...`
3. ✅ Everything else works the same

No need to:
- ❌ Learn Accelerate internals
- ❌ Rewrite configs
- ❌ Change training code
- ❌ Modify LLaMA-Factory

Your existing workflow + two small changes = FP8 speedup! 🚀
