# Bug Report: FP8 Training Fails with ValueError

## Describe the bug

FP8 training fails immediately with a `ValueError` when trying to use the current FP8 implementation. The error occurs during Accelerator initialization in the Trainer.

## Error Message

```
[INFO|2025-11-16 19:16:56] llamafactory.train.fp8_utils:143 >> Accelerate already configured via config/env - skipping FP8 environment setup
[INFO|2025-11-16 19:16:56] llamafactory.train.fp8_utils:143 >>   ACCELERATE_MIXED_PRECISION=bf16
[INFO|2025-11-16 19:16:56] llamafactory.train.sft.trainer:143 >> Injecting TERecipeKwargs into Accelerator (non-deprecated version)

ValueError: Passing in an FP8 configuration requires setting `mixed_precision='fp8'`.
```

**Full traceback:**
```
Traceback (most recent call last):
  File "/workspace/LLaMA-Factory/src/llamafactory/train/sft/trainer.py", line 79, in patched_accelerator_init
    return original_accelerator_init(self, *args, **accelerator_kwargs)
  File "/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py", line 511, in __init__
    raise ValueError("Passing in an FP8 configuration requires setting `mixed_precision='fp8'`.")
ValueError: Passing in an FP8 configuration requires setting `mixed_precision='fp8'`.
```

## Root Cause

The current implementation has a **fundamental conflict**:

1. **Accelerate is configured with `bf16`** (from command-line or environment)
2. **LLaMA-Factory then tries to inject `TERecipeKwargs`** via monkey-patching
3. **Accelerate rejects this** because `mixed_precision='bf16'` but FP8 config is being passed

The issue is in `src/llamafactory/train/sft/trainer.py`:

```python
# This monkey-patch approach doesn't work!
def patched_accelerator_init(self, *args, **accelerator_kwargs):
    """Inject TERecipeKwargs if missing."""
    if 'kwargs_handlers' not in accelerator_kwargs:
        fp8_recipe = TERecipeKwargs(...)  # ← Injecting FP8 config
        accelerator_kwargs['kwargs_handlers'] = [fp8_recipe]
    return original_accelerator_init(self, *args, **accelerator_kwargs)
    # ↑ But mixed_precision is still 'bf16'!

Accelerator.__init__ = patched_accelerator_init  # ← Monkey-patch
```

## To Reproduce

**Minimal reproduction:**

1. **Config file** (`configs/test_fp8.yaml`):
```yaml
model_name_or_path: Qwen/Qwen2.5-7B
stage: sft
do_train: true
finetuning_type: full
fp8: true
fp8_backend: te
dataset: alpaca_en_demo
template: qwen
cutoff_len: 512
max_samples: 10
per_device_train_batch_size: 1
bf16: true
output_dir: test_fp8_output
```

2. **Command:**
```bash
llamafactory-cli train configs/test_fp8.yaml
```

3. **Result:** Immediate crash with `ValueError`

## Expected behavior

FP8 training should work when `fp8: true` and `fp8_backend: te` are set in the config.

Expected to see:
- Model loads successfully
- Accelerator configured with FP8
- Training starts with FP8-enabled layers

## Environment

```
<PASTE llamafactory-cli env OUTPUT>
```

Example:
```
- Platform: Linux-5.15.0-1063-nvidia-x86_64-with-glibc2.35
- Python version: 3.12.0
- PyTorch version: 2.4.0+cu121
- Transformers version: 4.45.0
- Accelerate version: 1.1.1
- LLaMA-Factory version: 0.9.1.dev0
- GPU type: NVIDIA H100 80GB HBM3
- Transformer Engine version: 1.11.0
```

## Analysis

The current implementation attempts to **programmatically inject FP8 config** after Accelerate has already been initialized. This conflicts with Accelerate's requirements.

### Current Broken Flow:

```
1. TrainingArguments sets mixed_precision='bf16' (from bf16: true)
   ↓
2. Trainer creates Accelerator(mixed_precision='bf16')
   ↓
3. LLaMA-Factory monkey-patches to inject TERecipeKwargs
   ↓
4. Accelerator.__init__ receives:
   - mixed_precision='bf16'  ← Wrong!
   - kwargs_handlers=[TERecipeKwargs()]  ← FP8 config
   ↓
5. ❌ ValueError: "FP8 config requires mixed_precision='fp8'"
```

### Why This Happens:

1. **`bf16: true` in config** sets `TrainingArguments.bf16=True`
2. **Trainer translates this** to `Accelerator(mixed_precision='bf16')`
3. **Accelerate validates** that FP8 config requires `mixed_precision='fp8'`
4. **Conflict!** Can't have FP8 config with bf16 mixed precision

## Additional Context

### Why `bf16: true` is in FP8 configs

FP8 training **still uses BF16** for non-quantized operations (embeddings, layer norms, etc.). So configs correctly have:

```yaml
bf16: true    # Use BF16 for non-FP8 operations
fp8: true     # Use FP8 for quantizable layers
```

But LLaMA-Factory translates `bf16: true` → `mixed_precision='bf16'` which conflicts with FP8.

### Workarounds Attempted

I tried several approaches:
1. ❌ **Monkey-patch `Accelerator.__init__`** - Causes the error above
2. ❌ **Set env vars before Trainer** - Overridden by TrainingArguments
3. ❌ **Create Accelerator and pass to Trainer** - Trainer doesn't accept pre-created Accelerator
4. ❌ **Patch TrainingArguments** - Too invasive, breaks other features

None work reliably because of the bf16/fp8 mixed precision conflict.

## Proposed Fix

The **correct approach** is to use Accelerate's config file system:

### 1. Don't translate `bf16: true` to `mixed_precision='bf16'` when `fp8: true`

In `src/llamafactory/train/sft/workflow.py` or wherever TrainingArguments are created:

```python
# Detect FP8 mode
if model_args.fp8:
    # Don't set bf16 in TrainingArguments
    # Accelerate config will handle precision
    training_args.bf16 = False
    # Or better: Set via environment for Accelerate
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
```

### 2. Use `accelerate launch` with config file

This is the **official HuggingFace approach**:

```bash
accelerate launch --config_file configs/accelerate_fp8.yaml \
  llamafactory-cli train configs/my_model.yaml
```

Where `accelerate_fp8.yaml`:
```yaml
mixed_precision: fp8
fp8_config:
  backend: TE
  fp8_format: HYBRID
  amax_compute_algo: max
  amax_history_len: 1024
```

This completely avoids the monkey-patching and mixed precision conflicts.

## Related Issues

- HuggingFace Transformers #25333: "Support H100 training with FP8 in Trainer and Deepspeed"
- This is a known limitation when trying to use FP8 programmatically with Trainer

## References

- Accelerate FP8 docs: https://huggingface.co/docs/accelerate/usage_guides/low_precision_training
- Accelerate config guide: https://huggingface.co/docs/accelerate/usage_guides/explore
- My working implementation: https://github.com/sbhavani/LLaMA-Factory/tree/fix/accelerate-config-support

## Summary

**Current status:** FP8 training is broken due to mixed precision conflicts.

**Root cause:** Programmatic injection of FP8 config conflicts with TrainingArguments' bf16 setting.

**Recommended fix:** Support Accelerate config files (see follow-up feature request).

**Immediate workaround:** Use the `fix/accelerate-config-support` branch which properly handles this.

---

## Reproduction Checklist

- [x] Bug occurs with latest LLaMA-Factory version
- [x] Bug occurs on standard config (Qwen 7B, alpaca dataset)
- [x] Bug is related to FP8/mixed precision handling
- [x] Error message is clear: `ValueError: Passing in an FP8 configuration requires setting mixed_precision='fp8'`
- [x] Environment has all required dependencies (TE, Accelerate, etc.)

## Impact

**Severity:** High - FP8 training completely broken

**Affected users:** Anyone trying to use FP8 on H100/H200/B200 GPUs

**Workaround complexity:** Requires using a fork with fixes
