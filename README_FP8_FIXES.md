# LLaMA-Factory FP8 Fixes

This repository contains fixes for two critical bugs in LLaMA-Factory's FP8 implementation that prevent proper Transformer Engine usage on Hopper GPUs.

## Bugs Fixed

### Bug 1: model_args Not Passed to Trainer
**File**: `src/llamafactory/train/sft/workflow.py`

**Problem**: The `CustomSeq2SeqTrainer` is instantiated without passing `model_args`, so FP8 configuration is never applied even though the code checks for it.

**Fix**: Add `model_args=model_args,` when creating the trainer.

**Patch**: `patches/001-fix-model-args-not-passed.patch`

### Bug 2: Only TorchAO Backend Supported
**File**: `src/llamafactory/train/fp8_utils.py`

**Problem**: The `create_fp8_kwargs()` function only uses `AORecipeKwargs` (TorchAO backend), which is 2x slower than Transformer Engine on Hopper GPUs. Even when `fp8_backend: te` is set, it still uses TorchAO.

**Fix**: Check the backend and use `FP8RecipeKwargs` with Transformer Engine recipe when `backend='te'`.

**Fixed File**: `fp8_utils_fixed.py`

## Impact

**Before fixes**:
- FP8 was never actually enabled (model_args not passed)
- Even when enabled, used slow TorchAO backend
- Result: FP8 was 2x **slower** than BF16

**After fixes**:
- FP8 properly configured with Transformer Engine
- Expected: 1.3-1.5x **faster** than BF16 on H100/GH200

## How It Works

The Dockerfile automatically:
1. Clones LLaMA-Factory
2. Applies the workflow.py patch
3. Replaces fp8_utils.py with the fixed version

Then training with `fp8: true` and `fp8_backend: te` will use proper Transformer Engine FP8.

## Testing

```bash
# Build container with fixes
docker build -t llamafactory-fp8:latest .

# Run container
docker run -it --rm --gpus all llamafactory-fp8:latest

# Inside container - verify FP8 is working
bash /workspace/scripts/train_fp8.sh 2>&1 | grep "Transformer Engine FP8"

# Should see:
# "Using Transformer Engine FP8 backend (optimal for Hopper GPUs)"
# "Accelerate FP8 status - enabled: True, backend: FP8BackendType.TE"
```

## Reporting to LLaMA-Factory

These bugs should be reported upstream:
- Issue: FP8 support is implemented but not working
- PR opportunity: Submit these fixes

## Files

- `patches/001-fix-model-args-not-passed.patch` - Fixes workflow.py
- `fp8_utils_fixed.py` - Fixed version with TE support
- `Dockerfile` - Updated to apply fixes automatically
