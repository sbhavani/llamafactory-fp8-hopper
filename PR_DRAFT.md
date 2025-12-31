# PR: fix(fp8): Add Transformer Engine backend support for Hopper GPUs

## Summary

Add proper FP8 training support using Transformer Engine backend on NVIDIA Hopper/Blackwell GPUs (H100, H200, B200).

**Changes:**
- Pass `model_args` to `CustomSeq2SeqTrainer` for FP8 configuration
- Add `TERecipeKwargs` support when `fp8_backend: te` is set
- Patch Accelerator to inject FP8 kwargs and force `mixed_precision='fp8'`
- Add `te.fp8` stub for Accelerate 1.12+ compatibility
- Detect existing Accelerate config to avoid conflicts

## Motivation

The existing FP8 implementation only supports TorchAO backend. Transformer Engine provides better performance on Hopper GPUs but wasn't properly integrated. This PR enables users to use TE backend with:

```yaml
fp8: true
fp8_backend: te
```

## Test Plan

Tested on NVIDIA H100 80GB with Qwen2.5-7B:

| Config | Time/step | Speedup |
|--------|-----------|---------|
| BF16   | 1.02s     | 1.0x    |
| FP8    | 0.90s     | 1.13x   |

Test harness: https://github.com/sbhavani/llamafactory-fp8-hopper

```bash
# Reproduce
docker build -t llamafactory-fp8 .
docker run --gpus all --ipc=host -it llamafactory-fp8 bash
bash /workspace/scripts/train_bf16.sh
bash /workspace/scripts/train_fp8.sh
python /workspace/scripts/compare_performance.py
```

## Files Changed

- `src/llamafactory/train/fp8_utils.py` - Add TE backend support, simplify code
- `src/llamafactory/train/sft/trainer.py` - Add `_patch_accelerator_for_fp8()` method
- `src/llamafactory/train/sft/workflow.py` - Pass `model_args` to trainer (+1 line)

## Checklist

- [ ] Run `make style && make quality`
- [ ] Run `make test`
- [ ] Rebase on latest main before submitting

---

## Commands to submit PR

```bash
# Clone your fork
cd /tmp
git clone https://github.com/sbhavani/LLaMA-Factory.git
cd LLaMA-Factory
git checkout fix/accelerate-config-support

# Install dev dependencies
pip install -e ".[dev]"

# Run checks (required by contributing guide)
make style && make quality
make test

# Rebase on upstream
git remote add upstream https://github.com/hiyouga/LLaMA-Factory.git
git fetch upstream
git rebase upstream/main

# Push and create PR
git push -f origin fix/accelerate-config-support

# Create PR via GitHub UI or:
gh pr create \
  --repo hiyouga/LLaMA-Factory \
  --title "fix(fp8): Add Transformer Engine backend support for Hopper GPUs" \
  --body-file PR_BODY.md
```
