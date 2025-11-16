# How to Apply the FP8 Fix

## The Bugs Found

1. **Bug 1**: `workflow.py` doesn't pass `model_args` to the trainer
2. **Bug 2**: `fp8_utils.py` only supports TorchAO backend, not Transformer Engine

## Apply the Fixes

### Inside your Docker container:

```bash
# Fix 1: Patch workflow.py to pass model_args
cd /workspace/LLaMA-Factory/src/llamafactory/train/sft
cp workflow.py workflow.py.backup
sed -i 's/trainer = CustomSeq2SeqTrainer(/trainer = CustomSeq2SeqTrainer(\n        model_args=model_args,/' workflow.py

# Verify the fix
grep -A 3 "trainer = CustomSeq2SeqTrainer" workflow.py
# Should show: model_args=model_args, as a new line

# Fix 2: Replace fp8_utils.py with fixed version
cd /workspace/LLaMA-Factory/src/llamafactory/train
cp fp8_utils.py fp8_utils.py.backup

# Download the fixed version from the repo (if mounted) or copy it
# If you have the repo mounted:
cp /workspace/configs/../fp8_utils_fixed.py fp8_utils.py

# OR manually copy from host:
# On host: docker cp fp8_utils_fixed.py CONTAINER_ID:/workspace/LLaMA-Factory/src/llamafactory/train/fp8_utils.py
```

## Test the Fix

```bash
cd /workspace/LLaMA-Factory

# Clean previous checkpoint
rm -rf /workspace/checkpoints/qwen-fp8-deepspeed

# Run FP8 training
bash /workspace/scripts/train_fp8.sh 2>&1 | tee /tmp/fp8_te.log

# Verify Transformer Engine is being used
grep -i "transformer engine\|backend.*te\|fp8.*status" /tmp/fp8_te.log
```

You should see:
- ✅ "Using Transformer Engine FP8 backend (optimal for Hopper GPUs)"
- ✅ "backend: FP8BackendType.TE" (not AO)

## Compare Performance

After training completes:

```bash
python /workspace/scripts/compare_performance.py
```

You should now see proper FP8 speedup (1.3-1.5x on H100).
