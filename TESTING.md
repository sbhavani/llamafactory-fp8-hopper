# Testing FP8 Training with LLaMA-Factory

This guide walks you through testing the proper FP8 setup with Accelerate config files.

## Prerequisites

1. **Rebuild Docker image** (to get the updated LLaMA-Factory fork):
   ```bash
   docker build -t llamafactory-fp8:latest .
   ```

2. **Start container** with proper mounts:
   ```bash
   docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
     -v $(pwd)/checkpoints:/workspace/checkpoints \
     -v $(pwd)/configs:/workspace/configs \
     -v $(pwd)/scripts:/workspace/scripts \
     -v /tmp:/tmp \
     -it llamafactory-fp8:latest bash
   ```

## Test 1: Verify Accelerate Config Detection

This test confirms LLaMA-Factory correctly detects Accelerate config.

```bash
# Inside container
export ACCELERATE_MIXED_PRECISION=fp8

# Run a quick training test
cd /workspace/LLaMA-Factory
python src/train.py \
  --stage sft \
  --model_name_or_path Qwen/Qwen2.5-1.5B \
  --finetuning_type full \
  --dataset_dir data \
  --dataset alpaca_en_demo \
  --template qwen \
  --cutoff_len 512 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --max_samples 10 \
  --per_device_train_batch_size 1 \
  --bf16 true \
  --fp8 true \
  --output_dir /tmp/test-fp8-detection \
  --overwrite_output_dir true 2>&1 | tee /tmp/test-detection.log

# Check if LLaMA-Factory detected Accelerate config
grep -i "accelerate already configured" /tmp/test-detection.log
```

**Expected output:**
```
✅ Accelerate already configured via config/env - skipping FP8 environment setup
  ACCELERATE_MIXED_PRECISION=fp8
```

## Test 2: Run FP8 Training with Accelerate Config

Run a small benchmark to verify FP8 works end-to-end.

```bash
# Inside container
bash /workspace/scripts/train_fp8_proper.sh
```

**What to check in logs** (`/tmp/fp8_proper.log`):

1. **Accelerate config detected:**
   ```
   ✅ Accelerate already configured - skipping FP8 environment setup
   ```

2. **TE FP8 layers created:**
   ```
   ✅ Using Transformer Engine FP8 backend
   ```

3. **Training progresses without errors:**
   ```
   [INFO] Step 5: loss=2.345
   [INFO] Step 10: loss=2.123
   ```

## Test 3: Run BF16 Baseline

Run the same training with BF16 for comparison.

```bash
# Inside container
bash /workspace/scripts/train_bf16_proper.sh
```

## Test 4: Compare Performance

```bash
# Inside container
python /workspace/scripts/compare_performance.py
```

**Expected output for 7B model:**
```
FP8:  1.05s/step, 37.5 samples/s
BF16: 1.42s/step, 27.8 samples/s
Speedup: 1.35x (35% faster)
✅ FP8 is faster than BF16!
```

**Note:** For smaller models (< 7B), FP8 may be slower. This is expected behavior.

## Test 5: Verify TE Layers with Python

Check that Transformer Engine layers are actually being used:

```bash
# Inside container
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs
import os

# Set up FP8 via Accelerate
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
kwargs_handlers = [TERecipeKwargs(fp8_format="HYBRID", amax_history_len=32)]

accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)

# Load a small model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Prepare model
model = accelerator.prepare(model)

# Check for TE layers
te_layer_count = 0
for name, module in model.named_modules():
    if "transformer_engine" in str(type(module).__module__):
        te_layer_count += 1
        print(f"✅ Found TE layer: {name} ({type(module).__name__})")
        if te_layer_count >= 5:  # Just show first 5
            break

if te_layer_count > 0:
    print(f"\n✅ SUCCESS: Found {te_layer_count}+ Transformer Engine FP8 layers!")
else:
    print("\n❌ FAILED: No TE layers found")
EOF
```

**Expected output:**
```
✅ Found TE layer: model.layers.0.self_attn.q_proj (Linear)
✅ Found TE layer: model.layers.0.self_attn.k_proj (Linear)
✅ Found TE layer: model.layers.0.self_attn.v_proj (Linear)
✅ Found TE layer: model.layers.0.self_attn.o_proj (Linear)
✅ Found TE layer: model.layers.0.mlp.gate_proj (Linear)

✅ SUCCESS: Found 5+ Transformer Engine FP8 layers!
```

## Test 6: Run Official Accelerate Benchmark

Verify Accelerate's FP8 integration has zero overhead:

```bash
# Inside container
cd /workspace
git clone -b test/fp8-performance-benchmark https://github.com/sbhavani/accelerate.git accelerate-bench
cd accelerate-bench/benchmarks/fp8/transformer_engine
pip install evaluate
python test_fp8_speed.py
```

**Expected output:**
```
BASELINE: Pure TE with manual fp8_autocast
Steps: 229, Time per step: 0.053s

ACCELERATE: Using Accelerate's FP8 integration
Steps: 229, Time per step: 0.053s

✅ Accelerate performance is equivalent to pure TE!
```

## Troubleshooting

### Issue: "Accelerate already configured" but no TE layers

**Diagnosis:**
```bash
grep -i "te.*linear\|transformer.engine" /tmp/fp8_proper.log
```

If no TE layers found, check:
1. Is `ACCELERATE_FP8_BACKEND=TE` set in the script?
2. Is Transformer Engine installed? (`pip show transformer-engine`)
3. Are you using the right Accelerate config (`backend: TE`)?

### Issue: FP8 is slower than BF16

**Check model size:**
```bash
# Count parameters
python3 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B')
print(f'Parameters: {config.num_hidden_layers * config.hidden_size * 4 / 1e9:.1f}B')
"
```

If < 7B, FP8 overhead outweighs benefits. Use 7B+ models.

### Issue: CUDA Out of Memory

**Solutions:**
```bash
# Option 1: Reduce batch size
--per_device_train_batch_size 2  # was 4

# Option 2: Increase gradient accumulation
--gradient_accumulation_steps 8  # was 4

# Option 3: Enable gradient checkpointing
--gradient_checkpointing true

# Option 4: Use smaller sequence length
--cutoff_len 512  # was 1024
```

### Issue: Config file not found

**Check paths:**
```bash
ls -la /workspace/configs/accelerate_fp8.yaml
cat /workspace/configs/accelerate_fp8.yaml
```

If missing, the configs should be mounted as a volume. Check your `docker run` command includes:
```bash
-v $(pwd)/configs:/workspace/configs
```

## Success Criteria

✅ **Test passed if:**
1. LLaMA-Factory detects Accelerate config and skips its own setup
2. Training completes without errors
3. TE FP8 layers are created (verified in logs or Python test)
4. For 7B+ models: FP8 is 1.3-1.5x faster than BF16
5. Accelerate benchmark shows equivalent performance

❌ **Test failed if:**
1. No "Accelerate already configured" message in logs
2. No TE layers created
3. CUDA OOM errors (reduce batch size)
4. FP8 slower than BF16 for 7B+ model (investigate TE layer conversion)

## Next Steps

Once tests pass:

1. **Scale up to larger models** (7B → 14B → 72B)
2. **Test on B200 GPU** with larger batch sizes
3. **Run full fine-tuning** on production datasets
4. **Monitor memory usage** (`nvidia-smi dmon -s u`)
5. **Compare accuracy** between FP8 and BF16

## Getting Help

If tests fail:

1. Check logs: `/tmp/fp8_proper.log`
2. Enable debug logging: `export NVTE_DEBUG=1`
3. Review `HOW_IT_WORKS.md` for architecture details
4. Open issue on GitHub with logs

## Summary

The proper FP8 setup with Accelerate config files should:
- ✅ Work with zero code changes to LLaMA-Factory
- ✅ Automatically convert nn.Linear → te.Linear
- ✅ Provide 1.3-1.5x speedup for 7B+ models
- ✅ Have zero overhead compared to pure Transformer Engine

If your tests show this, **FP8 is working correctly!** 🎉
