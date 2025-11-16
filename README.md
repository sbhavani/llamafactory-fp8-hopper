# LLaMA-Factory FP8 Training Setup

This repository contains Docker-based setup for training LLMs with FP8 precision using NVIDIA Transformer Engine on Hopper (H100) or Blackwell (B200) GPUs.

## Quick Start

### 1. Build Docker Image

```bash
docker build -t llamafactory-fp8:latest .
```

### 2. Run Container

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/scripts:/workspace/scripts \
  -v /tmp:/tmp \
  -it llamafactory-fp8:latest bash
```

### 3. Train with FP8 (Proper Method)

Inside the container:

```bash
# FP8 training using Accelerate config (RECOMMENDED)
bash /workspace/scripts/train_fp8_proper.sh

# BF16 baseline for comparison
bash /workspace/scripts/train_bf16_proper.sh
```

## How It Works

### The Proper Way: Accelerate Config Files

The **official and correct way** to use FP8 with HuggingFace Trainer is via Accelerate config files:

1. **Create Accelerate config** (`configs/accelerate_fp8.yaml`):
   ```yaml
   mixed_precision: fp8
   fp8_config:
     backend: TE
     fp8_format: HYBRID
     amax_compute_algo: max
     amax_history_len: 1024
   ```

2. **Launch with `accelerate launch`**:
   ```bash
   accelerate launch --config_file configs/accelerate_fp8.yaml \
     /workspace/LLaMA-Factory/src/train.py \
     <training arguments>
   ```

3. **Trainer automatically uses FP8** - No code changes needed!

### Why This Works

- ✅ **`accelerate launch`** sets up the FP8 environment BEFORE running your script
- ✅ **Trainer detects Accelerate's FP8 config** automatically
- ✅ **Zero conflicts** - Official Accelerate workflow
- ✅ **Works with DeepSpeed/FSDP** - Fully compatible

### LLaMA-Factory Integration

Our fork of LLaMA-Factory ([sbhavani/LLaMA-Factory](https://github.com/sbhavani/LLaMA-Factory), branch `fix/accelerate-config-support`) includes a fix to:

- **Detect if Accelerate is already configured** via config file
- **Skip setting conflicting environment variables** when using `accelerate launch`
- **Maintain backward compatibility** with direct `python train.py` usage

See the key change in `src/llamafactory/train/fp8_utils.py`:

```python
def configure_fp8_environment(model_args):
    # Skip if Accelerate is already configured via config file
    if os.environ.get("ACCELERATE_MIXED_PRECISION"):
        logger.info("Accelerate already configured - skipping FP8 setup")
        return
    # ... otherwise set env vars
```

## Performance Benchmark

We verified that **Accelerate's FP8 integration has zero overhead** compared to pure Transformer Engine:

```bash
# Run official Accelerate benchmark
cd /workspace/accelerate-test/benchmarks/fp8/transformer_engine
python test_fp8_speed.py
```

**Results:**
- Baseline (Pure TE): 0.053s per step
- Accelerate: 0.053s per step
- ✅ **Equivalent performance!**

## Configuration Files

- `configs/accelerate_fp8.yaml` - Accelerate FP8 configuration
- `configs/accelerate_bf16.yaml` - Accelerate BF16 configuration (baseline)
- `configs/qwen_7b_fp8_full_benchmark.yaml` - LLaMA-Factory training config for FP8
- `configs/qwen_7b_bf16_full_benchmark.yaml` - LLaMA-Factory training config for BF16

## Scripts

- `scripts/train_fp8_proper.sh` - FP8 training using `accelerate launch`
- `scripts/train_bf16_proper.sh` - BF16 training using `accelerate launch`
- `scripts/compare_performance.py` - Compare FP8 vs BF16 performance

## Environment Variables

Key environment variables set in training scripts:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True  # Better CUDA memory management
export TOKENIZERS_PARALLELISM=false                 # Avoid tokenizer warnings
export HF_HOME=/tmp/huggingface                     # Cache models in /tmp
export NVTE_DEBUG=1                                 # Enable TE debug logging
```

## Expected Performance

FP8 training should provide:
- **1.3-1.5x speedup** on H100/H200 for models 7B+
- **20-30% memory reduction** compared to BF16
- **Equivalent accuracy** with proper hyperparameters

## Troubleshooting

### Issue: FP8 is slower than BF16
- **Cause**: Model too small (< 7B parameters)
- **Fix**: Use larger models (7B+) or disable FP8

### Issue: CUDA Out of Memory
- **Fix**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`

### Issue: "Accelerate already configured" warning
- **Expected**: When using `accelerate launch`, LLaMA-Factory detects the config and skips its own setup

## References

- [HuggingFace Accelerate FP8 Documentation](https://huggingface.co/docs/accelerate/usage_guides/low_precision_training)
- [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Our Fork](https://github.com/sbhavani/LLaMA-Factory/tree/fix/accelerate-config-support)

## Contributing

This setup was developed to enable proper FP8 training with LLaMA-Factory and HuggingFace Trainer. The key insight was using Accelerate's official config file approach instead of programmatic configuration.

To contribute upstream:
1. Test the Accelerate config approach with vanilla Transformers
2. Submit PR to HuggingFace Transformers if issues are found
3. Update LLaMA-Factory documentation with FP8 best practices
