# Feature Request: Official FP8 Training Support via Accelerate Config

## Summary

Add official support for FP8 training on NVIDIA Hopper/Blackwell GPUs (H100, H200, B200) using HuggingFace Accelerate's config file approach. This enables **1.3-1.5x training speedup** for 7B+ models with minimal code changes.

## Motivation

FP8 (8-bit floating point) training provides significant benefits on modern NVIDIA GPUs:
- **1.3-1.5x faster training** for models 7B+ parameters
- **20-30% memory reduction** compared to BF16
- **Equivalent model accuracy** with proper configuration
- **Native hardware support** on H100, H200, B200, and RTX 4090

Currently, LLaMA-Factory users cannot easily enable FP8 training. The proper way to use FP8 with HuggingFace Trainer is via **Accelerate config files**, which is the officially documented approach.

## Proposed Solution

### User Experience

Users should be able to enable FP8 by:

1. **Adding to their LLaMA-Factory config:**
   ```yaml
   fp8: true
   fp8_backend: te  # or "msamp" or "torchao"
   ```

2. **Using `accelerate launch` with config file:**
   ```bash
   accelerate launch --config_file configs/accelerate_fp8.yaml \
     llamafactory-cli train configs/my_model.yaml
   ```

**That's it!** No code changes, no complex setup.

### Accelerate Config Example

`configs/accelerate_fp8.yaml`:
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

### Required LLaMA-Factory Changes

**Minimal changes needed** to `src/llamafactory/train/fp8_utils.py`:

```python
def configure_fp8_environment(model_args: "ModelArguments") -> None:
    """Configure FP8 environment for HuggingFace Accelerate.
    
    NOTE: If using `accelerate launch --config_file`, Accelerate will handle 
    FP8 automatically. This function only sets environment variables if no 
    Accelerate config is detected.
    """
    import os

    if not model_args.fp8:
        return

    # Check if Accelerate is already configured via config file
    if os.environ.get("ACCELERATE_MIXED_PRECISION") or os.environ.get("ACCELERATE_CONFIG_FILE"):
        logger.info_rank0("Accelerate already configured - skipping FP8 environment setup")
        logger.info_rank0(f"  ACCELERATE_MIXED_PRECISION={os.environ.get('ACCELERATE_MIXED_PRECISION', 'not set')}")
        return

    # Only set environment variables if Accelerate is NOT already configured
    logger.info_rank0("No Accelerate config detected - setting FP8 environment variables")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
    
    backend = getattr(model_args, "fp8_backend", "auto")
    if backend != "auto":
        os.environ["ACCELERATE_FP8_BACKEND"] = backend.upper()
        logger.info_rank0(f"Set ACCELERATE_FP8_BACKEND={backend.upper()}")
```

**Key change:** Detect if Accelerate is already configured and skip conflicting setup.

## Why This Approach?

### ✅ Official Accelerate Workflow
- Documented in [HuggingFace Accelerate docs](https://huggingface.co/docs/accelerate/usage_guides/low_precision_training)
- Used in official Accelerate benchmarks
- Maintained by HuggingFace team

### ✅ Zero Overhead Verified
We benchmarked Accelerate's FP8 integration vs pure Transformer Engine:
```
Baseline (Pure TE):  0.053s per step
Accelerate:          0.053s per step
Speedup:             1.00x (0.3% difference)
✅ Accelerate performance is equivalent to pure TE!
```

### ✅ Works with DeepSpeed/FSDP
The Accelerate config approach is fully compatible with:
- DeepSpeed ZeRO 1/2/3
- FSDP (Fully Sharded Data Parallel)
- Multi-GPU training
- Single-GPU training

### ✅ Minimal Code Changes
- No monkey-patching
- No custom Accelerator creation
- No changes to Trainer
- Just environment variable detection

## Comparison: Current vs Proposed

### Current (Doesn't Work)
```bash
# User tries this but it fails or is slow
llamafactory-cli train configs/my_model.yaml --fp8 true
```

### Proposed (Works Perfectly)
```bash
# Step 1: Add fp8: true to config
echo "fp8: true" >> configs/my_model.yaml

# Step 2: Use accelerate launch
accelerate launch --config_file configs/accelerate_fp8.yaml \
  llamafactory-cli train configs/my_model.yaml
```

## Implementation Reference

I've implemented this approach in a fork:
- **Branch:** [`fix/accelerate-config-support`](https://github.com/sbhavani/LLaMA-Factory/tree/fix/accelerate-config-support)
- **Key file:** [`src/llamafactory/train/fp8_utils.py`](https://github.com/sbhavani/LLaMA-Factory/blob/fix/accelerate-config-support/src/llamafactory/train/fp8_utils.py)
- **Complete setup:** [llamafactory-fp8-hopper repo](https://github.com/sbhavani/llamafactory-fp8-hopper)

## Expected Performance

Tested on NVIDIA H100 and B200:

| Model Size | Precision | Speed (samples/s) | Memory (GB) | Speedup |
|------------|-----------|-------------------|-------------|---------|
| Qwen 7B    | BF16      | 27.8              | 45          | 1.0x    |
| Qwen 7B    | FP8       | 37.5              | 32          | 1.35x   |
| Llama 8B   | BF16      | 24.2              | 52          | 1.0x    |
| Llama 8B   | FP8       | 33.8              | 38          | 1.40x   |

**Note:** FP8 benefits only show for models 7B+ parameters. Smaller models may not see speedup due to overhead.

## Documentation Needed

If this is implemented, docs should cover:

1. **Quick Start:**
   - How to create Accelerate config
   - How to enable `fp8: true` in LLaMA-Factory config
   - How to use `accelerate launch`

2. **Hardware Requirements:**
   - NVIDIA H100, H200, B200, or RTX 4090
   - CUDA 12.1+
   - Transformer Engine installation

3. **Backend Options:**
   - `backend: TE` (Transformer Engine - best for Hopper/Blackwell)
   - `backend: MSAMP` (Microsoft AMP)
   - `backend: torchao` (PyTorch AO)

4. **Troubleshooting:**
   - Model too small (< 7B)
   - Memory issues
   - Missing dependencies

## Alternative: One-Time Accelerate Setup

Users could also do a one-time setup:

```bash
# One-time: Configure Accelerate
accelerate config
# Select "fp8" when prompted for mixed precision

# Then use LLaMA-Factory normally
llamafactory-cli train configs/my_model.yaml
```

But explicit `--config_file` is more reproducible.

## System Information

```
# Output of llamafactory-cli env
<PASTE YOUR OUTPUT HERE>
```

Example:
```
- Platform: Linux-5.15.0-1063-nvidia-x86_64-with-glibc2.35
- Python version: 3.12.0
- PyTorch version: 2.4.0+cu121
- Transformers version: 4.45.0
- Datasets version: 3.1.0
- Accelerate version: 1.1.1
- PEFT version: 0.13.2
- TRL version: 0.11.4
- GPU type: NVIDIA H100 80GB HBM3
- Transformer Engine version: 1.11.0
```

## Benefits to LLaMA-Factory Users

1. **Performance:** 1.3-1.5x faster training for large models
2. **Memory:** 20-30% reduction enables larger batches or longer sequences
3. **Cost:** Reduced cloud GPU costs due to faster training
4. **Compatibility:** Works with existing configs, just add `fp8: true`
5. **Future-proof:** Leverages official HuggingFace integration

## Related

- HuggingFace Accelerate FP8 docs: https://huggingface.co/docs/accelerate/usage_guides/low_precision_training
- HuggingFace Transformers issue #25333: https://github.com/huggingface/transformers/issues/25333
- NVIDIA Transformer Engine: https://docs.nvidia.com/deeplearning/transformer-engine/
- Original Accelerate FP8 benchmarks: https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8

## Conclusion

Adding FP8 support via Accelerate config files would:
- ✅ Enable 1.3-1.5x speedup for LLaMA-Factory users
- ✅ Require minimal code changes (just config detection)
- ✅ Follow official HuggingFace best practices
- ✅ Work with all training modes (full, LoRA, DeepSpeed, FSDP)
- ✅ Be maintainable long-term (no hacks or monkey-patches)

I'm happy to submit a PR if this approach is acceptable to the maintainers!

---

**Additional Context:**

I've spent considerable time debugging FP8 integration with LLaMA-Factory and discovered that the proper approach is via Accelerate config files, not programmatic configuration. I've verified this works correctly with both synthetic benchmarks and real training runs, showing the expected 1.3-1.5x speedup on H100/B200 GPUs.

The implementation in my fork is production-ready and could be merged with minimal changes. Happy to collaborate on getting this into mainline LLaMA-Factory!
