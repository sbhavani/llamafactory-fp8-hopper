# LLaMA-Factory FP8 Test Harness

Test environment for FP8 training with LLaMA-Factory on NVIDIA Hopper GPUs (H100/H200/B200).

## Quick Start

```bash
# Build
docker build -t llamafactory-fp8:latest .

# Run
docker run --gpus all --ipc=host -it llamafactory-fp8:latest bash

# Inside container - run benchmark
bash /workspace/scripts/train_bf16.sh  # BF16 baseline
bash /workspace/scripts/train_fp8.sh   # FP8 (requires Hopper GPU)

# Compare results
python /workspace/scripts/compare_performance.py
```

## Requirements

- NVIDIA H100, H200, or B200 GPU (FP8 support)
- Docker with NVIDIA Container Toolkit
- 80GB+ VRAM for 7B models

## Files

```
configs/
  accelerate_bf16.yaml          # Accelerate config for BF16
  accelerate_fp8.yaml           # Accelerate config for FP8
  qwen_7b_bf16_full_benchmark.yaml
  qwen_7b_fp8_full_benchmark.yaml

scripts/
  train_bf16.sh                 # BF16 training script
  train_fp8.sh                  # FP8 training script
  compare_performance.py        # Compare BF16 vs FP8 results
  verify_fp8.py                 # Verify FP8 environment
```

## LLaMA-Factory Fork

Uses fork with FP8/Accelerate fixes: [sbhavani/LLaMA-Factory@fix/accelerate-config-support](https://github.com/sbhavani/LLaMA-Factory/tree/fix/accelerate-config-support)

## License

Apache 2.0
