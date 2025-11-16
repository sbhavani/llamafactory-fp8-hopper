# LLaMA-Factory FP8 Training Setup

Production-ready FP8 training for LLaMA-Factory on NVIDIA Hopper/Blackwell GPUs (H100, H200, B200).

## Quick Start

### Docker Setup

```bash
# Build
docker build -t llamafactory-fp8:latest .

# Run
docker run --gpus all --ipc=host -v $(pwd)/checkpoints:/workspace/checkpoints -v $(pwd)/configs:/workspace/configs -v /tmp:/tmp -it llamafactory-fp8:latest bash

# Inside container - Train with FP8
bash /workspace/scripts/train_fp8_llamafactory.sh /workspace/configs/qwen_7b_fp8_b200.yaml
```

### Remote Server Setup (No Docker)

```bash
# One-command install
wget -O ~/install_fp8.sh https://raw.githubusercontent.com/sbhavani/llamafactory-fp8-hopper/master/install_fp8_fixed.sh
bash ~/install_fp8.sh

# Use it
source ~/llamafactory-fp8/setup.sh
cd ~/llamafactory-fp8/LLaMA-Factory
bash scripts/train_fp8.sh configs/qwen_7b_fp8_b200.yaml
```

See `REMOTE_SETUP.md` for details.

## How It Works

FP8 is enabled via **Accelerate config files** (the official HuggingFace way):

```bash
accelerate launch --config_file configs/accelerate_fp8.yaml \
  llamafactory-cli train configs/my_model.yaml
```

Your LLaMA-Factory configs just need:

```yaml
fp8: true
fp8_backend: te
bf16: true  # BF16 for non-FP8 ops
```

## Config Files

**Accelerate configs:**
- `configs/accelerate_fp8.yaml` - FP8 configuration
- `configs/accelerate_bf16.yaml` - BF16 baseline

**Training configs:**
- `configs/qwen_7b_fp8_b200.yaml` - Optimized for B200 (192GB)
- `configs/qwen_7b_bf16_b200.yaml` - BF16 baseline
- See `configs/` for more examples

## Performance

Expected on H100/B200 for 7B+ models:

| Metric | BF16 | FP8 | Improvement |
|--------|------|-----|-------------|
| Speed | 1.0x | 1.3-1.5x | 30-50% faster |
| Memory | 100% | 70-80% | 20-30% saved |

## LLaMA-Factory Fork

Uses fork with Accelerate config support: [`sbhavani/LLaMA-Factory:fix/accelerate-config-support`](https://github.com/sbhavani/LLaMA-Factory/tree/fix/accelerate-config-support)

**Key change:** Detects when Accelerate is already configured and skips conflicting setup.

## Documentation

- `REMOTE_SETUP.md` - Complete remote server setup guide
- `LLAMAFACTORY_USER_GUIDE.md` - Guide for existing LLaMA-Factory users
- `AUTHENTICATION.md` - HuggingFace authentication

## Contributing

See `.llm/` directory for:
- Bug reports and feature requests
- Implementation details
- Testing procedures

## Requirements

- NVIDIA H100, H200, B200, or RTX 4090
- CUDA 12.1+
- Python 3.10+
- 80GB+ VRAM (for 7B models)

## License

Same as LLaMA-Factory (Apache 2.0)
