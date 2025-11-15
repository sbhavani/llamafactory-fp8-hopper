# LLaMA-Factory FP8 Testing on Hopper GPUs

This repository provides a complete testing environment for FP8 training with LLaMA-Factory on NVIDIA Hopper GPUs (H100, GH200), demonstrating proper configuration and fixing common performance issues.

## 🔴 Problem Statement

Common issue reported: **2x slowdown** with FP8 compared to BF16 on Hopper GPUs:
- **BF16**: 8.15s/it
- **FP8**: 16.29s/it (should be ~5-6s/it!)

## ✅ Solution

This test environment provides:
1. **Corrected DeepSpeed configuration** (removes invalid FP8 section)
2. **Proper FP8 backend selection** (Transformer Engine, not TorchAO)
3. **Hopper-specific optimizations** (NVLink, communication overlap)
4. **Proper environment variable configuration**
5. **Performance comparison tools**

## 📋 Prerequisites

- NVIDIA Hopper GPU: H100 or GH200 (Compute Capability 8.9+)
- Docker with GPU support
- NVIDIA Container Toolkit
- For single GPU: 1x GH200/H100 (80GB+ VRAM recommended)

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
cd llamafactory-fp8-test
docker build -t llamafactory-fp8:latest .
```

### 2. Run the Container

```bash
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/configs:/workspace/configs \
  llamafactory-fp8:latest
```

### 3. Verify FP8 Setup

Inside the container:

```bash
python /workspace/scripts/verify_fp8.py
```

This checks:
- ✅ CUDA availability and FP8 hardware support
- ✅ Transformer Engine installation
- ✅ HuggingFace Accelerate FP8 support
- ✅ DeepSpeed installation
- ✅ LLaMA-Factory FP8 utilities
- ✅ Environment variables

### 4. Run BF16 Baseline Training

```bash
bash /workspace/scripts/train_bf16.sh
```

### 5. Run FP8 Training

```bash
bash /workspace/scripts/train_fp8.sh
```

### 6. Compare Performance

```bash
python /workspace/scripts/compare_performance.py
```

## 📊 Expected Results

### On Hopper GPUs with Proper Configuration:

| Metric | BF16 | FP8 (Corrected) | Speedup |
|--------|------|-----------------|---------|
| Iteration Time | 8.15 s/it | **~5.4-6.3 s/it** | **~1.3-1.5x** |
| Memory Usage | Baseline | ~70% of baseline | 30% reduction |
| Throughput | Baseline | 1.3-1.5x baseline | 30-50% faster |

### Common Configuration Errors:

❌ DeepSpeed config had invalid `"fp8"` section  
❌ Communication data type set to FP32 (expensive casting)  
❌ Missing `CUDA_DEVICE_MAX_CONNECTIONS=1` for overlap  
❌ Missing `NVTE_FP8_ALLREDUCE=1` for Hopper optimizations  
❌ FP8 backend defaulting to TorchAO instead of TE  
❌ NVTE environment variables being ignored  

## 🔧 Configuration Files

### DeepSpeed Config (`ds_z1_fp8_config.json`)

```json
{
  "zero_optimization": {
    "stage": 1,
    "overlap_comm": false,
    "contiguous_gradients": true
  }
}
```

**Key Points:**
- ✅ NO `"fp8"` section (DeepSpeed doesn't handle FP8!)
- ✅ NO `"communication_data_type"` (let TE handle it)
- ✅ NO `"data_types"` section

### Training Config (`llama3_fp8_deepspeed_sft.yaml`)

```yaml
fp8: true
fp8_backend: te  # Explicitly use Transformer Engine
bf16: true       # For non-FP8 layers
deepspeed: /workspace/configs/ds_z1_fp8_config.json
```

**Key Points:**
- ✅ `fp8_backend: te` explicitly set (not default `auto`)
- ✅ DeepSpeed only for optimizer sharding
- ✅ FP8 handled by Accelerate + Transformer Engine

### Environment Variables

```bash
# Critical for Hopper performance
export CUDA_DEVICE_MAX_CONNECTIONS=1         # Enable comm-compute overlap
export NVTE_FP8_ALLREDUCE=1                  # Hopper FP8 all-reduce
export NVTE_APPLY_QK_LAYER_SCALING=1         # QK layer scaling
export NVTE_FP8_DPA_BWD=1                    # FP8 in attention backward
export NVTE_FLASH_ATTN=1                     # Flash attention with FP8
export NVTE_FUSED_ATTN=1                     # Fused attention
```

## 🐛 Debugging

### Enable Debug Output

```bash
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=1
```

### Check FP8 is Actually Running

Look for these in the logs:

```
Transformer Engine version: X.X.X
FP8 training enabled with te backend
Accelerate FP8 status - enabled: True
```

### If FP8 is Still Slower

1. **Verify backend**: Check logs for `FP8 backend: te` (not `torchao`)
2. **Check GPU**: Must be CC 8.9+ (H100, GH200, etc.)
3. **Verify environment**: Run `verify_fp8.py` script
4. **Check DeepSpeed config**: Should NOT have `"fp8"` or `"communication_data_type"` sections

## 📁 Repository Structure

```
llamafactory-fp8-test/
├── Dockerfile                          # NGC PyTorch + LLaMA-Factory
├── README.md                           # This file
├── configs/
│   ├── ds_z1_fp8_config.json          # Corrected DeepSpeed config
│   ├── bf16_baseline_config.json      # BF16 baseline config
│   ├── llama3_fp8_deepspeed_sft.yaml  # FP8 training config
│   └── llama3_bf16_baseline_sft.yaml  # BF16 training config
├── scripts/
│   ├── train_fp8.sh                   # FP8 training script
│   ├── train_bf16.sh                  # BF16 baseline script
│   ├── verify_fp8.py                  # Environment verification
│   └── compare_performance.py         # Performance comparison
└── checkpoints/                        # Training outputs (created at runtime)
```

## 🎯 Key Differences from Common Broken Configs

| Aspect | Common Broken Config | This Config | Impact |
|--------|---------------|-------------|--------|
| DeepSpeed FP8 section | ❌ Present | ✅ Removed | No conflict |
| Communication dtype | ❌ FP32 | ✅ Auto (FP8) | 2x speedup |
| FP8 backend | ❌ Auto (→TorchAO) | ✅ Explicit TE | Proper FP8 |
| CUDA_DEVICE_MAX_CONNECTIONS | ❌ Missing | ✅ Set to 1 | Overlap enabled |
| NVTE_FP8_ALLREDUCE | ❌ Missing | ✅ Set to 1 | Hopper optimization |
| fp8_backend flag | ❌ Not set | ✅ `te` | Uses TE not TorchAO |

## 🔗 References

### Documentation
- [Transformer Engine User Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [HuggingFace Accelerate FP8 Guide](https://huggingface.co/docs/accelerate/usage_guides/low_precision_training)
- [LLaMA-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)

### Related Issues
- [HuggingFace Accelerate FP8 Examples](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/transformer_engine)
- [Transformer Engine Examples](https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch)

## 📝 Notes

### Multi-Node Training

For multi-node setups, set these on each node:

```bash
export GPU_NUM=8                    # GPUs per node
export WORLD_SIZE=2                 # Number of nodes
export RANK=0                       # Node rank (0, 1, 2, ...)
export MASTER_ADDR=192.168.0.1      # First node's IP
export MASTER_PORT=29500
```

### Custom Model Paths

Edit the YAML configs to point to your model:

```yaml
model_name_or_path: /path/to/your/model
```

### Custom Network Interfaces

Uncomment and adjust in training scripts:

```bash
export GLOO_SOCKET_IFNAME=enp50s0   # Your network interface
export NCCL_SOCKET_IFNAME=enp50s0
```

## 🤝 Contributing

Found an issue or improvement? Please open an issue or PR!

## 📜 License

Apache 2.0 (same as LLaMA-Factory)

## 🙏 Acknowledgments

- NVIDIA Transformer Engine team
- HuggingFace Accelerate team
- LLaMA-Factory maintainers
- Community contributors for reporting issues
