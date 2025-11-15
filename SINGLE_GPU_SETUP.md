# Single GPU Setup Guide (1x GH200)

This guide covers the specific configuration for running FP8 training on a single GH200 GPU.

## 🎯 Key Differences for Single GPU

### 1. **Simplified DeepSpeed Configuration**

For single GPU, you can use ZeRO-0 (no sharding) or skip DeepSpeed entirely:

**Option A: Use ZeRO-0 Config**
```yaml
# In your training YAML
deepspeed: /workspace/configs/ds_z0_single_gpu_config.json
```

**Option B: Remove DeepSpeed** (Recommended for single GPU)
```yaml
# Comment out or remove the deepspeed line
# deepspeed: /workspace/configs/ds_z1_fp8_config.json
```

### 2. **Environment Variables**

```bash
# Single GPU settings
export GPU_NUM=1
export WORLD_SIZE=1
export RANK=0

# No multi-node settings needed
# MASTER_ADDR, MASTER_PORT not required

# Keep FP8 optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=1
export NVTE_FP8_DPA_BWD=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NVTE_FP8_ALLREDUCE=1
```

### 3. **Training Command**

```bash
# Single GPU - use python directly (no torchrun needed)
python src/train.py /workspace/configs/llama3_fp8_deepspeed_sft.yaml

# Or with torchrun (works but not necessary)
torchrun --nproc_per_node=1 \
    --standalone \
    src/train.py /workspace/configs/llama3_fp8_deepspeed_sft.yaml
```

## 📝 GH200-Specific Notes

### Architecture

GH200 = Grace CPU (ARM) + Hopper GPU (H100)

- **GPU**: Full Hopper H100 with 96GB HBM3
- **CPU**: Grace ARM CPU with coherent memory
- **Key Benefit**: 900GB/s CPU-GPU bandwidth (7x faster than PCIe Gen5)

### FP8 Support

GH200's Hopper GPU has **identical FP8 support** to standalone H100:
- ✅ FP8 Tensor Cores
- ✅ Hardware delayed scaling
- ✅ E4M3 and E5M2 formats
- ✅ All Transformer Engine optimizations

### Memory Configuration

**For Llama-3.1-8B:**
- Model parameters: ~16GB (BF16)
- Activations + gradients: ~30-40GB (depends on batch size)
- Optimizer states: ~24GB (AdamW with BF16)
- **Total**: ~70-80GB

**Recommendation:**
- Use `per_device_train_batch_size: 4` or higher
- With GH200's 96GB, you have headroom for larger batches!

## 🚀 Quick Start for 1x GH200

### Step 1: Build Container

```bash
cd llamafactory-fp8-test
docker build -t llamafactory-fp8:latest .
```

### Step 2: Run Container

```bash
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  llamafactory-fp8:latest
```

### Step 3: Verify Setup

```bash
# Inside container
python /workspace/scripts/verify_fp8.py
```

Expected output should show:
```
GPU 0: NVIDIA GH200 120GB
Compute Capability: 9.0
✅ FP8 supported
```

### Step 4: Run BF16 Baseline

```bash
# Edit the script to set single GPU
export GPU_NUM=1
bash /workspace/scripts/train_bf16.sh
```

### Step 5: Run FP8 Training

```bash
export GPU_NUM=1
bash /workspace/scripts/train_fp8.sh
```

## ⚙️ Optimized Configuration for GH200

### Recommended YAML Settings

```yaml
# llama3_fp8_gh200_sft.yaml
model_name_or_path: meta-llama/Meta-Llama-3.1-8B
stage: sft
do_train: true
finetuning_type: full

dataset: ultrachat_200k
template: llama3
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16

output_dir: /workspace/checkpoints/llama3-fp8-gh200
logging_steps: 10
overwrite_output_dir: true

# Optimized batch size for GH200's 96GB
per_device_train_batch_size: 6  # Can go higher than multi-GPU!
gradient_accumulation_steps: 4   # Adjust to maintain effective batch size
gradient_checkpointing: true

learning_rate: 2.0e-4
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_steps: 100

# FP8 Configuration
fp8: true
fp8_backend: te
bf16: true
tf32: true

# For single GPU, DeepSpeed is optional
# deepspeed: /workspace/configs/ds_z0_single_gpu_config.json

# Attention
flash_attn: fa2

# Data loading
dataloader_num_workers: 8  # Grace CPU has many cores!

report_to: none
```

### Key Optimizations for GH200:

1. **Larger Batch Size**: GH200 has 96GB, use it!
   ```yaml
   per_device_train_batch_size: 6  # vs 4 on typical H100
   ```

2. **More Data Workers**: Grace ARM CPU has 72 cores
   ```yaml
   dataloader_num_workers: 8  # vs 4 on x86
   preprocessing_num_workers: 32  # vs 16
   ```

3. **No DeepSpeed Overhead**: Single GPU doesn't need ZeRO
   ```yaml
   # Comment out deepspeed line
   # Or use ZeRO-0 for minimal overhead
   ```

## 📊 Expected Performance on 1x GH200

### Llama-3.1-8B, Batch Size 6, Seq Len 4096

| Metric | BF16 | FP8 | Improvement |
|--------|------|-----|-------------|
| **Iteration Time** | ~8.2 s/it | ~5.5 s/it | **1.49x faster** |
| **Throughput** | ~0.73 samples/s | ~1.09 samples/s | **1.49x higher** |
| **Memory Usage** | ~85GB | ~75GB | **12% less** |
| **GPU Utilization** | ~92% | ~95% | **Better** |

## 🐛 Troubleshooting Single GPU

### Issue: "No module named 'deepspeed'"

**Solution**: DeepSpeed not needed for single GPU
```yaml
# Remove or comment out in YAML:
# deepspeed: /workspace/configs/ds_z1_fp8_config.json
```

### Issue: "Address already in use"

**Solution**: NCCL trying to bind port for distributed training
```bash
# Set single GPU mode explicitly
export WORLD_SIZE=1
export RANK=0

# Or use standalone flag
torchrun --standalone --nproc_per_node=1 ...
```

### Issue: "CUDA out of memory" on 96GB GPU

**Solution**: Even GH200 has limits!
```yaml
# Reduce batch size
per_device_train_batch_size: 4  # Down from 6
# Or reduce sequence length
cutoff_len: 2048  # Down from 4096
# Or enable CPU offloading
optim_state_cpu_offload: true
```

### Issue: FP8 slower than expected

**Check:**
1. Verify backend is TE (not TorchAO)
   ```bash
   grep "FP8 backend" /workspace/checkpoints/*/training.log
   # Should show: "te" not "torchao"
   ```

2. Confirm FP8 environment variables are set
   ```bash
   env | grep NVTE
   ```

3. Check GPU utilization
   ```bash
   nvidia-smi dmon -s pucvmet
   # Should see >90% GPU utilization
   ```

## 🎯 Performance Tuning Tips

### 1. Maximize Batch Size

GH200 has 96GB - use it!
```bash
# Try incrementally larger batches
per_device_train_batch_size: 6  # Start here
per_device_train_batch_size: 8  # If no OOM
per_device_train_batch_size: 10 # Push the limits
```

### 2. Optimize Data Loading

Grace CPU is powerful - leverage it:
```yaml
dataloader_num_workers: 8      # More workers
preprocessing_num_workers: 32  # Heavy preprocessing
dataloader_pin_memory: true    # Faster GPU transfer
```

### 3. Profile to Find Bottlenecks

```bash
# Use NVIDIA Nsight Systems
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=gh200_profile.qdrep \
  python src/train.py your_config.yaml

# View results
nsight-sys gh200_profile.qdrep
```

### 4. Monitor Memory Bandwidth

```bash
# GH200's 900GB/s CPU-GPU bandwidth is a key advantage
nvidia-smi dmon -s m
# Watch for high memory copy activity
```

## ✅ Verification Checklist

Before considering your single GPU setup optimized:

- [ ] `verify_fp8.py` passes all checks
- [ ] Training logs show `FP8 backend: te`
- [ ] Single GPU mode confirmed (`WORLD_SIZE=1`)
- [ ] FP8 is 1.4-1.5x faster than BF16
- [ ] GPU utilization >90%
- [ ] No OOM errors at reasonable batch size
- [ ] Data loading not bottlenecking (check with profiler)
- [ ] Memory usage reasonable (~75-85GB for 8B model)

## 📚 Additional Resources

- [GH200 Technical Brief](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)

---

**You're ready to run FP8 training on your GH200!** 🚀
