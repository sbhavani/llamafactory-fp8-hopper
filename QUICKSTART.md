# Quick Start Guide

Get up and running with FP8 testing in 5 minutes!

## ⚡ TL;DR

```bash
# 1. Build and run
docker-compose up -d
docker-compose exec llamafactory-fp8 bash

# 2. Inside container: verify setup
python /workspace/scripts/verify_fp8.py

# 3. Run tests
bash /workspace/scripts/train_bf16.sh  # Baseline
bash /workspace/scripts/train_fp8.sh   # FP8 test

# 4. Compare results
python /workspace/scripts/compare_performance.py
```

## 📋 Step-by-Step Guide

### Step 1: Prerequisites Check

Ensure you have:
- ✅ NVIDIA GPU with Compute Capability 8.9+ (H100, Hopper GPU)
- ✅ Docker with NVIDIA Container Toolkit
- ✅ At least 80GB GPU memory (for Llama-3.1-8B)

Check your GPU:
```bash
nvidia-smi
```

### Step 2: Build the Environment

```bash
cd llamafactory-fp8-test

# Option A: Using docker-compose (recommended)
docker-compose up -d

# Option B: Using docker directly
docker build -t llamafactory-fp8:latest .
docker run -it --rm --gpus all --ipc=host \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  llamafactory-fp8:latest
```

### Step 3: Enter the Container

```bash
docker-compose exec llamafactory-fp8 bash
```

You should see:
```
root@container:/workspace/LLaMA-Factory#
```

### Step 4: Verify FP8 Support

Run the verification script:
```bash
python /workspace/scripts/verify_fp8.py
```

Expected output:
```
==============================================================
FP8 Environment Verification
==============================================================

==============================================================
CUDA Check
==============================================================
✅ CUDA is available
   Version: 12.6
   Device count: 8
   GPU 0: NVIDIA Hopper GPU
      Compute Capability: 9.0
      Memory: 192.00 GB
      ✅ FP8 supported (CC 9.0)

==============================================================
Transformer Engine Check
==============================================================
✅ Transformer Engine installed
   Version: 1.11.0
✅ FP8 recipe creation successful
✅ TE Linear module creation successful

==============================================================
Summary
==============================================================
✅ PASS: CUDA
✅ PASS: Transformer Engine
✅ PASS: Accelerate
✅ PASS: DeepSpeed
✅ PASS: LLaMA-Factory

✅ All checks passed! Ready for FP8 training.
```

### Step 5: Run BF16 Baseline

This establishes your baseline performance:

```bash
bash /workspace/scripts/train_bf16.sh
```

Watch for output like:
```
Starting BF16 training...
[GPU-0] Training started
[GPU-0] Step 1/100: 8.15s/it
[GPU-0] Step 2/100: 8.12s/it
...
```

### Step 6: Run FP8 Training

Now test with FP8:

```bash
bash /workspace/scripts/train_fp8.sh
```

Look for:
```
✅ Transformer Engine version: 1.11.0
✅ FP8 training enabled with te backend
✅ Accelerate FP8 status - enabled: True

Starting FP8 training...
[GPU-0] Training started
[GPU-0] Step 1/100: 5.42s/it  ← Should be FASTER than BF16!
[GPU-0] Step 2/100: 5.38s/it
...
```

### Step 7: Compare Performance

```bash
python /workspace/scripts/compare_performance.py
```

Expected output:
```
======================================================================
FP8 vs BF16 Performance Comparison
======================================================================

📊 BF16 Baseline:
   Average iteration time: 8.15 s/it
   Samples: 100
   Range: 8.05 - 8.25 s/it

📊 FP8 Training:
   Average iteration time: 5.42 s/it
   Samples: 100
   Range: 5.35 - 5.50 s/it

🚀 Performance Comparison:
   Speedup: 1.50x
   ✅ FP8 is 33.5% FASTER than BF16

📈 Expected Results on Hopper GPU:
   Target speedup: ~1.3-1.5x
   ✅ WITHIN EXPECTED RANGE!

======================================================================
```

## 🔍 What to Look For

### ✅ Good Signs (FP8 Working Correctly)

1. **Verification passes all checks**
2. **Training logs show:**
   - `Transformer Engine version: X.X.X`
   - `FP8 training enabled with te backend`
   - `Accelerate FP8 status - enabled: True`
3. **Iteration time is ~1.3-1.5x faster than BF16**
4. **No "FP8 NOT supported" warnings**

### ❌ Bad Signs (Issues Present)

1. **Verification fails checks**
2. **Training logs show:**
   - `FP8 backend: torchao` (should be `te`)
   - `WARNING: Transformer Engine not found`
   - `fp8_enabled=False`
3. **Iteration time is SLOWER than BF16** (like Users's issue!)
4. **OOM errors or CUDA errors**

## 🐛 Quick Troubleshooting

### FP8 is Slower than BF16

**Most Common Causes:**

1. **Wrong backend being used**
   ```bash
   # Check logs for:
   grep "FP8 backend" /workspace/checkpoints/*/training.log
   # Should show: "FP8 backend: te"
   # NOT: "FP8 backend: torchao"
   ```

2. **DeepSpeed config has FP32 communication**
   ```bash
   # Check your DeepSpeed config has NO:
   # - "fp8" section
   # - "communication_data_type" field
   cat /workspace/configs/ds_z1_fp8_config.json | grep -E "fp8|communication_data_type"
   # Should return nothing!
   ```

3. **Missing environment variables**
   ```bash
   # Inside container, verify:
   env | grep NVTE
   # Should show multiple NVTE_* variables
   ```

### Out of Memory Errors

**Solutions:**
1. Reduce batch size in YAML config:
   ```yaml
   per_device_train_batch_size: 2  # Was 4
   gradient_accumulation_steps: 16  # Was 8
   ```

2. Enable gradient checkpointing (already enabled in configs)

3. Use smaller model for testing:
   ```yaml
   model_name_or_path: meta-llama/Meta-Llama-3-8B  # Smaller version
   ```

### Container Won't Start

**Check Docker GPU support:**
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If this fails, install NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 🎯 Next Steps

### For Production Use

1. **Scale up training:**
   - Remove `max_samples: 1000` from YAML
   - Increase `num_train_epochs` as needed
   - Enable W&B logging: `report_to: wandb`

2. **Multi-node setup:**
   ```bash
   # On each node, set:
   export WORLD_SIZE=2
   export RANK=0  # 0 for first node, 1 for second, etc.
   export MASTER_ADDR=192.168.0.1  # First node's IP
   ```

3. **Use full dataset:**
   ```yaml
   dataset: your_custom_dataset
   max_samples: null  # Use all data
   ```

### For Development

1. **Enable debug logging:**
   ```bash
   export NVTE_DEBUG=1
   export NVTE_DEBUG_LEVEL=2
   export NCCL_DEBUG=INFO
   ```

2. **Profile training:**
   ```bash
   nsys profile --trace cuda,nvtx -o profile.qdrep \
     python src/train.py /workspace/configs/llama3_fp8_deepspeed_sft.yaml
   ```

3. **Monitor GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```

## 📚 Additional Resources

- [Full README](README.md) - Complete documentation
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Detailed debugging
- [Configuration Reference](CONFIGURATION.md) - All config options

## 💬 Getting Help

If you encounter issues:

1. Run `verify_fp8.py` and share output
2. Check training logs in `/workspace/checkpoints/`
3. Review the [Troubleshooting Guide](TROUBLESHOOTING.md)
4. Open an issue with:
   - GPU model and driver version
   - Verification script output
   - Training log excerpts
   - Error messages

## ✅ Success Checklist

Before considering your setup working, verify:

- [ ] `verify_fp8.py` passes all checks
- [ ] Training logs show `FP8 backend: te`
- [ ] Training logs show `fp8_enabled=True`
- [ ] FP8 training is 1.3-1.5x faster than BF16
- [ ] No CUDA or OOM errors
- [ ] GPU utilization is >90% during training
- [ ] Loss curves look normal (no NaN/Inf)

Once all checked, you're ready for production FP8 training! 🚀
