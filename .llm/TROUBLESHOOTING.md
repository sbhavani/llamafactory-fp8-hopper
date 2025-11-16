# Troubleshooting Guide

Detailed solutions for common issues when running FP8 training.

## 📊 Issue: FP8 is Slower than BF16

**Symptom:** FP8 training shows **16.29s/it** vs BF16's **8.15s/it** (2x slower!)

This is the exact issue Users encountered. Here's how to diagnose and fix it:

### Root Cause Analysis

Run this diagnostic:

```bash
# 1. Check which backend is being used
cd /workspace/LLaMA-Factory
python -c "
from llamafactory.hparams import ModelArguments
args = ModelArguments(model_name_or_path='dummy')
args.fp8 = True
args.fp8_backend = 'auto'  # Check what 'auto' resolves to
print(f'Backend: {args.fp8_backend}')
"

# 2. Check DeepSpeed config
cat /workspace/configs/ds_z1_fp8_config.json | python -m json.tool

# 3. Check environment variables
env | grep -E "NVTE|CUDA_DEVICE"
```

### Solution Checklist

#### ✅ Fix 1: Set Explicit TE Backend

**Problem:** `fp8_backend: auto` defaults to TorchAO, not Transformer Engine

**Fix:** Edit `/workspace/configs/llama3_fp8_deepspeed_sft.yaml`:

```yaml
fp8: true
fp8_backend: te  # ← MUST be explicit, not "auto"
```

Verify in logs:
```bash
grep "FP8 backend" /workspace/checkpoints/llama3-fp8-deepspeed/training.log
# Should show: "FP8 training enabled with te backend"
```

#### ✅ Fix 2: Remove Invalid DeepSpeed FP8 Section

**Problem:** DeepSpeed config has `"fp8"` section that doesn't work

**Check:**
```bash
grep -A 10 '"fp8"' /workspace/configs/ds_z1_fp8_config.json
```

If you see:
```json
"fp8": {
  "enabled": true,
  "loss_scale": 0,
  ...
}
```

**Fix:** Remove entire section! DeepSpeed doesn't handle FP8 training. Use this config:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": false
  },
  "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
```

#### ✅ Fix 3: Remove FP32 Communication

**Problem:** Config forces FP32 for all-reduce (expensive casting!)

**Check:**
```bash
grep "communication_data_type\|grad_accum_dtype" /workspace/configs/ds_z1_fp8_config.json
```

If you see:
```json
"communication_data_type": "fp32",
"data_types": {
  "grad_accum_dtype": "fp32"
}
```

**Fix:** Remove these fields completely! Let Transformer Engine handle data types.

#### ✅ Fix 4: Enable Communication Overlap

**Problem:** Missing `CUDA_DEVICE_MAX_CONNECTIONS=1`

**Check:**
```bash
echo $CUDA_DEVICE_MAX_CONNECTIONS
# Should output: 1
```

**Fix:** Add to training script:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

#### ✅ Fix 5: Enable Hopper GPU Optimizations

**Problem:** Missing Hopper GPU-specific environment variables

**Check:**
```bash
env | grep NVTE
```

**Fix:** Ensure these are set:
```bash
export NVTE_APPLY_QK_LAYER_SCALING=1
export NVTE_FP8_DPA_BWD=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=1
export NVTE_FP8_ALLREDUCE=1  # Hopper GPU-specific
```

### Verification

After applying fixes, you should see in logs:

```
✅ Transformer Engine version: 1.11.0
✅ FP8 training enabled with te backend  ← Not "torchao"!
✅ Accelerate FP8 status - enabled: True, backend: TransformerEngineBackend
```

And performance should improve from **16.29s/it** → **~5.4-6.3s/it** (1.3-1.5x speedup).

---

## 🚫 Issue: ImportError: No module named 'transformer_engine'

**Symptom:**
```
ImportError: No module named 'transformer_engine'
```

### Solution 1: Reinstall in Container

```bash
pip uninstall -y transformer-engine
pip install --no-cache-dir transformer-engine[pytorch]

# Verify
python -c "import transformer_engine; print(transformer_engine.__version__)"
```

### Solution 2: Check PyTorch Compatibility

Transformer Engine requires specific PyTorch versions:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should be 2.1.0+

python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
# Should be 12.1+
```

If versions are wrong, rebuild container:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 💾 Issue: Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

### Solution 1: Reduce Batch Size

Edit YAML config:
```yaml
per_device_train_batch_size: 2  # Reduce from 4
gradient_accumulation_steps: 16  # Increase to maintain effective batch
```

### Solution 2: Enable Memory Optimizations

Add to YAML:
```yaml
gradient_checkpointing: true
optim: adamw_torch_fused  # More memory efficient
```

### Solution 3: Use ZeRO-3 Instead of ZeRO-1

Edit YAML:
```yaml
deepspeed: /workspace/configs/ds_z3_config.json  # Create ZeRO-3 config
```

Create `/workspace/configs/ds_z3_config.json`:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### Solution 4: Use Smaller Model

For testing, use Llama-3-8B instead of Llama-3.1-8B:
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B
```

---

## ⚠️ Issue: FP8 Not Actually Being Used

**Symptom:** Training runs but no speedup, logs don't mention FP8

### Diagnosis

```bash
# Check if FP8 is enabled in logs
grep -i fp8 /workspace/checkpoints/llama3-fp8-deepspeed/training.log

# Should see:
# ✅ "FP8 training enabled"
# ✅ "fp8_backend: te"
# ✅ "Accelerate FP8 status - enabled: True"
```

### Solution 1: Check GPU Support

```bash
python -c "
import torch
props = torch.cuda.get_device_properties(0)
cc = f'{props.major}.{props.minor}'
print(f'Compute Capability: {cc}')
if props.major > 8 or (props.major == 8 and props.minor >= 9):
    print('✅ FP8 supported')
else:
    print(f'❌ FP8 NOT supported (need CC 8.9+, have {cc})')
"
```

If FP8 not supported, you need H100, Hopper GPU, or Hopper+ GPU.

### Solution 2: Verify Accelerate Configuration

```bash
python -c "
from accelerate import __version__
major, minor = map(int, __version__.split('.')[:2])
if major > 1 or (major == 1 and minor >= 8):
    print(f'✅ Accelerate {__version__} supports FP8')
else:
    print(f'❌ Accelerate {__version__} too old, need 1.8.0+')
"
```

### Solution 3: Check YAML Config

Ensure your YAML has:
```yaml
fp8: true  # ← Must be explicitly true
fp8_backend: te  # ← Must be "te", not "auto" or "torchao"
```

---

## 🔌 Issue: NCCL/Communication Errors

**Symptom:**
```
NCCL error: unhandled system error
RuntimeError: NCCL error in: ...
```

### Solution 1: Set Network Interface

Edit training script:
```bash
# Find your network interface
ip addr show

# Set it (e.g., for enp50s0)
export GLOO_SOCKET_IFNAME=enp50s0
export NCCL_SOCKET_IFNAME=enp50s0
```

### Solution 2: Enable NCCL Debugging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Then look for errors in output
```

### Solution 3: Use NVLink/PCIe

For Hopper GPU with NVLink:
```bash
export NCCL_P2P_LEVEL=NVL
```

For systems without NVLink:
```bash
export NCCL_P2P_LEVEL=PHB
```

### Solution 4: Increase Timeouts

```bash
export NCCL_TIMEOUT_MS=600000  # 10 minutes
```

---

## 🐌 Issue: Training is Very Slow (But Not OOM)

**Symptom:** Training runs but GPUs are <50% utilized

### Diagnosis

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check for bottlenecks
nvidia-smi dmon -s pucvmet
```

### Solution 1: Increase Batch Size

```yaml
per_device_train_batch_size: 8  # Increase if memory allows
```

### Solution 2: Reduce Data Loading Overhead

```yaml
dataloader_num_workers: 8  # Increase for faster data loading
preprocessing_num_workers: 32
```

### Solution 3: Use Faster Dataset Format

If using JSON/JSONL, convert to Arrow format:
```bash
# Preprocess dataset
llamafactory-cli train /workspace/configs/llama3_preprocess.yaml
```

Then use tokenized dataset:
```yaml
tokenized_path: /workspace/checkpoints/tokenized_data
```

### Solution 4: Disable Logging

```yaml
logging_steps: 100  # Reduce from 10
save_steps: 10000  # Reduce checkpoint frequency
report_to: none  # Disable external logging
```

---

## 🔥 Issue: NaN or Inf in Loss

**Symptom:**
```
Step 50: loss=nan
RuntimeError: Loss is NaN
```

### Solution 1: Reduce Learning Rate

```yaml
learning_rate: 1.0e-4  # Reduce from 2.0e-4
```

### Solution 2: Add Gradient Clipping

```yaml
gradient_clipping: 1.0  # Clip at 1.0
```

### Solution 3: Increase Warmup

```yaml
warmup_steps: 500  # Increase from 100
warmup_ratio: 0.1
```

### Solution 4: Check FP8 Scaling

Enable debug logging:
```bash
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
```

Look for amax values in logs. If consistently 0 or inf, there's an issue with FP8 scaling.

---

## 🐳 Issue: Docker Container Fails to Start

**Symptom:**
```
docker: Error response from daemon: could not select device driver
```

### Solution: Install/Configure NVIDIA Container Toolkit

```bash
# Remove old installations
sudo apt-get remove nvidia-docker nvidia-docker2

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## 📞 Getting Additional Help

If none of these solutions work:

### 1. Collect Diagnostic Information

```bash
# Run verification
python /workspace/scripts/verify_fp8.py > diagnostic.txt 2>&1

# Collect logs
tar -czf logs.tar.gz /workspace/checkpoints/*/training.log

# System info
nvidia-smi > gpu_info.txt
docker version > docker_info.txt
```

### 2. Check Configurations

```bash
# Show all configs
cat /workspace/configs/*.yaml > configs.txt
cat /workspace/configs/*.json >> configs.txt
```

### 3. Open an Issue

Include:
- `diagnostic.txt`
- `logs.tar.gz`
- `configs.txt`
- GPU model and driver version
- Clear description of the issue

---

## 🔧 Advanced Debugging

### Enable Full Debug Logging

```bash
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
```

### Profile Training

```bash
# Use NVIDIA Nsight Systems
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
  --output=profile.qdrep \
  --force-overwrite=true \
  python src/train.py /workspace/configs/llama3_fp8_deepspeed_sft.yaml

# View profile
nsight-sys profile.qdrep
```

### Memory Profiling

```python
# Add to training script
import torch
torch.cuda.memory._record_memory_history(max_entries=100000)

# After OOM, save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

---

## ✅ Verification After Fixes

After applying any fixes, run:

```bash
# 1. Verify environment
python /workspace/scripts/verify_fp8.py

# 2. Run short test
# Edit YAML to add: max_steps: 10
bash /workspace/scripts/train_fp8.sh

# 3. Check logs for correct FP8 usage
grep -E "FP8|fp8|Transformer Engine" \
  /workspace/checkpoints/llama3-fp8-deepspeed/training.log
```

Look for:
- ✅ `FP8 training enabled with te backend`
- ✅ `Accelerate FP8 status - enabled: True`
- ✅ No warnings about FP8 not supported
- ✅ Iteration times faster than BF16
