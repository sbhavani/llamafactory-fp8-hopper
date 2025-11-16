# Remote Server Setup (No Docker)

Guide for setting up FP8 training on a remote server without Docker.

## Prerequisites

- NVIDIA GPU with FP8 support (H100, H200, B200, or 4090)
- CUDA 12.1+ installed
- Python 3.10+ installed
- Git installed

## One-Time Setup

### 1. Create Working Directory

```bash
# SSH into your remote server
ssh user@your-server

# Create workspace
mkdir -p ~/llamafactory-fp8
cd ~/llamafactory-fp8
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers datasets accelerate
pip install deepspeed wandb
pip install transformer-engine[pytorch]
pip install torchao

# Install flash-attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

### 3. Clone LLaMA-Factory Fork

```bash
cd ~/llamafactory-fp8

# Clone the fork with FP8 support
git clone -b fix/accelerate-config-support https://github.com/sbhavani/LLaMA-Factory.git

# Install LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[deepspeed,metrics]"
```

### 4. Clone This Config Repository

```bash
cd ~/llamafactory-fp8

# Clone configs and scripts
git clone https://github.com/sbhavani/llamafactory-fp8-hopper.git configs-repo

# Copy configs and scripts to LLaMA-Factory
cp -r configs-repo/configs .
cp -r configs-repo/scripts .
chmod +x scripts/*.sh
```

### 5. Setup Directories

```bash
cd ~/llamafactory-fp8/LLaMA-Factory

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Create cache directory
mkdir -p ~/.cache/huggingface
```

## Environment Setup (Per-Session)

**Run this every time you SSH in:**

```bash
cd ~/llamafactory-fp8/LLaMA-Factory
source ../venv/bin/activate

# Set environment variables
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets

# Optional: Enable debug logging
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
```

**Or create a setup script:**

```bash
cat > ~/llamafactory-fp8/setup.sh << 'EOF'
#!/bin/bash
cd ~/llamafactory-fp8/LLaMA-Factory
source ../venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
echo "✅ Environment activated"
EOF

chmod +x ~/llamafactory-fp8/setup.sh

# Then just run this each time:
source ~/llamafactory-fp8/setup.sh
```

## Verify Installation

```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check Transformer Engine
python -c "import transformer_engine.pytorch as te; print(f'TE version: {te.__version__}')"

# Check LLaMA-Factory
llamafactory-cli version

# Check Accelerate
accelerate --version
```

## Update Paths in Scripts

The scripts from the Docker setup assume `/workspace` paths. Update them:

```bash
cd ~/llamafactory-fp8/LLaMA-Factory

# Update FP8 training script
cat > scripts/train_fp8.sh << 'EOF'
#!/bin/bash
set -e

CONFIG_FILE=${1:-"configs/qwen_7b_fp8_full_benchmark.yaml"}
BASE_DIR="$HOME/llamafactory-fp8/LLaMA-Factory"

echo "=================================================="
echo "FP8 Training with LLaMA-Factory"
echo "=================================================="

accelerate launch \
  --config_file $BASE_DIR/configs/accelerate_fp8.yaml \
  $(which llamafactory-cli) train \
  $BASE_DIR/$CONFIG_FILE 2>&1 | tee logs/fp8_training.log

echo "Training complete! Logs: logs/fp8_training.log"
EOF

chmod +x scripts/train_fp8.sh

# Update BF16 training script
cat > scripts/train_bf16.sh << 'EOF'
#!/bin/bash
set -e

CONFIG_FILE=${1:-"configs/qwen_7b_bf16_full_benchmark.yaml"}
BASE_DIR="$HOME/llamafactory-fp8/LLaMA-Factory"

echo "=================================================="
echo "BF16 Training with LLaMA-Factory"
echo "=================================================="

accelerate launch \
  --config_file $BASE_DIR/configs/accelerate_bf16.yaml \
  $(which llamafactory-cli) train \
  $BASE_DIR/$CONFIG_FILE 2>&1 | tee logs/bf16_training.log

echo "Training complete! Logs: logs/bf16_training.log"
EOF

chmod +x scripts/train_bf16.sh
```

## Update Config File Paths

Update the configs to use correct paths:

```bash
cd ~/llamafactory-fp8/LLaMA-Factory

# Example: Update Qwen 7B FP8 config
cat > configs/qwen_7b_fp8_quick.yaml << 'EOF'
### Model
model_name_or_path: Qwen/Qwen2.5-7B

### Method
stage: sft
do_train: true
finetuning_type: full
fp8: true
fp8_backend: te

### Dataset
dataset: alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 100
overwrite_cache: true
preprocessing_num_workers: 16

### Output
output_dir: checkpoints/qwen-7b-fp8-quick
logging_steps: 5
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-05
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
EOF

# Create corresponding BF16 config
cat > configs/qwen_7b_bf16_quick.yaml << 'EOF'
### Model
model_name_or_path: Qwen/Qwen2.5-7B

### Method
stage: sft
do_train: true
finetuning_type: full

### Dataset
dataset: alpaca_en_demo
template: qwen
cutoff_len: 1024
max_samples: 100
overwrite_cache: true
preprocessing_num_workers: 16

### Output
output_dir: checkpoints/qwen-7b-bf16-quick
logging_steps: 5
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-05
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
EOF
```

## Run Training

### Quick Test (100 samples)

```bash
cd ~/llamafactory-fp8/LLaMA-Factory
source ../venv/bin/activate
source ~/llamafactory-fp8/setup.sh

# FP8 training
bash scripts/train_fp8.sh configs/qwen_7b_fp8_quick.yaml

# BF16 baseline
bash scripts/train_bf16.sh configs/qwen_7b_bf16_quick.yaml
```

### Full Training

```bash
# Create your own config
cat > configs/my_training.yaml << 'EOF'
model_name_or_path: Qwen/Qwen2.5-7B
stage: sft
do_train: true
finetuning_type: full
fp8: true
fp8_backend: te
dataset: your_dataset
template: qwen
cutoff_len: 2048
learning_rate: 5.0e-05
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
bf16: true
output_dir: checkpoints/my-training-fp8
EOF

# Run training
bash scripts/train_fp8.sh configs/my_training.yaml
```

## Monitor Training

### In Another Terminal

```bash
# SSH into server again
ssh user@your-server

# Watch GPU usage
watch -n 1 nvidia-smi

# Or more detailed
nvidia-smi dmon -s u
```

### Check Logs

```bash
# Follow training log
tail -f ~/llamafactory-fp8/LLaMA-Factory/logs/fp8_training.log

# Check for FP8 activation
grep -i "accelerate already configured\|fp8 training enabled" logs/fp8_training.log

# Check training progress
grep -i "loss\|epoch" logs/fp8_training.log
```

## Background Training with Screen/Tmux

### Using Screen

```bash
# Start screen session
screen -S fp8-training

# Activate environment
source ~/llamafactory-fp8/setup.sh

# Run training
cd ~/llamafactory-fp8/LLaMA-Factory
bash scripts/train_fp8.sh configs/qwen_7b_fp8_quick.yaml

# Detach: Press Ctrl+A then D

# Later, reattach
screen -r fp8-training
```

### Using Tmux

```bash
# Start tmux session
tmux new -s fp8-training

# Activate environment
source ~/llamafactory-fp8/setup.sh

# Run training
cd ~/llamafactory-fp8/LLaMA-Factory
bash scripts/train_fp8.sh configs/qwen_7b_fp8_quick.yaml

# Detach: Press Ctrl+B then D

# Later, reattach
tmux attach -t fp8-training
```

### Using Nohup

```bash
cd ~/llamafactory-fp8/LLaMA-Factory
source ../venv/bin/activate

# Run in background
nohup bash scripts/train_fp8.sh configs/qwen_7b_fp8_quick.yaml > logs/nohup.log 2>&1 &

# Check status
jobs
tail -f logs/nohup.log

# Get PID
ps aux | grep train
```

## Compare Performance

```bash
cd ~/llamafactory-fp8/LLaMA-Factory

# Create comparison script
cat > scripts/compare_results.py << 'EOF'
import json
import os

def load_trainer_state(checkpoint_dir):
    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return None

fp8_state = load_trainer_state("checkpoints/qwen-7b-fp8-quick")
bf16_state = load_trainer_state("checkpoints/qwen-7b-bf16-quick")

if fp8_state and bf16_state:
    fp8_time = fp8_state["train_runtime"]
    bf16_time = bf16_state["train_runtime"]
    speedup = bf16_time / fp8_time
    
    print(f"FP8 time:  {fp8_time:.2f}s")
    print(f"BF16 time: {bf16_time:.2f}s")
    print(f"Speedup:   {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
else:
    print("Could not load training states")
EOF

python scripts/compare_results.py
```

## Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Reduce batch size in your config
per_device_train_batch_size: 2  # was 4
gradient_accumulation_steps: 8  # was 4
```

### Issue: Permission Denied

```bash
# Fix permissions
chmod +x ~/llamafactory-fp8/LLaMA-Factory/scripts/*.sh
```

### Issue: Module Not Found

```bash
# Make sure virtual environment is activated
source ~/llamafactory-fp8/venv/bin/activate

# Reinstall dependencies
pip install -e ".[deepspeed,metrics]"
```

### Issue: Slow Model Download

```bash
# Use HF_HUB_DOWNLOAD_RESUME to resume interrupted downloads
export HF_HUB_DOWNLOAD_RESUME=1

# Or download model first
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B')"
```

## Directory Structure

After setup, your structure should look like:

```
~/llamafactory-fp8/
├── venv/                          # Python virtual environment
├── setup.sh                       # Environment setup script
├── configs-repo/                  # Cloned config repository
└── LLaMA-Factory/                 # LLaMA-Factory installation
    ├── configs/
    │   ├── accelerate_fp8.yaml    # Accelerate FP8 config
    │   ├── accelerate_bf16.yaml   # Accelerate BF16 config
    │   ├── qwen_7b_fp8_quick.yaml # Training config
    │   └── qwen_7b_bf16_quick.yaml
    ├── scripts/
    │   ├── train_fp8.sh           # FP8 training script
    │   ├── train_bf16.sh          # BF16 training script
    │   └── compare_results.py     # Performance comparison
    ├── checkpoints/               # Training outputs
    │   ├── qwen-7b-fp8-quick/
    │   └── qwen-7b-bf16-quick/
    ├── logs/                      # Training logs
    │   ├── fp8_training.log
    │   └── bf16_training.log
    └── data/                      # LLaMA-Factory datasets
```

## Clean Installation Script (All-in-One)

Save this as `~/install_fp8.sh`:

```bash
#!/bin/bash
set -e

echo "Installing LLaMA-Factory with FP8 support..."

# Create workspace
mkdir -p ~/llamafactory-fp8
cd ~/llamafactory-fp8

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate deepspeed wandb
pip install transformer-engine[pytorch] torchao

# Clone and install LLaMA-Factory
git clone -b fix/accelerate-config-support https://github.com/sbhavani/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[deepspeed,metrics]"

# Clone configs
cd ~/llamafactory-fp8
git clone https://github.com/sbhavani/llamafactory-fp8-hopper.git configs-repo
cd LLaMA-Factory
cp -r ../configs-repo/configs .
cp -r ../configs-repo/scripts .
chmod +x scripts/*.sh

# Create directories
mkdir -p checkpoints logs

# Create setup script
cat > ~/llamafactory-fp8/setup.sh << 'EOFSETUP'
#!/bin/bash
cd ~/llamafactory-fp8/LLaMA-Factory
source ../venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
export NVTE_DEBUG=1
echo "✅ Environment activated"
EOFSETUP

chmod +x ~/llamafactory-fp8/setup.sh

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use, run:"
echo "  source ~/llamafactory-fp8/setup.sh"
echo "  cd ~/llamafactory-fp8/LLaMA-Factory"
echo "  bash scripts/train_fp8.sh configs/qwen_7b_fp8_quick.yaml"
```

Then run:

```bash
bash ~/install_fp8.sh
```

## Quick Start Summary

```bash
# 1. Install (one-time)
bash ~/install_fp8.sh

# 2. Setup environment (per session)
source ~/llamafactory-fp8/setup.sh

# 3. Run training
cd ~/llamafactory-fp8/LLaMA-Factory
bash scripts/train_fp8.sh configs/qwen_7b_fp8_quick.yaml
```

That's it! 🚀
