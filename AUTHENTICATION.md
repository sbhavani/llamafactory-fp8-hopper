# HuggingFace Authentication Guide

## Issue: Gated Models

Some models like Meta Llama require authentication to access. If you see this error:

```
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B.
```

## Solution Options

### Option 1: Use Public Models (Recommended for Testing)

Use the provided Qwen configs that don't require authentication:

```bash
# FP8 training with Qwen
bash /workspace/scripts/train_fp8.sh /workspace/configs/qwen_fp8_deepspeed_sft.yaml

# BF16 baseline with Qwen
bash /workspace/scripts/train_bf16.sh /workspace/configs/qwen_bf16_baseline_sft.yaml
```

**Public Models Available:**
- `Qwen/Qwen2.5-7B` - Similar size to Llama 3.1 8B, no auth required
- `mistralai/Mistral-7B-v0.1` - Public Mistral model
- `microsoft/phi-2` - Smaller model for quick testing

### Option 2: Authenticate with HuggingFace

If you need to use Llama or other gated models:

#### Step 1: Get HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new access token (read permission is sufficient)
3. Copy the token

#### Step 2: Request Model Access

For Meta Llama models:
1. Visit https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
2. Click "Request Access"
3. Accept the license agreement
4. Wait for approval (usually instant)

#### Step 3: Login in Docker Container

Inside your Docker container, run:

```bash
# Login with your token
huggingface-cli login

# Or set the token as environment variable
export HF_TOKEN="your_token_here"
```

#### Step 4: Run Training

Now you can use the Llama configs:

```bash
# FP8 training with Llama
bash /workspace/scripts/train_fp8.sh

# BF16 baseline with Llama  
bash /workspace/scripts/train_bf16.sh
```

### Option 3: Use Local Model

If you have a model downloaded locally:

1. Mount your model directory in Docker:
```bash
docker run -it --rm \
  --gpus all \
  --ipc=host \
  -v /path/to/your/model:/workspace/models/llama \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/configs:/workspace/configs \
  llamafactory-fp8:latest
```

2. Update the YAML config:
```yaml
model_name_or_path: /workspace/models/llama
```

## Recommended Testing Workflow

1. **Start with Qwen** (no auth) to verify FP8 setup works
2. **Compare FP8 vs BF16** performance with Qwen
3. **Switch to Llama** after authenticating, if needed

## Quick Test Commands

```bash
# Inside Docker container

# Test with Qwen (no auth needed)
bash /workspace/scripts/train_fp8.sh /workspace/configs/qwen_fp8_deepspeed_sft.yaml

# If using Llama (requires auth)
huggingface-cli login
bash /workspace/scripts/train_fp8.sh /workspace/configs/llama3_fp8_deepspeed_sft.yaml
```

## Environment Variable for Token

You can also set the token in `docker-compose.yml`:

```yaml
environment:
  - HF_TOKEN=your_token_here
```

Or in the Docker run command:

```bash
docker run -e HF_TOKEN="your_token_here" ...
```
