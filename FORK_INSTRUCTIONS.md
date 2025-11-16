# Fork and Fix LLaMA-Factory for FP8

## Step 1: Fork the Repository

1. Go to https://github.com/hiyouga/LLaMA-Factory
2. Click "Fork" in the top-right
3. Create fork at your GitHub account

## Step 2: Clone Your Fork

```bash
cd /tmp
git clone https://github.com/YOUR_USERNAME/LLaMA-Factory.git
cd LLaMA-Factory

# Create a branch for FP8 fixes
git checkout -b fix/fp8-transformer-engine
```

## Step 3: Apply Fixes

### Fix 1: Pass model_args to trainer

**File**: `src/llamafactory/train/sft/workflow.py`

Around line 82, change:
```python
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    finetuning_args=finetuning_args,
```

To:
```python
trainer = CustomSeq2SeqTrainer(
    model=model,
    model_args=model_args,  # ADD THIS LINE
    args=training_args,
    finetuning_args=finetuning_args,
```

### Fix 2: Use TERecipeKwargs for Transformer Engine

**File**: `src/llamafactory/train/fp8_utils.py`

Replace the `create_fp8_kwargs()` function:

```python
def create_fp8_kwargs(model_args: "ModelArguments") -> list[Any]:
    """Create TERecipeKwargs or AORecipeKwargs for FP8 training with HuggingFace Accelerate.

    Args:
        model_args: Model arguments containing FP8 configuration

    Returns:
        List containing TERecipeKwargs (for TE) or AORecipeKwargs (for TorchAO) if FP8 is enabled
    """
    if not model_args.fp8:
        return []

    backend = getattr(model_args, "fp8_backend", "auto")
    logger.info_rank0(f"Creating FP8 configuration with backend: {backend}")

    try:
        # Use Transformer Engine backend (recommended for H100/GH200)
        if backend == "te":
            from accelerate.utils import TERecipeKwargs
            
            logger.info_rank0("Using Transformer Engine FP8 backend (optimal for Hopper GPUs)")
            
            # TERecipeKwargs will read from environment variables if not set:
            # ACCELERATE_FP8_FORMAT, ACCELERATE_FP8_AMAX_COMPUTE_ALGO, etc.
            return [TERecipeKwargs(
                fp8_format="HYBRID",
                amax_history_len=16,
                amax_compute_algo="max"
            )]
        
        # Use TorchAO backend (existing code stays the same)
        else:
            from accelerate.utils import AORecipeKwargs
            from torchao.float8 import Float8LinearConfig

            config = Float8LinearConfig.from_recipe_name("rowwise")

            if hasattr(config, "enable_amax_init"):
                config.enable_amax_init = True
            if hasattr(config, "enable_pre_and_post_forward"):
                config.enable_pre_and_post_forward = True

            def module_filter_func(module, layer_name):
                skip_layers = ["embed", "lm_head", "output", "classifier"]
                if any(skip_name in layer_name.lower() for skip_name in skip_layers):
                    return False
                if not (hasattr(module, "weight") and len(module.weight.shape) == 2):
                    return False
                weight = module.weight
                in_features, out_features = weight.shape[1], weight.shape[0]
                if in_features % 16 != 0 or out_features % 16 != 0:
                    return False
                return True

            logger.info_rank0("Using TorchAO FP8 backend")
            return [AORecipeKwargs(config=config, module_filter_func=module_filter_func)]

    except Exception as e:
        logger.warning_rank0(f"Failed to create FP8 configuration: {e}")
        return []
```

## Step 4: Commit and Push

```bash
git add src/llamafactory/train/sft/workflow.py src/llamafactory/train/fp8_utils.py
git commit -m "fix(fp8): add Transformer Engine backend support

- Pass model_args to CustomSeq2SeqTrainer to enable FP8 configuration
- Use TERecipeKwargs for Transformer Engine backend instead of deprecated FP8RecipeKwargs
- Properly support fp8_backend='te' for Hopper GPU optimization

Fixes FP8 training which was defaulting to slower TorchAO backend even when
Transformer Engine was requested."

git push origin fix/fp8-transformer-engine
```

## Step 5: Update Dockerfile to Use Your Fork

**File**: `Dockerfile`

Change:
```dockerfile
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
```

To:
```dockerfile
RUN git clone --depth 1 -b fix/fp8-transformer-engine https://github.com/YOUR_USERNAME/LLaMA-Factory.git && \
```

Remove the patch application steps since the fixes are now in the fork.

## Step 6: Rebuild and Test

```bash
docker build -t llamafactory-fp8:latest .
docker run -it --rm --gpus all llamafactory-fp8:latest

# Inside container
bash /workspace/scripts/train_fp8.sh 2>&1 | tee /tmp/fp8_test.log
grep "FP8BackendType" /tmp/fp8_test.log
# Should show: "backend: FP8BackendType.TE"
```

## Step 7: Submit PR to Upstream (Optional)

Once verified working:
1. Go to your fork on GitHub
2. Click "Contribute" → "Open pull request"
3. Title: "fix(fp8): add Transformer Engine backend support for Hopper GPUs"
4. Reference the issues with FP8 performance
