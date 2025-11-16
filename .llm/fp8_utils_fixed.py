# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Optional

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import ModelArguments

logger = logging.get_logger(__name__)


def create_fp8_kwargs(model_args: "ModelArguments") -> list[Any]:
    """Create FP8RecipeKwargs or AORecipeKwargs for FP8 training with HuggingFace Accelerate.

    Args:
        model_args: Model arguments containing FP8 configuration

    Returns:
        List containing FP8RecipeKwargs (for TE) or AORecipeKwargs (for TorchAO) if FP8 is enabled
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
            return [TERecipeKwargs(
                fp8_format="HYBRID",
                amax_history_len=16,
                amax_compute_algo="max"
            )]
        
        # Use TorchAO backend
        else:
            from accelerate.utils import AORecipeKwargs
            from torchao.float8 import Float8LinearConfig

            # Create Float8LinearConfig for torchao
            config = Float8LinearConfig.from_recipe_name("rowwise")

            if hasattr(config, "enable_amax_init"):
                config.enable_amax_init = True
            if hasattr(config, "enable_pre_and_post_forward"):
                config.enable_pre_and_post_forward = True

            # Module filter to skip problematic layers
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


def get_fp8_mixed_precision(model_args: "ModelArguments") -> Optional[str]:
    """Get the mixed precision setting for Accelerate when using FP8.

    Args:
        model_args: Model arguments containing FP8 configuration

    Returns:
        "fp8" if FP8 is enabled, None otherwise
    """
    return "fp8" if model_args.fp8 else None


def configure_fp8_environment(model_args: "ModelArguments") -> None:
    """Configure FP8 environment for HuggingFace Accelerate.

    Args:
        model_args: Model arguments containing FP8 configuration
    """
    import os

    if not model_args.fp8:
        return

    # Set mixed precision to fp8 for HuggingFace Accelerate
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
    logger.info_rank0("Set ACCELERATE_MIXED_PRECISION=fp8")

    # Configure FP8 backend
    backend = getattr(model_args, "fp8_backend", "auto")
    if backend != "auto":
        os.environ["FP8_BACKEND"] = backend
        logger.info_rank0(f"Set FP8_BACKEND={backend}")

    # Create FP8 recipe kwargs
    fp8_kwargs = create_fp8_kwargs(model_args)
    logger.info_rank0(f"FP8 recipe kwargs created: {len(fp8_kwargs)} items")

    logger.info_rank0("FP8 environment configured - all FP8 training handled by HuggingFace Accelerate")


def verify_fp8_status(accelerator, model_args: "ModelArguments") -> None:
    """Verify that FP8 training is actually working after model preparation.

    Args:
        accelerator: The HuggingFace Accelerator instance
        model_args: Model arguments containing FP8 configuration
    """
    if not model_args.fp8:
        return

    fp8_enabled = getattr(accelerator, "fp8_enabled", False)
    fp8_backend_type = getattr(accelerator, "fp8_backend", "UNKNOWN")

    backend = getattr(model_args, "fp8_backend", "auto")
    logger.info_rank0(f"FP8 training enabled with {backend} backend.")
    logger.info_rank0(f"Accelerate FP8 status - enabled: {fp8_enabled}, backend: {fp8_backend_type}")

    if not fp8_enabled:
        logger.warning_rank0("WARNING: FP8 was requested but Accelerate shows fp8_enabled=False.")
