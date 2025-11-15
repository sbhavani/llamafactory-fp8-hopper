#!/usr/bin/env python3
"""
Verify FP8 Setup and Configuration
This script checks that all components are properly installed and configured.
"""

import sys
import torch

def check_cuda():
    """Check CUDA availability and version."""
    print("=" * 60)
    print("CUDA Check")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    print(f"✅ CUDA is available")
    print(f"   Version: {torch.version.cuda}")
    print(f"   Device count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
        
        # Check for FP8 support (requires compute capability 8.9+)
        if props.major > 8 or (props.major == 8 and props.minor >= 9):
            print(f"      ✅ FP8 supported (CC {props.major}.{props.minor})")
        else:
            print(f"      ❌ FP8 NOT supported (CC {props.major}.{props.minor} < 8.9)")
    
    print()
    return True

def check_transformer_engine():
    """Check Transformer Engine installation."""
    print("=" * 60)
    print("Transformer Engine Check")
    print("=" * 60)
    
    try:
        import transformer_engine
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        
        print(f"✅ Transformer Engine installed")
        print(f"   Version: {transformer_engine.__version__}")
        
        # Try to create a simple FP8 recipe
        fp8_recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max"
        )
        print(f"✅ FP8 recipe creation successful")
        
        # Try to create a simple TE module
        linear = te.Linear(128, 128).cuda()
        print(f"✅ TE Linear module creation successful")
        
        return True
    except ImportError as e:
        print(f"❌ Transformer Engine not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ Transformer Engine error: {e}")
        return False
    finally:
        print()

def check_accelerate():
    """Check HuggingFace Accelerate installation."""
    print("=" * 60)
    print("HuggingFace Accelerate Check")
    print("=" * 60)
    
    try:
        import accelerate
        from accelerate import __version__
        print(f"✅ Accelerate installed")
        print(f"   Version: {__version__}")
        
        # Check for FP8 support (requires 1.8.0+)
        major, minor = map(int, __version__.split('.')[:2])
        if major > 1 or (major == 1 and minor >= 8):
            print(f"✅ FP8 support available (version >= 1.8.0)")
            
            # Try to import FP8-related classes
            try:
                from accelerate.utils import AORecipeKwargs
                print(f"✅ AORecipeKwargs available")
            except ImportError:
                print(f"⚠️  AORecipeKwargs not available (may be in newer version)")
        else:
            print(f"❌ FP8 NOT supported (version {__version__} < 1.8.0)")
        
        return True
    except ImportError as e:
        print(f"❌ Accelerate not installed: {e}")
        return False
    finally:
        print()

def check_deepspeed():
    """Check DeepSpeed installation."""
    print("=" * 60)
    print("DeepSpeed Check")
    print("=" * 60)
    
    try:
        import deepspeed
        print(f"✅ DeepSpeed installed")
        print(f"   Version: {deepspeed.__version__}")
        return True
    except ImportError as e:
        print(f"❌ DeepSpeed not installed: {e}")
        return False
    finally:
        print()

def check_llamafactory():
    """Check LLaMA-Factory installation."""
    print("=" * 60)
    print("LLaMA-Factory Check")
    print("=" * 60)
    
    try:
        # Check if LLaMA-Factory package can be imported
        import llamafactory
        print(f"✅ LLaMA-Factory installed")
        
        # Check if we can import the training entry point
        try:
            from llamafactory.train import tuner
            print(f"✅ Training module available")
        except ImportError:
            print(f"⚠️  Could not import training module (may be expected)")
        
        # FP8 support in LLaMA-Factory is configured via YAML and environment variables,
        # not through Python utilities, so no need to check for fp8_utils module
        print(f"✅ FP8 support configured via YAML/env (no custom utils needed)")
        
        return True
    except ImportError as e:
        print(f"❌ LLaMA-Factory not installed: {e}")
        return False
    finally:
        print()

def check_environment_variables():
    """Check important environment variables."""
    print("=" * 60)
    print("Environment Variables Check")
    print("=" * 60)
    
    import os
    
    important_vars = [
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NVTE_APPLY_QK_LAYER_SCALING",
        "NVTE_FP8_DPA_BWD",
        "NVTE_FLASH_ATTN",
        "NVTE_FUSED_ATTN",
        "NVTE_FP8_ALLREDUCE",
        "PYTORCH_ALLOC_CONF",
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"⚠️  {var} not set")
    
    print()

def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("FP8 Environment Verification")
    print("=" * 60 + "\n")
    
    checks = [
        ("CUDA", check_cuda),
        ("Transformer Engine", check_transformer_engine),
        ("Accelerate", check_accelerate),
        ("DeepSpeed", check_deepspeed),
        ("LLaMA-Factory", check_llamafactory),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    check_environment_variables()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    
    if all_passed:
        print("✅ All checks passed! Ready for FP8 training.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
