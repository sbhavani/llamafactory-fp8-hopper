#!/usr/bin/env python3
"""
Patch LLaMA-Factory trainer to properly handle FP8 mixed_precision.

The issue: When using `accelerate launch --config_file accelerate_fp8.yaml`,
the Accelerator is created with mixed_precision from the config file.
However, LLaMA-Factory's trainer injects FP8RecipeKwargs without also
setting mixed_precision='fp8', causing a ValueError.

This patch fixes the trainer to force mixed_precision='fp8' when
FP8 kwargs_handlers are being injected.

Usage:
    python patch_trainer_fp8.py [path_to_llamafactory]
    
Example:
    python patch_trainer_fp8.py /workspace/LLaMA-Factory
    python patch_trainer_fp8.py ~/llamafactory-fp8/LLaMA-Factory
"""

import sys
import os

def find_trainer_file(base_path):
    """Find the trainer.py file in LLaMA-Factory."""
    possible_paths = [
        os.path.join(base_path, "src/llamafactory/train/sft/trainer.py"),
        os.path.join(base_path, "llamafactory/train/sft/trainer.py"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def patch_trainer(trainer_file):
    """Patch the trainer to force mixed_precision='fp8' when using FP8 kwargs."""
    
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "Force mixed_precision to fp8" in content:
        print(f"✅ {trainer_file} is already patched!")
        return True
    
    # Pattern 1: Look for the return statement in patched_accelerator_init
    old_pattern = "return original_accelerator_init(self, *args, **accelerator_kwargs)"
    new_code = """# Force mixed_precision to fp8 when using FP8 kwargs
            if 'kwargs_handlers' in accelerator_kwargs:
                accelerator_kwargs['mixed_precision'] = 'fp8'
            return original_accelerator_init(self, *args, **accelerator_kwargs)"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_code)
        with open(trainer_file, 'w') as f:
            f.write(content)
        print(f"✅ Successfully patched {trainer_file}")
        print("   Added: Force mixed_precision='fp8' when FP8 kwargs_handlers present")
        return True
    
    # Pattern 2: Alternative pattern with different indentation
    old_pattern2 = "return original_accelerator_init(self, *args, **accelerator_kwargs)"
    if old_pattern2 in content:
        new_code2 = """# Force mixed_precision to fp8 when using FP8 kwargs
        if 'kwargs_handlers' in accelerator_kwargs:
            accelerator_kwargs['mixed_precision'] = 'fp8'
        return original_accelerator_init(self, *args, **accelerator_kwargs)"""
        content = content.replace(old_pattern2, new_code2)
        with open(trainer_file, 'w') as f:
            f.write(content)
        print(f"✅ Successfully patched {trainer_file}")
        return True
    
    print(f"❌ Could not find the expected code pattern in {trainer_file}")
    print("   Manual patching may be required.")
    print("\n   Look for 'patched_accelerator_init' function and add:")
    print("   accelerator_kwargs['mixed_precision'] = 'fp8'")
    print("   before the return statement.")
    return False

def main():
    # Default paths to try
    default_paths = [
        "/workspace/LLaMA-Factory",
        os.path.expanduser("~/llamafactory-fp8/LLaMA-Factory"),
        os.path.expanduser("~/LLaMA-Factory"),
        ".",
    ]
    
    # Use command line arg if provided
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
        paths_to_try = [base_path]
    else:
        paths_to_try = default_paths
    
    trainer_file = None
    for path in paths_to_try:
        trainer_file = find_trainer_file(path)
        if trainer_file:
            break
    
    if not trainer_file:
        print("❌ Could not find LLaMA-Factory trainer.py")
        print("\nUsage: python patch_trainer_fp8.py [path_to_llamafactory]")
        print("\nTried paths:")
        for p in paths_to_try:
            print(f"  - {p}")
        sys.exit(1)
    
    print(f"Found trainer file: {trainer_file}")
    success = patch_trainer(trainer_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
