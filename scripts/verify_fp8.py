#!/usr/bin/env python3
"""Verify FP8 environment setup."""

import sys
import os
import torch


def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA: not available")
        return False

    print(f"CUDA: {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        fp8_ok = props.major > 8 or (props.major == 8 and props.minor >= 9)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.0f}GB, FP8={'yes' if fp8_ok else 'no'}")
    return True


def check_transformer_engine():
    try:
        import transformer_engine
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format

        DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        te.Linear(128, 128).cuda()
        print(f"Transformer Engine: {transformer_engine.__version__}")
        return True
    except Exception as e:
        print(f"Transformer Engine: failed ({e})")
        return False


def check_accelerate():
    try:
        from accelerate import __version__
        major, minor = map(int, __version__.split('.')[:2])
        fp8_ok = major > 1 or (major == 1 and minor >= 8)
        print(f"Accelerate: {__version__}, FP8={'yes' if fp8_ok else 'no'}")
        return True
    except Exception as e:
        print(f"Accelerate: failed ({e})")
        return False


def check_llamafactory():
    try:
        import llamafactory
        print("LLaMA-Factory: ok")
        return True
    except Exception as e:
        print(f"LLaMA-Factory: failed ({e})")
        return False


def main():
    print("FP8 Environment Check\n" + "=" * 40)
    results = [check_cuda(), check_transformer_engine(), check_accelerate(), check_llamafactory()]
    print("=" * 40)
    print("Ready for FP8" if all(results) else "Some checks failed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
