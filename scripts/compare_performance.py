#!/usr/bin/env python3
"""Compare FP8 vs BF16 training performance from checkpoint logs."""

import os
import json
import sys
import glob


def find_checkpoint_dirs(checkpoint_base="/workspace/checkpoints", model_size=None):
    """Find BF16 and FP8 checkpoint directories."""
    all_dirs = [d for d in glob.glob(os.path.join(checkpoint_base, "*")) if os.path.isdir(d)]
    all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if model_size:
        all_dirs = [d for d in all_dirs if model_size in os.path.basename(d)]

    bf16_dir = fp8_dir = None
    for d in all_dirs:
        basename = os.path.basename(d)
        if not bf16_dir and ('bf16' in basename or 'baseline' in basename):
            bf16_dir = d
        if not fp8_dir and 'fp8' in basename:
            fp8_dir = d
        if bf16_dir and fp8_dir:
            break

    return bf16_dir, fp8_dir


def parse_trainer_state(checkpoint_dir):
    """Parse trainer_state.json for metrics."""
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None

    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_file):
        return None

    try:
        with open(trainer_state_file, 'r') as f:
            state = json.load(f)

        log_history = state.get('log_history', [])
        if not log_history:
            return None

        train_metrics = None
        for entry in reversed(log_history):
            if 'train_runtime' in entry:
                train_metrics = entry
                break

        if not train_metrics:
            return None

        total_steps = state.get('global_step', 0)
        runtime = train_metrics.get('train_runtime', 0)

        if total_steps > 0 and runtime > 0:
            return {
                "total_steps": total_steps,
                "train_runtime": runtime,
                "time_per_step": runtime / total_steps,
                "train_samples_per_second": train_metrics.get('train_samples_per_second', 0),
                "train_loss": train_metrics.get('train_loss', 0),
            }
    except Exception as e:
        print(f"Error parsing {trainer_state_file}: {e}")

    return None


def generate_report(bf16_metrics, fp8_metrics, bf16_dir, fp8_dir):
    """Print performance comparison."""
    print("\n" + "=" * 60)
    print("FP8 vs BF16 Performance Comparison")
    print("=" * 60)

    if bf16_dir:
        print(f"\nBF16: {bf16_dir}")
    if fp8_dir:
        print(f"FP8:  {fp8_dir}")

    bf16_time = fp8_time = None

    if bf16_metrics:
        bf16_time = bf16_metrics['time_per_step']
        print(f"\nBF16: {bf16_time:.3f} s/step, {bf16_metrics['train_samples_per_second']:.2f} samples/s, loss={bf16_metrics['train_loss']:.4f}")
    else:
        print("\nBF16: not available")

    if fp8_metrics:
        fp8_time = fp8_metrics['time_per_step']
        print(f"FP8:  {fp8_time:.3f} s/step, {fp8_metrics['train_samples_per_second']:.2f} samples/s, loss={fp8_metrics['train_loss']:.4f}")
    else:
        print("FP8:  not available")

    if bf16_time and fp8_time:
        speedup = bf16_time / fp8_time
        print(f"\nSpeedup: {speedup:.2f}x {'(FP8 faster)' if speedup > 1 else '(BF16 faster)'}")

    print("=" * 60)


def main():
    checkpoint_base = "/workspace/checkpoints"

    if len(sys.argv) >= 3:
        bf16_dir = sys.argv[1] if sys.argv[1] != 'auto' else None
        fp8_dir = sys.argv[2] if sys.argv[2] != 'auto' else None
    elif len(sys.argv) == 2:
        bf16_dir, fp8_dir = find_checkpoint_dirs(checkpoint_base, sys.argv[1])
    else:
        bf16_dir, fp8_dir = find_checkpoint_dirs(checkpoint_base)

    if not bf16_dir and not fp8_dir:
        print("No checkpoints found. Run training first or specify directories.")
        return

    generate_report(
        parse_trainer_state(bf16_dir),
        parse_trainer_state(fp8_dir),
        bf16_dir, fp8_dir
    )


if __name__ == "__main__":
    main()
