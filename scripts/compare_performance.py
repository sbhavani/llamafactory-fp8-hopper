#!/usr/bin/env python3
"""
Compare FP8 vs BF16 Performance
Parses training logs and generates a performance comparison report.

Usage:
  python compare_performance.py [bf16_dir] [fp8_dir]
  
If no directories specified, will auto-detect the most recent matching pair.
"""

import os
import json
from pathlib import Path
import glob
import sys

def find_checkpoint_dirs(checkpoint_base="/workspace/checkpoints", model_size=None):
    """Find BF16 and FP8 checkpoint directories, preferring most recent."""
    all_dirs = [d for d in glob.glob(os.path.join(checkpoint_base, "*")) if os.path.isdir(d)]
    
    # Sort by modification time (most recent first)
    all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # If model_size specified, filter
    if model_size:
        all_dirs = [d for d in all_dirs if model_size in os.path.basename(d)]
    
    # Find most recent FP8 and BF16 dirs
    bf16_dir = None
    fp8_dir = None
    
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
    """Parse trainer_state.json to extract performance metrics."""
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None
    
    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    
    if not os.path.exists(trainer_state_file):
        return None
    
    try:
        with open(trainer_state_file, 'r') as f:
            state = json.load(f)
        
        # Extract metrics from log_history
        log_history = state.get('log_history', [])
        
        if not log_history:
            return None
        
        # Find the final training metrics
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
            time_per_step = runtime / total_steps
            
            return {
                "total_steps": total_steps,
                "train_runtime": runtime,
                "time_per_step": time_per_step,
                "train_samples_per_second": train_metrics.get('train_samples_per_second', 0),
                "train_loss": train_metrics.get('train_loss', 0),
                "epoch": train_metrics.get('epoch', 0)
            }
    
    except Exception as e:
        print(f"Error parsing {trainer_state_file}: {e}")
    
    return None

def generate_report(bf16_metrics, fp8_metrics, bf16_dir, fp8_dir):
    """Generate performance comparison report."""
    print("\n" + "=" * 70)
    print("FP8 vs BF16 Performance Comparison")
    print("=" * 70)
    
    if bf16_dir:
        print(f"\n📁 BF16 Directory: {bf16_dir}")
    if fp8_dir:
        print(f"📁 FP8 Directory:  {fp8_dir}")
    
    if not bf16_metrics:
        print("\n⚠️  BF16 metrics not available")
        print("   Make sure BF16 training completed successfully")
        bf16_time = None
    else:
        bf16_time = bf16_metrics['time_per_step']
        print(f"\n📊 BF16 Baseline:")
        print(f"   Time per step:    {bf16_time:.3f} s/step")
        print(f"   Total steps:      {bf16_metrics['total_steps']}")
        print(f"   Total runtime:    {bf16_metrics['train_runtime']:.2f} s")
        print(f"   Samples/second:   {bf16_metrics['train_samples_per_second']:.2f}")
        print(f"   Final loss:       {bf16_metrics['train_loss']:.4f}")
    
    if not fp8_metrics:
        print("\n⚠️  FP8 metrics not available")
        print("   Make sure FP8 training completed successfully")
        fp8_time = None
    else:
        fp8_time = fp8_metrics['time_per_step']
        print(f"\n📊 FP8 Training:")
        print(f"   Time per step:    {fp8_time:.3f} s/step")
        print(f"   Total steps:      {fp8_metrics['total_steps']}")
        print(f"   Total runtime:    {fp8_metrics['train_runtime']:.2f} s")
        print(f"   Samples/second:   {fp8_metrics['train_samples_per_second']:.2f}")
        print(f"   Final loss:       {fp8_metrics['train_loss']:.4f}")
    
    if bf16_time and fp8_time:
        speedup = bf16_time / fp8_time
        improvement = ((bf16_time - fp8_time) / bf16_time) * 100
        
        print(f"\n🚀 Performance Comparison:")
        print(f"   Speedup:          {speedup:.2f}x")
        print(f"   Time difference:  {bf16_time - fp8_time:.3f} s/step")
        
        if speedup > 1.0:
            print(f"   ✅ FP8 is {improvement:.1f}% FASTER than BF16")
            
            if speedup >= 1.25:
                print(f"   🎯 EXCELLENT: Achieving good FP8 performance!")
            elif speedup >= 1.15:
                print(f"   ✓  GOOD: Reasonable FP8 speedup")
            else:
                print(f"   ⚠️  MODERATE: FP8 speedup is modest")
        else:
            slowdown = ((fp8_time - bf16_time) / bf16_time) * 100
            print(f"   ❌ FP8 is {slowdown:.1f}% SLOWER than BF16")
            print(f"\n   ⚠️  ISSUE DETECTED: FP8 should be faster!")
            print(f"   Possible causes:")
            print(f"   - DeepSpeed config has FP32 communication")
            print(f"   - FP8 backend not properly enabled (check for 'te' backend)")
            print(f"   - Missing NVTE environment variables")
            print(f"   - Communication-compute overlap not enabled")
        
        print(f"\n📈 Expected Results on Hopper GPUs (H100):")
        print(f"   Target speedup:        1.3-1.5x")
        print(f"   Expected FP8 time:     {bf16_time / 1.3:.3f} - {bf16_time / 1.5:.3f} s/step")
        
        if bf16_metrics and fp8_metrics:
            throughput_improvement = ((fp8_metrics['train_samples_per_second'] - 
                                      bf16_metrics['train_samples_per_second']) / 
                                     bf16_metrics['train_samples_per_second']) * 100
            print(f"   Throughput gain:       {throughput_improvement:.1f}%")
    
    print("\n" + "=" * 70)

def main():
    """Main entry point."""
    checkpoint_base = "/workspace/checkpoints"
    
    # Check for command line arguments
    if len(sys.argv) >= 3:
        # Explicit directories provided
        bf16_dir = sys.argv[1] if sys.argv[1] != 'auto' else None
        fp8_dir = sys.argv[2] if sys.argv[2] != 'auto' else None
        print(f"Using specified directories:")
        print(f"  BF16: {bf16_dir}")
        print(f"  FP8:  {fp8_dir}")
    elif len(sys.argv) == 2:
        # Model size hint provided (e.g., "7b")
        model_size = sys.argv[1]
        print(f"Searching for most recent {model_size} checkpoints...")
        bf16_dir, fp8_dir = find_checkpoint_dirs(checkpoint_base, model_size)
    else:
        # Auto-detect most recent
        print("Searching for most recent training checkpoints...")
        bf16_dir, fp8_dir = find_checkpoint_dirs(checkpoint_base)
    
    if not bf16_dir:
        print(f"❌ Could not find BF16 checkpoint directory in {checkpoint_base}")
    if not fp8_dir:
        print(f"❌ Could not find FP8 checkpoint directory in {checkpoint_base}")
    
    if not bf16_dir and not fp8_dir:
        print("\nPlease run training first:")
        print("  bash /workspace/scripts/train_bf16.sh")
        print("  bash /workspace/scripts/train_fp8.sh")
        print("\nOr specify directories:")
        print("  python compare_performance.py <bf16_dir> <fp8_dir>")
        print("  python compare_performance.py 7b  # Find most recent 7b checkpoints")
        return
    
    bf16_metrics = parse_trainer_state(bf16_dir)
    fp8_metrics = parse_trainer_state(fp8_dir)
    
    generate_report(bf16_metrics, fp8_metrics, bf16_dir, fp8_dir)

if __name__ == "__main__":
    main()
