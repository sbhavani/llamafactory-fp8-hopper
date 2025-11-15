#!/usr/bin/env python3
"""
Compare FP8 vs BF16 Performance
Parses training logs and generates a performance comparison report.
"""

import os
import re
import json
from pathlib import Path

def parse_training_log(log_file):
    """Parse training log to extract performance metrics."""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract iteration time
    # Look for patterns like "8.15s/it" or "16.29s/it"
    time_pattern = r'(\d+\.?\d*)s/it'
    times = re.findall(time_pattern, content)
    
    if not times:
        return None
    
    # Convert to float and calculate average
    times = [float(t) for t in times]
    avg_time = sum(times) / len(times)
    
    return {
        "avg_iteration_time": avg_time,
        "samples": len(times),
        "min_time": min(times),
        "max_time": max(times)
    }

def generate_report(bf16_metrics, fp8_metrics):
    """Generate performance comparison report."""
    print("\n" + "=" * 70)
    print("FP8 vs BF16 Performance Comparison")
    print("=" * 70)
    
    if not bf16_metrics:
        print("⚠️  BF16 metrics not available")
        bf16_time = None
    else:
        bf16_time = bf16_metrics['avg_iteration_time']
        print(f"\n📊 BF16 Baseline:")
        print(f"   Average iteration time: {bf16_time:.2f} s/it")
        print(f"   Samples: {bf16_metrics['samples']}")
        print(f"   Range: {bf16_metrics['min_time']:.2f} - {bf16_metrics['max_time']:.2f} s/it")
    
    if not fp8_metrics:
        print("\n⚠️  FP8 metrics not available")
        fp8_time = None
    else:
        fp8_time = fp8_metrics['avg_iteration_time']
        print(f"\n📊 FP8 Training:")
        print(f"   Average iteration time: {fp8_time:.2f} s/it")
        print(f"   Samples: {fp8_metrics['samples']}")
        print(f"   Range: {fp8_metrics['min_time']:.2f} - {fp8_metrics['max_time']:.2f} s/it")
    
    if bf16_time and fp8_time:
        speedup = bf16_time / fp8_time
        improvement = ((bf16_time - fp8_time) / bf16_time) * 100
        
        print(f"\n🚀 Performance Comparison:")
        print(f"   Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"   ✅ FP8 is {improvement:.1f}% FASTER than BF16")
        else:
            slowdown = ((fp8_time - bf16_time) / bf16_time) * 100
            print(f"   ❌ FP8 is {slowdown:.1f}% SLOWER than BF16")
            print(f"\n   ⚠️  ISSUE DETECTED: FP8 should be faster!")
            print(f"   Possible causes:")
            print(f"   - DeepSpeed config has FP32 communication")
            print(f"   - FP8 backend not properly enabled (check for 'te' backend)")
            print(f"   - Missing NVTE environment variables")
            print(f"   - Communication-compute overlap not enabled")
        
        print(f"\n📈 Expected Results on Hopper GPUs:")
        print(f"   Target speedup: ~1.3-1.5x")
        print(f"   Expected FP8 time: ~5.4-6.3 s/it (from {bf16_time:.2f} s/it baseline)")
    
    print("\n" + "=" * 70)

def main():
    """Main entry point."""
    # Look for training logs
    checkpoint_dir = Path("/workspace/checkpoints")
    
    bf16_log = checkpoint_dir / "llama3-bf16-baseline" / "trainer_log.jsonl"
    fp8_log = checkpoint_dir / "llama3-fp8-deepspeed" / "trainer_log.jsonl"
    
    # Alternative log locations
    if not bf16_log.exists():
        bf16_log = checkpoint_dir / "llama3-bf16-baseline" / "training.log"
    if not fp8_log.exists():
        fp8_log = checkpoint_dir / "llama3-fp8-deepspeed" / "training.log"
    
    print("Parsing training logs...")
    print(f"BF16 log: {bf16_log}")
    print(f"FP8 log: {fp8_log}")
    
    bf16_metrics = parse_training_log(str(bf16_log))
    fp8_metrics = parse_training_log(str(fp8_log))
    
    generate_report(bf16_metrics, fp8_metrics)

if __name__ == "__main__":
    main()
