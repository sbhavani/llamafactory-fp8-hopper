---
name: Bug Report
about: Report an issue with FP8 training
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
<!-- A clear description of what the bug is -->

## Environment
**GPU Model**: <!-- e.g., GH200, H100 -->
**CUDA Version**: 
**Driver Version**: 
**Docker Image**: <!-- e.g., nvcr.io/nvidia/pytorch:25.10-py3 -->

## Verification Output
```bash
# Paste output from: 
python /workspace/scripts/verify_fp8.py
```

## Training Configuration
<!-- Which config file are you using? -->
- [ ] llama3_fp8_deepspeed_sft.yaml
- [ ] llama3_bf16_baseline_sft.yaml
- [ ] Custom (please attach)

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
<!-- What you expected to happen -->

## Actual Behavior
<!-- What actually happened -->

## Logs
```
<!-- Paste relevant training logs -->
```

## Performance Metrics
**BF16 Iteration Time**: 
**FP8 Iteration Time**: 
**Expected Speedup**: 1.3-1.5x
**Actual Result**: 

## Additional Context
<!-- Any other context about the problem -->
