#!/bin/bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 \
  --host 0.0.0.0 \
  --port 8129 \
  --gpu-memory-utilization 0.3 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype half \
  --quantization gptq