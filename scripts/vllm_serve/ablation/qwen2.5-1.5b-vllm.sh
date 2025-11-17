CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8130 \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 1 \
  --max-model-len 16384