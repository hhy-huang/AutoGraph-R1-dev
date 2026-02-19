export HF_HOME=/data/haoyuhuang/model
export VLLM_CACHE_DIR=/data/haoyuhuang/model
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES=6 vllm serve Qwen/Qwen3-Embedding-0.6B \
  --host 0.0.0.0 \
  --port 8128 \
  --gpu-memory-utilization 0.3 \
  --tensor-parallel-size 1 \
  --max-model-len 32768