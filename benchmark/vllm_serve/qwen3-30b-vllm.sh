export NCCL_P2P_LEVEL=DIS
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8129 \
  --tensor-parallel-size 1 \
  --max-model-len 20000 \
  # --disable-structured-output