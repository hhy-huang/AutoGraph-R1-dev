# Adjust CHECKPOINT_PATH and STEP_NUM as needed
CHECKPOINT_PATH="/data/haoyuhuang/data/AtlasTune/checkpoints/20251211_234743_qwen2.5-3B-autograph-easy-docsize15-textlinkingFalse-loose"
STEP_NUM="350"

CUDA_VISIBLE_DEVICES=7 vllm serve $CHECKPOINT_PATH/global_step_$STEP_NUM/actor/huggingface \
    --host 0.0.0.0 \
    --port 8111 \
    --gpu-memory-utilization 0.65 \
    --tensor-parallel-size 1 \
    --max-model-len 16384