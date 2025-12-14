# Replace CHECKPOINT_PATH with the trainer.default_local_dir from your training script
# and STEP_NUM with the checkpoint step you want to convert (e.g., 50).
CHECKPOINT_PATH="/data/haoyuhuang/data/AtlasTune/checkpoints/20251211_234743_qwen2.5-3B-autograph-easy-docsize15-textlinkingFalse-loose"
STEP_NUM="350"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_PATH/global_step_$STEP_NUM/actor \
    --target_dir $CHECKPOINT_PATH/global_step_$STEP_NUM/actor/huggingface