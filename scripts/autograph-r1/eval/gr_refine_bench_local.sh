CHECKPOINT_PATH="/data/checkpoints/20260301_010653_qwen2.5-3B-autograph-easy-docsize15-textlinkingFalse-loose"
STEP_NUM="50"
python3 benchmark/autograph/benchmarking_text_refiner.py  --model_name $CHECKPOINT_PATH/global_step_$STEP_NUM/actor/huggingface --refine