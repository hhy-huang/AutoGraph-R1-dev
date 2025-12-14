CHECKPOINT_PATH="/data/haoyuhuang/data/AtlasTune/checkpoints/20251211_234743_qwen2.5-3B-autograph-easy-docsize15-textlinkingFalse-loose"
STEP_NUM="350"

# Adjust the API url in the python script as needed
python benchmark/autograph/custom_kg_extraction.py --model_name $CHECKPOINT_PATH/global_step_$STEP_NUM/actor/huggingface