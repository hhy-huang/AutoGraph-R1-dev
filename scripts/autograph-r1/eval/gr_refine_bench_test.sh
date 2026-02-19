CHECKPOINT_PATH="/data/haoyuhuang/data/AtlasTune/checkpoints/20251211_234743_qwen2.5-3B-autograph-easy-docsize15-textlinkingFalse-loose"
STEP_NUM="350"

python benchmark/autograph/benchmarking_graph_refiner_case.py  --model_name $CHECKPOINT_PATH/global_step_$STEP_NUM/actor/huggingface --refine