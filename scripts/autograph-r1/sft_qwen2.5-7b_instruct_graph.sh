export CUDA_VISIBLE_DEVICES=1,2
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_LEVEL=NVL

CONFIG_FILE="verl/third_party/autograph_r1/config.ini"
get_config_value() {
    local section=$1
    local key=$2
    awk -F '=' -v section="[$section]" -v key="$key" '
    $0 ~ section { in_section=1; next }
    /^\[.*\]/ { in_section=0 }
    in_section && $1 ~ key { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }
    ' "$CONFIG_FILE"
}

# Example: Get values from the config file
WANDB_API_KEY=$(get_config_value "logging" "WANDB_API_KEY")
# Print the values (for debugging)
export WANDB_API_KEY
set -x

torchrun --rdzv_endpoint=127.0.0.1:12887 --nproc_per_node=2 -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=128 \
    data.train_files=/home/tht/AutoGraph-R1/data/sft_data/train_sft_1000.parquet \
    data.val_files=/home/tht/AutoGraph-R1/data/sft_data/validation_sft.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=8192 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.strategy=fsdp \
    trainer.project_name=auto_graph_sft \
    trainer.experiment_name=autograph-sft-qwen2.5-7b-instruct-5step \
    trainer.total_epochs=5 \
    trainer.n_gpus_per_node=2 \
    trainer.logger='["console","wandb"]'\
    trainer.default_local_dir="/data/tht/AutoGraph-R1/checkpoints/sft_qwen2.5-7b_instruct-5step" \