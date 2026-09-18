#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
source ${REPO_ROOT}/videorepa/bin/activate
export PATH="${REPO_ROOT}/videorepa/bin:$PATH"

export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${REPO_ROOT}/finetune/output_dir"
    --report_to "none"
)

# Data Configuration
DATA_ARGS=(
    --train_data_path ${REPO_ROOT}/finetune/openvid/openvid_6w4.csv
    --data_root "${REPO_ROOT}/finetune"
    --train_resolution "49x480x720"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 4
    --seed 42
    --batch_size 4
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 0
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 800
    --checkpointing_limit 2
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "${REPO_ROOT}/finetune/validation"
    --validation_steps 800
    --validation_prompts "prompts.txt"
    --gen_fps 8
)

# Model Configuration - 使用2B模型做预计算
MODEL_ARGS=(
    --model_path "${CKPT_DIR}/cogvideox-2b"
    --model_name "cogvideox-t2v"
    --model_type "t2v"
    --training_type "sft"
    --precomputing
)

# VideoREPA Configuration (预计算时这些参数不影响，但需要传入)
VideoREPA_ARGS=(
    --loss token_relation_distillation
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.5
    --margin 0.1
    --comment 'precompute_6w4'
    --learning_rate 2e-6
)

JOB_NAME='test'
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

cd ${REPO_ROOT}/finetune
accelerate launch --main_process_port $((12000 + $RANDOM % 20000)) --num_processes 8 --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${VideoREPA_ARGS[@]}"
