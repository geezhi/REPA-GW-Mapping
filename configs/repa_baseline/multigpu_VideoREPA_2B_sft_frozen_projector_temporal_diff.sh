#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
source ${REPO_ROOT}/videorepa/bin/activate
export PATH="${REPO_ROOT}/videorepa/bin:$PATH"
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

BASE_PATH="${REPO_ROOT}"

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${BASE_PATH}/finetune/output_dir"
    --report_to "none"
)

# Data Configuration
DATA_ARGS=(
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_3w2.csv
    --data_root "${BASE_PATH}/finetune"
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
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 400
    --checkpointing_limit 2
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "${BASE_PATH}/finetune/validation"
    --validation_steps 400
    --validation_prompts "prompts.txt"
    --gen_fps 8
)

# Model Configuration
MODEL_ARGS=(
    --model_path "${CKPT_DIR}/cogvideox-2b"
    --model_name "cogvideox-t2v-align"
    --model_type "t2v"
    --training_type "sft"
)

# VideoREPA Configuration - Temporal Diff Only with frozen pretrained projector
VideoREPA_ARGS=(
    --loss cosine_similarity_temporal_diff_only
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.5
    --margin 0.1
    --comment 'exp_frozen_projector_temporal_diff_only_3w2_4ep'
    --learning_rate 2e-6
    --pretrained_projector_path "${BASE_PATH}/finetune/pretrained_projector.pth"
)

# Launch training with 8 GPUs
accelerate launch --main_process_port $((12000 + $RANDOM % 20000)) --num_processes 8 --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${VideoREPA_ARGS[@]}"
