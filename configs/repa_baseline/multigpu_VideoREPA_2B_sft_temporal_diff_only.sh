#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

BASE_PATH="${REPO_ROOT}"
FINETUNE_DIR="${BASE_PATH}/finetune"

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${BASE_PATH}/finetune/output_dir_cogvideox-t2v-align_cosine_similarity_exp2_temporal_diff_only_coeff0.1"
    --report_to "none"
)

# Data Configuration
DATA_ARGS=(
    # training data
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_3w2.csv
    --data_root "${REPO_ROOT}/finetune"
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
    # place holder (useless)
    --caption_column "prompt.txt"
    --video_column "videos.txt"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 4
    --seed 42 

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 4
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    ########################################################################
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

# VideoREPA Configuration - Pure Temporal Difference Alignment
# Key idea: ALL frames use inter-frame differences for alignment (no original features).
# Unlike temporal_diff which keeps the first frame as original features,
# this variant computes frame[t] - frame[t-1] for ALL frames (discarding frame 0),
# focusing purely on motion dynamics alignment.
VideoREPA_ARGS=(
    --loss cosine_similarity_temporal_diff_only
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.1
    --margin 0.1
    --comment 'exp2_temporal_diff_only_alignment_coeff0.1'
    --learning_rate 2e-6
)

JOB_NAME='test'
GPUS=${GPUS:-1}        
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# Combine all arguments and launch training
cd "${FINETUNE_DIR}"
accelerate launch --main_process_port $((12000 + $RANDOM % 20000)) --num_processes 8 --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${VideoREPA_ARGS[@]}"
