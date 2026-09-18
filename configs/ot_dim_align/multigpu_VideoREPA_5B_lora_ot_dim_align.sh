#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY

source ${REPO_ROOT}/videorepa/bin/activate
export PATH="${REPO_ROOT}/videorepa/bin:$PATH"

BASE_PATH="${REPO_ROOT}"
FINETUNE_DIR="${BASE_PATH}/finetune"

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${BASE_PATH}/finetune/output_dir"
    --report_to "none"
)

# Data Configuration
DATA_ARGS=(
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_6w4.csv
    --data_root "${BASE_PATH}/finetune"
    --train_resolution "49x480x720"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 1
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
    --checkpointing_steps 800
    --checkpointing_limit 2
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "${BASE_PATH}/finetune/validation"
    --validation_steps 800
    --validation_prompts "prompts.txt"
    --gen_fps 8
)

# Model Configuration
MODEL_ARGS=(
    --model_path "${CKPT_DIR}/cogvideox-5b"
    --model_name "cogvideox-t2v-ot-dim-align"
    --model_type "t2v"
    --training_type "lora"
)

# OT Dimension Alignment Configuration (5B LoRA version)
#
# Key idea: Use standard OT (Sinkhorn) instead of GW to find dimension mapping.
# Since tokens are spatially aligned, we can directly compute cross-space cost:
#   Cost[i,j] = cosine_distance(student_dim_i, teacher_dim_j)
# This avoids the expensive GW outer loop (~50x faster).
#
# "OT in reverse": dimensions are mass, tokens are features.
OT_DIM_ALIGN_ARGS=(
    --loss gw_relational
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.5
    --margin 0.0
    # OT parameters
    --ot_reg 0.1
    --ot_sinkhorn_iters 100
    --ot_tol 1e-4
    --comment '5B_lora_ot_dim_align_L18_coeff0.5'
    --learning_rate 1e-4
    --rank 128
    --lora_alpha 64
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
    "${OT_DIM_ALIGN_ARGS[@]}"
