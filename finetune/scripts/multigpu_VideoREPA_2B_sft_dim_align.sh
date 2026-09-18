#!/usr/bin/env bash
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

BASE_PATH="/efs/zixianhuang/VideoREPA"
FINETUNE_DIR="${BASE_PATH}/finetune"

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${BASE_PATH}/finetune/output_dir"
    --report_to "none"
)

# Data Configuration
DATA_ARGS=(
    # training data
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_3w2.csv
    --data_root "/efs/zixianhuang/VideoREPA/finetune"
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
    --model_path "/efs/zixianhuang/ckpt/cogvideox-2b"
    --model_name "cogvideox-t2v-dim-align"
    --model_type "t2v"
    --training_type "sft"
)

# Dimension-Level GW Alignment Configuration
# Key idea: Instead of aligning token-level similarity (N×N), we align
# dimension-level similarity (D1×D1 vs D2×D2) using GW to handle D1≠D2.
# This captures "which feature dimensions co-activate together" patterns.
#
# Computational notes:
# - GW operates on (D1=1920, D2=768) transport plan
# - Distance matrices: (1920×1920) and (768×768)
# - token_sample_size controls how many tokens are used to compute dim distances
# - dim_sample_size can subsample dimensions if D1/D2 too large (0 = use all)
DIM_ALIGN_ARGS=(
    --loss gw_relational
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 50
    --margin 0.1
    # Dimension-level GW parameters
    --dim_gw_reg 0.1
    --dim_gw_outer_iters 30
    --dim_gw_sinkhorn_iters 100
    --dim_gw_dim_sample_size 0
    --dim_gw_token_sample_size 512
    --dim_gw_update_interval 50
    --dim_gw_outer_tol 1e-4
    --dim_gw_sinkhorn_tol 1e-4
    --comment 'exp2_dim_gw_relational_L18_no_projector'
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
    "${DIM_ALIGN_ARGS[@]}"
