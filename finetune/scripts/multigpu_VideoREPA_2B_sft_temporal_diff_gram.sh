#!/usr/bin/env bash
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY

BASE_PATH="/efs/zixianhuang/VideoREPA"
FINETUNE_DIR="${BASE_PATH}/finetune"

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${BASE_PATH}/finetune/output_dir"
    --report_to "none"
)

# Data Configuration
DATA_ARGS=(
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_3w2.csv
    --data_root "/efs/zixianhuang/VideoREPA/finetune"
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
    --model_path "/efs/zixianhuang/ckpt/cogvideox-2b"
    --model_name "cogvideox-t2v-local-gram-flow-align"
    --model_type "t2v"
    --training_type "sft"
)

# Temporal Diff + Gram Dual Loss
#
# Two complementary alignment signals at different abstraction levels:
#
# 1. Temporal Diff Cosine (AFTER projector, 768d space):
#    - Per-token motion direction alignment
#    - Student projected to 768d via MLP, then frame diff cosine with teacher
#    - Captures: "is each spatial location moving in the right direction?"
#
# 2. Global Gram (BEFORE projector, native 1920d/768d space):
#    - Token-relation structure alignment (dimension-independent)
#    - Cosine Gram matrices computed in each model's NATIVE space, then MSE
#    - Captures: "do tokens have the correct relative similarity structure?"
#    - No projector needed (Gram shape only depends on token count)
#
# Key design: Gram is computed PRE-projector so it captures raw backbone
# structure without projector distortion. Temporal diff uses projector
# for dimension matching but only on DIFFERENCES (motion), not absolute values.
DUAL_LOSS_ARGS=(
    --loss local_gram_flow
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.5
    --margin 0.1
    # Disable Local Gram Flow (we use global Gram instead)
    --lgf_patch_size_h 10
    --lgf_patch_size_w 15
    --lgf_alpha 0.0
    --lgf_beta 0.0
    # Temporal Diff Cosine (post-projector)
    --lgf_temporal_diff_weight 1.0
    # Global Gram (pre-projector, native space)
    --lgf_gram_weight 1.0
    --comment 'exp4_temporal_diff_plus_gram_native'
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
    "${DUAL_LOSS_ARGS[@]}"
