#!/usr/bin/env bash
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
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
    --model_name "cogvideox-t2v-fixed-proj-align"
    --model_type "t2v"
    --training_type "sft"
)

# Fixed Random Projection Alignment Configuration
#
# Strategy:
# 1. Token downsampling: downsampler_cogvideo_output (30x45 -> 10x15), learnable Conv2d stride=3
# 2. Dimension projection: frozen random W (1920 -> 768), zero learnable parameters
# 3. Per-token cosine loss: mean(1 - cos(X @ W_fixed, Y))
#
# Key insight: Learnable projectors absorb alignment gradients, leaving the
# transformer backbone with little effective signal. A frozen random projection
# (justified by Johnson-Lindenstrauss lemma) provides a valid dimension mapping
# while ensuring 100% of the gradient flows to the backbone.
#
# Unlike Gram matrix alignment (which only preserves relational structure),
# this provides per-token feature-level supervision, so the model actually
# learns dimensional semantics from the teacher.
ALIGN_ARGS=(
    --loss fixed_proj_align
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 1
    --margin 0.1
    # GW params kept for arg parsing compatibility (not used)
    --gw_reg 0.1
    --gw_outer_iters 50
    --gw_sinkhorn_iters 100
    --gw_sample_size 0
    --gw_update_interval 1
    --gw_distance_type euclidean
    --gw_outer_tol 1e-4
    --gw_sinkhorn_tol 1e-4
    --comment 'exp5_fixed_random_proj_cosine'
    --learning_rate 2e-6
)

# Launch training
cd "${FINETUNE_DIR}"
accelerate launch --main_process_port $((12000 + $RANDOM % 20000)) --num_processes 8 --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${ALIGN_ARGS[@]}"
