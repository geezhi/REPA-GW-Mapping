#!/usr/bin/env bash
# =============================================================================
# Epoch Ablation Experiment for VideoREPA
# =============================================================================
# Hypothesis: REPA fails in video generation because few epochs lead to sparse
# timestep coverage per sample. Each sample only gets aligned at a few random
# timesteps across training, unlike image REPA which trains for hundreds of epochs.
#
# Experiment: Small dataset (3k) + many epochs vs Full dataset (32k) + few epochs
# Keep total training steps comparable to isolate the effect of per-sample
# timestep coverage.
#
# Usage:
#   1. First run: python create_subset_data.py  (to create the 3k subset)
#   2. Then run this script with CONFIG variable set:
#      CONFIG=B bash epoch_ablation_experiment.sh
#
# Configs:
#   A - Baseline: full data (32k), 4 epochs, with REPA
#   B - Small data (3k), 20 epochs, with REPA
#   C - Small data (3k), 40 epochs, with REPA
#   D - Small data (3k), 80 epochs, with REPA
#   E - Control: full data (32k), 4 epochs, WITHOUT REPA (vanilla finetune)
# =============================================================================

export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

BASE_PATH="/efs/zixianhuang/VideoREPA"

# Select experiment config (default: B)
CONFIG=${CONFIG:-"B"}

echo "============================================="
echo "  Running Epoch Ablation Experiment: Config ${CONFIG}"
echo "============================================="

# ---- Config-specific settings ----
case $CONFIG in
    A)
        TRAIN_DATA="${BASE_PATH}/finetune/openvid/openvid_3w2.csv"
        EPOCHS=4
        LOSS="cosine_similarity_dual_timestep"
        MODEL_NAME="cogvideox-t2v-align"
        COMMENT="ablation_A_full32k_4ep_REPA"
        CKPT_STEPS=800
        VAL_STEPS=800
        ;;
    B)
        TRAIN_DATA="${BASE_PATH}/finetune/openvid/openvid_subset_3000.csv"
        EPOCHS=20
        LOSS="cosine_similarity_dual_timestep"
        MODEL_NAME="cogvideox-t2v-align"
        COMMENT="ablation_B_sub3k_20ep_REPA"
        CKPT_STEPS=200
        VAL_STEPS=200
        ;;
    C)
        TRAIN_DATA="${BASE_PATH}/finetune/openvid/openvid_subset_3000.csv"
        EPOCHS=40
        LOSS="cosine_similarity_dual_timestep"
        MODEL_NAME="cogvideox-t2v-align"
        COMMENT="ablation_C_sub3k_40ep_REPA"
        CKPT_STEPS=400
        VAL_STEPS=400
        ;;
    D)
        TRAIN_DATA="${BASE_PATH}/finetune/openvid/openvid_subset_3000.csv"
        EPOCHS=80
        LOSS="cosine_similarity_dual_timestep"
        MODEL_NAME="cogvideox-t2v-align"
        COMMENT="ablation_D_sub3k_80ep_REPA"
        CKPT_STEPS=800
        VAL_STEPS=800
        ;;
    E)
        TRAIN_DATA="${BASE_PATH}/finetune/openvid/openvid_3w2.csv"
        EPOCHS=4
        LOSS="cosine_similarity_dual_timestep"
        MODEL_NAME="cogvideox-t2v"  # No alignment!
        COMMENT="ablation_E_full32k_4ep_noREPA"
        CKPT_STEPS=800
        VAL_STEPS=800
        ;;
    *)
        echo "Unknown config: $CONFIG. Use A/B/C/D/E."
        exit 1
        ;;
esac

echo "  Train data: ${TRAIN_DATA}"
echo "  Epochs: ${EPOCHS}"
echo "  Model: ${MODEL_NAME}"
echo "  Loss: ${LOSS}"
echo "  Comment: ${COMMENT}"
echo "============================================="

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "${BASE_PATH}/finetune/output_dir_ablation/${COMMENT}"
    --report_to "none"
)

# Data Configuration
DATA_ARGS=(
    --train_data_path ${TRAIN_DATA}
    --data_root "${BASE_PATH}/finetune"
    --train_resolution "49x480x720"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs ${EPOCHS}
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
    --checkpointing_steps ${CKPT_STEPS}
    --checkpointing_limit 3
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "${BASE_PATH}/finetune/validation"
    --validation_steps ${VAL_STEPS}
    --validation_prompts "prompts.txt"
    --gen_fps 8
)

# Model Configuration
MODEL_ARGS=(
    --model_path "/efs/zixianhuang/ckpt/cogvideox-2b"
    --model_name "${MODEL_NAME}"
    --model_type "t2v"
    --training_type "sft"
)

# VideoREPA Configuration
VideoREPA_ARGS=(
    --loss ${LOSS}
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.5
    --margin 0.1
    --comment "${COMMENT}"
    --learning_rate 2e-6
)

# Launch training
accelerate launch --main_process_port $((12000 + $RANDOM % 20000)) --num_processes 8 --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${VideoREPA_ARGS[@]}"
