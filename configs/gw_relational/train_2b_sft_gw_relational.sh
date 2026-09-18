#!/usr/bin/env bash
# =============================================================================
# GW-Relational alignment  --  CogVideoX-2B full fine-tuning (SFT) on OpenVid
#
# Reproduces the ``exp4_zero_param_avgpool_gram_cosine`` configuration.
#
# Usage:
#   bash configs/gw_relational/train_2b_sft_gw_relational.sh [extra train.py args]
#
# Environment overrides:
#   CKPT_DIR   root containing cogvideox-2b / VideoMAEv2  (default /efs/zixianhuang/ckpt)
#   NUM_GPUS   number of processes for accelerate          (default 8)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${CKPT_DIR:-/efs/zixianhuang/ckpt}"
NUM_GPUS="${NUM_GPUS:-8}"

export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

if [ -f "${REPO_ROOT}/videorepa/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/videorepa/bin/activate"
fi

OUTPUT_ARGS=(
    --output_dir "${REPO_ROOT}/finetune/output_dir"
    --report_to "none"
)

DATA_ARGS=(
    --train_data_path "${REPO_ROOT}/finetune/openvid/openvid_3w2.csv"
    --data_root "${REPO_ROOT}/finetune"
    --train_resolution "49x480x720"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
)

TRAIN_ARGS=(
    --train_epochs 4
    --seed 42
    --batch_size 4
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    --learning_rate 2e-6
)

SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

CHECKPOINT_ARGS=(
    --checkpointing_steps 400
    --checkpointing_limit 2
)

VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "${REPO_ROOT}/finetune/validation"
    --validation_steps 400
    --validation_prompts "prompts.txt"
    --gen_fps 8
)

MODEL_ARGS=(
    --model_path "${CKPT_DIR}/cogvideox-2b"
    --model_name "cogvideox-t2v-gw-align"
    --model_type "t2v"
    --training_type "sft"
)

# ---- GW alignment -----------------------------------------------------------
GW_ALIGN_ARGS=(
    --loss gw_relational
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 1.0
    --gw_reg 0.1
    --gw_outer_iters 50
    --gw_sinkhorn_iters 100
    --gw_sample_size 0
    --gw_outer_tol 1e-4
    --gw_sinkhorn_tol 1e-4
    --gw_margin 0.0
    --comment 'exp4_zero_param_avgpool_gram_cosine'
)

cd "${REPO_ROOT}/finetune"
accelerate launch \
    --main_process_port $((12000 + RANDOM % 20000)) \
    --num_processes "${NUM_GPUS}" \
    --config_file accelerate_config.yaml \
    train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${GW_ALIGN_ARGS[@]}" \
    "$@"
