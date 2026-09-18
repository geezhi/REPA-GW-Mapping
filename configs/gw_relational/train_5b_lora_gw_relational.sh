#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
# =============================================================================
# GW-Relational alignment (default variant)  --  CogVideoX-5B LoRA on OpenVid
#
# Reproduces the ``5B_lora_gw_align_L18_coeff0.5`` run:
#   * teacher        : VideoMAEv2 (frozen), aligned at transformer layer 18
#   * alignment      : GW dimension transport plan (D1=1920 -> D2=768, no projector)
#   * loss           : gw_relational  (per-token cosine after GW projection)
#
# Usage:
#   bash configs/gw_relational/train_5b_lora_gw_relational.sh [extra train.py args]
#
# Environment overrides:
#   CKPT_DIR   root containing cogvideox-5b / VideoMAEv2  (default ${CKPT_DIR})
#   NUM_GPUS   number of processes for accelerate          (default 8)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
NUM_GPUS="${NUM_GPUS:-8}"

export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY

if [ -f "${REPO_ROOT}/videorepa/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/videorepa/bin/activate"
fi

OUTPUT_ARGS=(
    --output_dir "${REPO_ROOT}/finetune/output_dir"
    --report_to "none"
)

DATA_ARGS=(
    --train_data_path "${REPO_ROOT}/finetune/openvid/openvid_6w4.csv"
    --data_root "${REPO_ROOT}/finetune"
    --train_resolution "49x480x720"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
)

TRAIN_ARGS=(
    --train_epochs 1
    --seed 42
    --batch_size 4
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"
    --learning_rate 1e-4
    --rank 128
    --lora_alpha 64
)

SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

CHECKPOINT_ARGS=(
    --checkpointing_steps 800
    --checkpointing_limit 2
)

VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "${REPO_ROOT}/finetune/validation"
    --validation_steps 800
    --validation_prompts "prompts.txt"
    --gen_fps 8
)

MODEL_ARGS=(
    --model_path "${CKPT_DIR}/cogvideox-5b"
    --model_name "cogvideox-t2v-gw-align"
    --model_type "t2v"
    --training_type "lora"
)

# ---- GW alignment -----------------------------------------------------------
GW_ALIGN_ARGS=(
    --loss gw_relational
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.5       # lambda in L = L_diffusion + lambda * L_GW
    --gw_reg 0.1           # eps      : entropic regularization
    --gw_outer_iters 50    # K_outer  : linearization steps
    --gw_sinkhorn_iters 100 # K_sink  : inner Sinkhorn steps
    --gw_sample_size 0     # 0 = use all 3600 tokens
    --gw_outer_tol 1e-4
    --gw_sinkhorn_tol 1e-4
    --gw_margin 0.0        # 0 = no hinge on the cosine distance
    --comment '5B_lora_gw_align_L18_coeff0.5'
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
