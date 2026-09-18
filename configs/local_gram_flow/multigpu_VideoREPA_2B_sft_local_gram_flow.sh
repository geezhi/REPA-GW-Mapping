#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
# Prevent CUDA re-initialization errors in forked DataLoader workers
export CUDA_MODULE_LOADING=LAZY

BASE_PATH="${REPO_ROOT}"
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
    --model_name "cogvideox-t2v-local-gram-flow-align"
    --model_type "t2v"
    --training_type "sft"
)

# Local Gram Flow Alignment Configuration
# Key idea: Align LOCAL structural dynamics — how local feature correlations
# change between frames — rather than absolute features or global structure.
#
# Design:
# - Divide spatial tokens (10x15) into local patches (5x5 = 25 tokens each, 6 patches total)
# - Compute local Gram matrices per patch per frame (25x25, cosine-normalized)
# - Align TEMPORAL DIFFERENCES of local Gram (how structure changes) between student & teacher
# - Zero additional parameters, dimension-independent (no projector needed)
#
# Hyperparameters:
# - lgf_patch_size_h/w: local patch size (5x5 -> 6 patches covering 10x15)
# - lgf_alpha: weight for temporal flow loss (main loss)
# - lgf_beta: weight for static Gram loss (optional, 0 = pure motion alignment)
# - proj_coeff: overall alignment loss coefficient
LOCAL_GRAM_FLOW_ARGS=(
    --loss local_gram_flow
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 50
    --margin 0.1
    # Local Gram Flow parameters
    --lgf_patch_size_h 5
    --lgf_patch_size_w 5
    --lgf_alpha 1.0
    --lgf_beta 0.0
    --comment 'exp1_local_gram_flow_5x5_alpha1_beta0'
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
    "${LOCAL_GRAM_FLOW_ARGS[@]}"
