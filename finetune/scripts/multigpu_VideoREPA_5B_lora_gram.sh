#!/usr/bin/env bash
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY

source /efs/zixianhuang/VideoREPA/videorepa/bin/activate
export PATH="/efs/zixianhuang/VideoREPA/videorepa/bin:$PATH"

BASE_PATH="/efs/zixianhuang/VideoREPA"
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
    --model_path "/efs/zixianhuang/ckpt/cogvideox-5b"
    --model_name "cogvideox-t2v-align"
    --model_type "t2v"
    --training_type "lora"
)

# Gram Matrix Alignment Configuration (with learnable projector)
# Key idea: Project student features (1920d) to teacher space (768d) via MLP projector,
# then align cosine Gram matrices (structure-preserving) with MSE loss.
GRAM_ARGS=(
    --loss gram_matrix
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 0.5
    --margin 0.0
    --comment '5B_lora_gram_proj_L18_coeff0.5'
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
    "${GRAM_ARGS[@]}"
