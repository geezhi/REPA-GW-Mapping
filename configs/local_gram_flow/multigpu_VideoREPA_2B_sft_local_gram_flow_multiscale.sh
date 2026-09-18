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
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_3w2.csv
    --data_root "${REPO_ROOT}/finetune"
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
    --model_path "${CKPT_DIR}/cogvideox-2b"
    --model_name "cogvideox-t2v-local-gram-flow-align"
    --model_type "t2v"
    --training_type "sft"
)

# Multi-Scale Local Gram Flow Alignment Configuration
#
# Key idea: Align LOCAL structural dynamics at MULTIPLE spatial scales simultaneously.
# Different physical phenomena manifest at different scales:
# - Fine scale (2x3 patches, 6 tokens each): local texture/particle motion (fluid)
# - Medium scale (5x5 patches, 25 tokens each): object part motion (articulated)
# - Global scale (10x15 = full frame, 150 tokens): overall scene structure change (rigid)
#
# By aggregating across scales, we capture both fine-grained fluid dynamics
# and coarse rigid-body motion — addressing the complementarity gap between
# TRD (good at solid) and single-scale Gram (good at fluid).
#
# Zero additional parameters, dimension-independent, no projector needed.
LOCAL_GRAM_FLOW_ARGS=(
    --loss local_gram_flow
    --align_models VideoMAEv2
    --align_layer 18
    --align_dims 768
    --proj_coeff 50
    --margin 0.1
    # Multi-scale Local Gram Flow parameters
    --lgf_multiscale
    --lgf_alpha 1.0
    --lgf_beta 0.0
    --comment 'exp2_multiscale_local_gram_flow_alpha1_beta0'
    --learning_rate 2e-6
)

JOB_NAME='test'
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# ==============================================================================
#  Helper: 无论后续步骤成功或失败，最终都启动占卡
# ==============================================================================
cleanup_and_occupy() {
    echo "[$(date '+%F %T')] 启动占卡程序..."
    # Optional machine-specific GPU occupancy guard, not shipped with this repo.
    if [ -n "${OCCUPY_GUARD:-}" ]; then bash "${OCCUPY_GUARD}" start 2>/dev/null || true; fi
    echo "[$(date '+%F %T')] 占卡守护已启动"
}
trap cleanup_and_occupy EXIT  # 脚本退出时（正常/异常/被kill）都会执行

# ==============================================================================
#  Stage 0: Training
# ==============================================================================
echo "[$(date '+%F %T')] ===== 开始训练 ====="
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
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[$(date '+%F %T')] ERROR: 训练失败 (exit code: $TRAIN_EXIT)，跳过评测"
    exit $TRAIN_EXIT  # trap 会自动启动占卡
fi
echo "[$(date '+%F %T')] 训练完成"

# ==============================================================================
#  Stage 1: Merge DeepSpeed checkpoint
# ==============================================================================
echo "[$(date '+%F %T')] Stage 1: Merge checkpoint..."
ORIG_MODEL="${CKPT_DIR}/cogvideox-2b"
EXP_COMMENT="exp2_multiscale_local_gram_flow_alpha1_beta0"
EXP_DIR=$(ls -dt ${FINETUNE_DIR}/output_dir_*${EXP_COMMENT}* 2>/dev/null | head -1)

if [ -z "${EXP_DIR}" ]; then
    echo "[$(date '+%F %T')] ERROR: 找不到实验目录，跳过评测"
    exit 1
fi
echo "[$(date '+%F %T')] 实验目录: ${EXP_DIR}"

CHECKPOINT_STEP=$(ls -d ${EXP_DIR}/checkpoint-* 2>/dev/null | sed 's#.*/##' | sort -t- -k2 -n | tail -1)
if [ -z "${CHECKPOINT_STEP}" ]; then
    echo "[$(date '+%F %T')] ERROR: 找不到 checkpoint，跳过评测"
    exit 1
fi
echo "[$(date '+%F %T')] 使用 checkpoint: ${CHECKPOINT_STEP}"

if [ ! -d "${EXP_DIR}/transformer" ] || [ ! -f "${EXP_DIR}/transformer/config.json" ]; then
    cd "${EXP_DIR}/${CHECKPOINT_STEP}"
    python zero_to_fp32.py ./ ../transformer --safe_serialization
    if [ -f "../transformer/model.safetensors.index.json" ]; then
        mv ../transformer/model.safetensors.index.json ../transformer/diffusion_pytorch_model.safetensors.index.json
    fi
    cp "${ORIG_MODEL}/transformer/config.json" "../transformer/config.json"
    # 清除 downsampler/projector 等额外 key
    python -c "
from safetensors.torch import load_file, save_file
import glob, os, json
path = '${EXP_DIR}/transformer'
for f in glob.glob(os.path.join(path, '*.safetensors')):
    tensors = load_file(f)
    removed = [k for k in tensors if 'downsampler' in k or 'projector' in k]
    if removed:
        for k in removed: del tensors[k]
        save_file(tensors, f)
        print(f'Removed from {os.path.basename(f)}: {removed}')
index_file = os.path.join(path, 'diffusion_pytorch_model.safetensors.index.json')
if os.path.exists(index_file):
    with open(index_file) as fi: index = json.load(fi)
    index['weight_map'] = {k:v for k,v in index['weight_map'].items() if 'downsampler' not in k and 'projector' not in k}
    with open(index_file, 'w') as fo: json.dump(index, fo, indent=2)
"
    echo "[$(date '+%F %T')] Merge 完成"
else
    echo "[$(date '+%F %T')] Transformer 已存在，跳过 merge"
fi

# 组装推理模型（符号链接）
INFER_MODEL_DIR="${EXP_DIR}"
for comp in scheduler text_encoder tokenizer vae model_index.json; do
    [ ! -e "${INFER_MODEL_DIR}/${comp}" ] && ln -sf "${ORIG_MODEL}/${comp}" "${INFER_MODEL_DIR}/${comp}"
done

# ==============================================================================
#  Stage 2: Generate VideoPhy v1 test videos (8 GPU)
# ==============================================================================
INFERENCE_DIR="${BASE_PATH}/inference"
EXP_NAME="local_gram_flow_multiscale"
OUTPUT_VIDEO_DIR="${INFERENCE_DIR}/output_dir/${EXP_NAME}_videophy"
echo "[$(date '+%F %T')] Stage 2: 生成 344 个视频..."
cd "${INFERENCE_DIR}"
mkdir -p "${OUTPUT_VIDEO_DIR}"

for gpu_id in $(seq 0 7); do
    (
        for retry in $(seq 1 5); do
            find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
            generated=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
            [ "${generated}" -ge 344 ] && break
            env CUDA_VISIBLE_DEVICES=$gpu_id python generate.py \
                --input_file "./videophy.txt" \
                --output_dir "${OUTPUT_VIDEO_DIR}" \
                --model_path "${INFER_MODEL_DIR}" \
                --generate_type t2v \
                --gw_model \
                --upsampled \
                --seed 42
            [ $? -eq 0 ] && break
            find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
            sleep 10
        done
    ) &
    sleep 2
done
wait
find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
echo "[$(date '+%F %T')] Stage 2 完成: $(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" | wc -l)/344"

# ==============================================================================
#  Stage 3: VideoPhy v1 AutoEval
# ==============================================================================
EVAL_V1_DIR="${BASE_PATH}/evaluation/videophy"
VIDEOPHY_V1_CKPT="${CKPT_DIR}/videocon_physics"
echo "[$(date '+%F %T')] Stage 3: VideoPhy 评测..."
cd "${EVAL_V1_DIR}"

CSV_NAME="${EXP_NAME}_videophy"
CSV_FILE="./csv_file/${CSV_NAME}.csv"
OUTPUT_FOLDER="./output_dir/${CSV_NAME}"
mkdir -p "./csv_file" "${OUTPUT_FOLDER}"

VIDEO_PREFIX="${OUTPUT_VIDEO_DIR}/"
python -c "
import csv, os
prefix = '${VIDEO_PREFIX}'
if not any(f.endswith('.mp4') for f in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, f))):
    for sub in os.listdir(prefix):
        sp = os.path.join(prefix, sub)
        if os.path.isdir(sp) and any(f.endswith('.mp4') for f in os.listdir(sp)):
            prefix = sp + '/'; break
with open('videophy.txt') as fa, open('${CSV_FILE}', 'w') as fb:
    w = csv.writer(fb); w.writerow(['videopath', 'caption'])
    for line in fa:
        c = line.strip().rstrip('.')
        w.writerow([prefix + c.replace(' ', '_') + '.mp4', c])
count = sum(1 for row in csv.reader(open('${CSV_FILE}')) if os.path.exists(row[0]))
print(f'Found {count-1}/344 videos')
"

python utils/prepare_data.py --input_csv "${CSV_FILE}" --output_folder "${OUTPUT_FOLDER}"

echo "[$(date '+%F %T')] Running SA evaluation..."
python videocon/training/pipeline_video/entailment_inference.py \
    --input_csv "${OUTPUT_FOLDER}/sa_testing.csv" \
    --output_csv "${OUTPUT_FOLDER}/videocon_physics_sa_testing.csv" \
    --checkpoint "${VIDEOPHY_V1_CKPT}"

echo "[$(date '+%F %T')] Running PC evaluation..."
python videocon/training/pipeline_video/entailment_inference.py \
    --input_csv "${OUTPUT_FOLDER}/physics_testing.csv" \
    --output_csv "${OUTPUT_FOLDER}/videocon_physics_pc_testing.csv" \
    --checkpoint "${VIDEOPHY_V1_CKPT}"

cp calculate_mean.py videophy.txt reference.csv "${OUTPUT_FOLDER}/"
cd "${OUTPUT_FOLDER}" && python calculate_mean.py

echo ""
echo "========== VideoPhy v1 Results: ${EXP_NAME} =========="
cat sa_*.txt 2>/dev/null
cat pc_*.txt 2>/dev/null
echo ""
echo "[$(date '+%F %T')] 全部完成!"
# trap EXIT 会自动启动占卡
