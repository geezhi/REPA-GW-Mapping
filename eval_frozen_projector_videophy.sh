#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
set -e
source ${REPO_ROOT}/videorepa/bin/activate
export PATH="${REPO_ROOT}/videorepa/bin:$PATH"

###############################################################################
# Evaluate all 3 frozen-projector experiments on VideoPhy v1
#
# Experiments:
#   1. cosine_similarity (REPA)
#   2. token_relation_distillation (TRD)
#   3. cosine_similarity_temporal_diff_only
#
# Three stages per experiment:
#   Stage 1: Merge DeepSpeed checkpoint -> inference model
#   Stage 2: Generate 344 VideoPhy v1 test videos
#   Stage 3: Run VideoPhy v1 AutoEval (SA + PC)
#
# Usage:
#   bash eval_frozen_projector_videophy.sh [--stage 1|2|3|all] [--exp 1|2|3|all]
###############################################################################

BASE_PATH="${REPO_ROOT}"
FINETUNE_DIR="${BASE_PATH}/finetune"
INFERENCE_DIR="${BASE_PATH}/inference"
EVAL_V1_DIR="${BASE_PATH}/evaluation/videophy"
ORIG_MODEL="${CKPT_DIR}/cogvideox-2b"
VIDEOPHY_V1_CKPT="${CKPT_DIR}/videocon_physics"

# Experiment configs: (name, output_dir_suffix, checkpoint_step)
declare -A EXP_NAMES
declare -A EXP_DIRS
declare -A EXP_CKPTS

EXP_NAMES[1]="frozen_proj_REPA"
EXP_DIRS[1]="${FINETUNE_DIR}/output_dir_cogvideox-t2v-align_cosine_similarity_exp_frozen_projector_3w2_4ep"
EXP_CKPTS[1]="checkpoint-4000"

EXP_NAMES[2]="frozen_proj_TRD"
EXP_DIRS[2]="${FINETUNE_DIR}/output_dir_cogvideox-t2v-align_token_relation_distillation_exp_frozen_projector_TRD_3w2_4ep"
EXP_CKPTS[2]="checkpoint-4000"

EXP_NAMES[3]="frozen_proj_temporal_diff"
EXP_DIRS[3]="${FINETUNE_DIR}/output_dir_cogvideox-t2v-align_cosine_similarity_temporal_diff_only_exp_frozen_projector_temporal_diff_only_3w2_4ep"
EXP_CKPTS[3]="checkpoint-4000"

# Parse arguments
STAGE="all"
EXP_ID="all"
NUM_GPUS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        --exp) EXP_ID="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Determine which experiments to run
if [ "$EXP_ID" = "all" ]; then
    EXP_LIST="1 2 3"
else
    EXP_LIST="$EXP_ID"
fi

###############################################################################
# Stage 0: Download VideoPhy v1 AutoEval checkpoint if needed
###############################################################################
if [ ! -d "${VIDEOPHY_V1_CKPT}" ]; then
    echo ">>> Downloading VideoPhy v1 AutoEval checkpoint..."
    huggingface-cli download --repo-type model videophysics/videocon_physics \
        --local-dir "${VIDEOPHY_V1_CKPT}"
    echo ">>> Downloaded to ${VIDEOPHY_V1_CKPT}"
fi

###############################################################################
# Run evaluation for each experiment
###############################################################################
for exp_id in $EXP_LIST; do
    EXP_NAME="${EXP_NAMES[$exp_id]}"
    EXP_DIR="${EXP_DIRS[$exp_id]}"
    CHECKPOINT_STEP="${EXP_CKPTS[$exp_id]}"
    INFER_MODEL_DIR="${BASE_PATH}/infer_model/${EXP_NAME}"
    OUTPUT_VIDEO_DIR="${INFERENCE_DIR}/output_dir/${EXP_NAME}_videophy"

    echo ""
    echo "============================================================"
    echo "Evaluating Experiment ${exp_id}: ${EXP_NAME}"
    echo "  Checkpoint: ${EXP_DIR}/${CHECKPOINT_STEP}"
    echo "  Stage: ${STAGE}"
    echo "============================================================"

    # Check if checkpoint exists
    if [ ! -d "${EXP_DIR}/${CHECKPOINT_STEP}" ]; then
        echo ">>> WARNING: Checkpoint not found at ${EXP_DIR}/${CHECKPOINT_STEP}, skipping."
        continue
    fi

    # ======================== Stage 1: Merge Checkpoint ========================
    if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
        echo ""
        echo "--- Stage 1: Merging DeepSpeed Checkpoint ---"

        if [ ! -d "${EXP_DIR}/transformer" ]; then
            echo ">>> Running zero_to_fp32.py..."
            cd "${EXP_DIR}/${CHECKPOINT_STEP}"
            python zero_to_fp32.py ./ ../transformer --safe_serialization

            if [ -f "../transformer/model.safetensors.index.json" ]; then
                mv ../transformer/model.safetensors.index.json ../transformer/diffusion_pytorch_model.safetensors.index.json
            fi

            if [ -f "${ORIG_MODEL}/transformer/config.json" ]; then
                cp "${ORIG_MODEL}/transformer/config.json" "../transformer/config.json"
            fi
            echo ">>> Checkpoint merged."
        else
            echo ">>> Transformer directory exists, skipping merge."
        fi

        if [ ! -d "${INFER_MODEL_DIR}" ]; then
            echo ">>> Creating inference model directory..."
            mkdir -p "${INFER_MODEL_DIR}"
            for item in "${ORIG_MODEL}"/*; do
                basename_item=$(basename "$item")
                if [ "$basename_item" != "transformer" ]; then
                    cp -r "$item" "${INFER_MODEL_DIR}/${basename_item}"
                fi
            done
            cp -r "${EXP_DIR}/transformer" "${INFER_MODEL_DIR}/transformer"
            echo ">>> Inference model at: ${INFER_MODEL_DIR}"
        else
            echo ">>> Inference directory exists, skipping."
        fi
        echo ">>> Stage 1 Complete!"
    fi

    # ======================== Stage 2: Generate Videos ========================
    if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
        echo ""
        echo "--- Stage 2: Generating VideoPhy v1 Test Videos (344 prompts) ---"
        echo ">>> Model: ${INFER_MODEL_DIR}"
        echo ">>> Output: ${OUTPUT_VIDEO_DIR}"

        cd "${INFERENCE_DIR}"
        mkdir -p "${OUTPUT_VIDEO_DIR}"

        TOTAL_PROMPTS=344
        MAX_RETRIES=5

        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            (
                retry=0
                while [ $retry -lt ${MAX_RETRIES} ]; do
                    find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
                    generated=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
                    if [ "${generated}" -ge "${TOTAL_PROMPTS}" ]; then
                        break
                    fi
                    echo "[GPU ${gpu_id}] Attempt $((retry + 1)), ${generated}/${TOTAL_PROMPTS} done."
                    env CUDA_VISIBLE_DEVICES=$gpu_id python generate.py \
                        --input_file "./videophy.txt" \
                        --output_dir "${OUTPUT_VIDEO_DIR}" \
                        --model_path "${INFER_MODEL_DIR}" \
                        --generate_type t2v \
                        --upsampled \
                        --seed 42
                    exit_code=$?
                    retry=$((retry + 1))
                    if [ $exit_code -ne 0 ]; then
                        find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
                        sleep 10
                    else
                        break
                    fi
                done
            ) &
            sleep 2
        done
        wait

        find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
        FINAL_COUNT=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
        echo ">>> Stage 2 Complete! ${FINAL_COUNT}/${TOTAL_PROMPTS} videos generated."
    fi

    # ======================== Stage 3: VideoPhy v1 AutoEval ========================
    if [ "$STAGE" = "3" ] || [ "$STAGE" = "all" ]; then
        echo ""
        echo "--- Stage 3: VideoPhy v1 AutoEval ---"

        if [ ! -d "${VIDEOPHY_V1_CKPT}" ]; then
            echo ">>> ERROR: videocon_physics checkpoint not found!"
            continue
        fi

        cd "${EVAL_V1_DIR}"

        CSV_NAME="${EXP_NAME}_videophy"
        CSV_FILE="./csv_file/${CSV_NAME}.csv"
        OUTPUT_FOLDER="./output_dir/${CSV_NAME}"
        mkdir -p "./csv_file" "${OUTPUT_FOLDER}"

        # Generate CSV
        VIDEO_PREFIX="${OUTPUT_VIDEO_DIR}/"
        python -c "
import csv, os
prefix = '${VIDEO_PREFIX}'
# Check if videos are in a subdirectory (e.g. videophy/)
if not any(f.endswith('.mp4') for f in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, f))):
    for sub in os.listdir(prefix):
        sub_path = os.path.join(prefix, sub)
        if os.path.isdir(sub_path) and any(f.endswith('.mp4') for f in os.listdir(sub_path)):
            prefix = sub_path + '/'
            break
with open('videophy.txt', 'r') as fa, open('${CSV_FILE}', 'w') as fb:
    w = csv.writer(fb)
    w.writerow(['videopath', 'caption'])
    for line in fa:
        caption = line.strip().rstrip('.')
        video_file = caption.replace(' ', '_') + '.mp4'
        w.writerow([prefix + video_file, caption])
count = 0
with open('${CSV_FILE}') as f:
    r = csv.reader(f); next(r)
    for row in r:
        if os.path.exists(row[0]): count += 1
print(f'Found {count}/344 videos')
"

        # Prepare SA and PC
        python utils/prepare_data.py --input_csv "${CSV_FILE}" --output_folder "${OUTPUT_FOLDER}"

        # SA evaluation
        echo ">>> Running SA evaluation..."
        python videocon/training/pipeline_video/entailment_inference.py \
            --input_csv "${OUTPUT_FOLDER}/sa_testing.csv" \
            --output_csv "${OUTPUT_FOLDER}/videocon_physics_sa_testing.csv" \
            --checkpoint "${VIDEOPHY_V1_CKPT}"

        # PC evaluation
        echo ">>> Running PC evaluation..."
        python videocon/training/pipeline_video/entailment_inference.py \
            --input_csv "${OUTPUT_FOLDER}/physics_testing.csv" \
            --output_csv "${OUTPUT_FOLDER}/videocon_physics_pc_testing.csv" \
            --checkpoint "${VIDEOPHY_V1_CKPT}"

        # Calculate metrics
        cp calculate_mean.py "${OUTPUT_FOLDER}/"
        cp videophy.txt "${OUTPUT_FOLDER}/"
        cp reference.csv "${OUTPUT_FOLDER}/"
        cd "${OUTPUT_FOLDER}"
        python calculate_mean.py

        echo ""
        echo "========== VideoPhy v1 Results: ${EXP_NAME} =========="
        cat sa_*.txt 2>/dev/null || true
        cat pc_*.txt 2>/dev/null || true
        echo ""
        echo ">>> Stage 3 Complete!"
    fi

done

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
