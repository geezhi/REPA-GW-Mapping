#!/usr/bin/env bash
set -e

###############################################################################
# VideoREPA Exp2 - Temporal Diff - VideoPhy v1 Evaluation Pipeline
#
# This script evaluates the exp2 temporal_diff trained model (checkpoint-4000)
# on the VideoPhy v1 benchmark.
#
# Three stages:
#   Stage 1: Merge DeepSpeed checkpoint-4000 to inference-ready model
#   Stage 2: Generate VideoPhy v1 test videos (344 prompts)
#   Stage 3: Evaluate generated videos using VideoPhy v1 AutoEval
#
# Usage:
#   bash eval_exp2_temporal_diff_videophy.sh [--stage 1|2|3|all]
#
# Prerequisites:
#   - GPU machine with CUDA
#   - Original CogVideoX-2B model components (scheduler, text_encoder, tokenizer, vae)
###############################################################################

# ======================== Configuration ========================
BASE_PATH="/group/40059/zixianhuang/VideoREPA_exp2"
FINETUNE_DIR="${BASE_PATH}/finetune"
INFERENCE_DIR="${BASE_PATH}/inference"
EVAL_V1_DIR="${BASE_PATH}/evaluation/videophy"

# Original CogVideoX-2B model path (reuse from existing ckpt)
ORIG_MODEL="/group/40059/zixianhuang/VideoREPA/ckpt/cogvideox-2b-cosine-sim"

# VideoPhy v1 AutoEval checkpoint path
VIDEOPHY_V1_CKPT="/group/40059/zixianhuang/VideoREPA/ckpt/videocon_physics"

# Experiment directory (exp2: temporal_diff alignment, checkpoint-4000)
EXP_DIR="${BASE_PATH}/finetune/output_dir_cogvideox-t2v-align_cosine_similarity_temporal_diff_exp2_temporal_diff_alignment"
CHECKPOINT_STEP="checkpoint-4000"

# Inference model directory (will be created)
INFER_MODEL_DIR="${BASE_PATH}/infer_model/cogvideox-2b-exp2-temporal-diff"

# Output directory for generated videos (VideoPhy v1)
OUTPUT_VIDEO_DIR="${INFERENCE_DIR}/output_dir/exp2_temporal_diff_videophy"

# ======================== Parse Arguments ========================
STAGE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "VideoREPA Exp2 Temporal Diff - VideoPhy v1 Evaluation"
echo "Stage: ${STAGE}"
echo "Checkpoint: ${EXP_DIR}/${CHECKPOINT_STEP}"
echo "============================================"

# ======================== Stage 1: Merge Checkpoint ========================
if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
    echo ""
    echo "============================================"
    echo "STAGE 1: Merging DeepSpeed Checkpoint"
    echo "============================================"

    # Step 1: Run zero_to_fp32.py to merge DeepSpeed shards
    if [ ! -d "${EXP_DIR}/transformer" ]; then
        echo ">>> Running zero_to_fp32.py to merge checkpoint..."
        cd "${EXP_DIR}/${CHECKPOINT_STEP}"
        python zero_to_fp32.py ./ ../transformer --safe_serialization

        # Rename index file to match diffusers convention
        if [ -f "../transformer/model.safetensors.index.json" ]; then
            mv ../transformer/model.safetensors.index.json ../transformer/diffusion_pytorch_model.safetensors.index.json
        fi

        # Copy config.json from original model's transformer directory
        if [ -f "${ORIG_MODEL}/transformer/config.json" ]; then
            cp "${ORIG_MODEL}/transformer/config.json" "../transformer/config.json"
            echo ">>> Copied config.json from original model."
        fi
        echo ">>> Checkpoint merged successfully."
    else
        echo ">>> Transformer directory already exists, skipping merge."
    fi

    # Step 2: Create inference directory with full model structure
    if [ ! -d "${INFER_MODEL_DIR}" ]; then
        echo ">>> Creating inference model directory..."
        mkdir -p "${INFER_MODEL_DIR}"

        # Copy all components from original model except transformer
        for item in "${ORIG_MODEL}"/*; do
            basename_item=$(basename "$item")
            if [ "$basename_item" != "transformer" ]; then
                cp -r "$item" "${INFER_MODEL_DIR}/${basename_item}"
            fi
        done

        # Copy the merged transformer
        cp -r "${EXP_DIR}/transformer" "${INFER_MODEL_DIR}/transformer"
        echo ">>> Inference model directory created at: ${INFER_MODEL_DIR}"
    else
        echo ">>> Inference directory already exists, skipping."
    fi

    echo ""
    echo ">>> Stage 1 Complete!"
fi

# ======================== Stage 2: Generate Videos ========================
NUM_GPUS=8
MAX_RETRIES_PER_GPU=5  # Maximum retries per GPU

if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
    echo ""
    echo "============================================"
    echo "STAGE 2: Generating VideoPhy v1 Test Videos (344 prompts)"
    echo "============================================"

    echo ">>> Model: ${INFER_MODEL_DIR}"
    echo ">>> Output: ${OUTPUT_VIDEO_DIR}"
    echo ">>> Using ${NUM_GPUS} GPUs in parallel"
    echo ">>> Max retries per GPU: ${MAX_RETRIES_PER_GPU}"

    cd "${INFERENCE_DIR}"
    mkdir -p "${OUTPUT_VIDEO_DIR}"

    TOTAL_PROMPTS=344

    # Each GPU runs in its own retry loop independently
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        (
            retry=0
            while [ $retry -lt ${MAX_RETRIES_PER_GPU} ]; do
                # Clean up any leftover .lock directories
                find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true

                # Check if all videos are done
                generated=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
                if [ "${generated}" -ge "${TOTAL_PROMPTS}" ]; then
                    echo "[GPU ${gpu_id}] All ${TOTAL_PROMPTS} videos done. Exiting."
                    break
                fi

                echo "[GPU ${gpu_id}] Attempt $((retry + 1))/${MAX_RETRIES_PER_GPU}, ${generated}/${TOTAL_PROMPTS} videos done so far."
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
                    echo "[GPU ${gpu_id}] Process crashed (exit code: ${exit_code}). Retrying in 15s..."
                    # Clean up .lock files left by this crashed process
                    find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
                    sleep 15
                else
                    echo "[GPU ${gpu_id}] Process exited normally."
                    break
                fi
            done
        ) &
        sleep 3  # Stagger GPU launches slightly
    done

    echo ">>> All ${NUM_GPUS} GPU workers launched (each with independent retry). Waiting..."
    wait

    # Final cleanup
    find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true

    FINAL_COUNT=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
    echo ""
    echo ">>> Stage 2 Complete! ${FINAL_COUNT}/${TOTAL_PROMPTS} videos saved to: ${OUTPUT_VIDEO_DIR}"
fi

# ======================== Stage 3: Evaluate with VideoPhy v1 AutoEval ========================
if [ "$STAGE" = "3" ] || [ "$STAGE" = "all" ]; then
    echo ""
    echo "============================================"
    echo "STAGE 3: VideoPhy v1 AutoEval Evaluation"
    echo "============================================"

    # Check that VideoPhy v1 checkpoint exists
    if [ ! -d "${VIDEOPHY_V1_CKPT}" ]; then
        echo ">>> ERROR: VideoPhy v1 checkpoint not found at: ${VIDEOPHY_V1_CKPT}"
        echo ">>> Please download from: https://huggingface.co/videophysics/videocon_physics"
        exit 1
    fi

    cd "${EVAL_V1_DIR}"

    # Step 1: Generate evaluation CSV from generated videos
    CSV_NAME="exp2_temporal_diff_videophy"
    CSV_FILE="./csv_file/${CSV_NAME}.csv"
    OUTPUT_FOLDER="./output_dir/${CSV_NAME}"
    mkdir -p "./csv_file"
    mkdir -p "${OUTPUT_FOLDER}"

    echo ">>> Generating evaluation CSV..."
    VIDEO_PREFIX="${OUTPUT_VIDEO_DIR}/"

    python -c "
import csv
import os

prefix = '${VIDEO_PREFIX}'
# Also check if videos are in a subdirectory
if not any(f.endswith('.mp4') for f in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, f))):
    # Try videophy subdirectory
    alt_prefix = prefix + 'videophy/'
    if os.path.exists(alt_prefix):
        prefix = alt_prefix

with open('videophy.txt', 'r', encoding='utf-8') as file_a, open('${CSV_FILE}', 'w', encoding='utf-8') as file_b:
    csv_writer = csv.writer(file_b)
    csv_writer.writerow(['videopath', 'caption'])
    
    for line in file_a:
        words = line.strip().rstrip('.').split(' ')
        video_file = '_'.join(words) + '.mp4'
        video_path = prefix + video_file
        caption = line.strip().rstrip('.')
        csv_writer.writerow([video_path, caption])

print(f'CSV written to ${CSV_FILE}')
# Count existing videos
count = 0
with open('${CSV_FILE}', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if os.path.exists(row[0]):
            count += 1
print(f'Found {count} existing video files out of 344')
"

    # Step 2: Prepare SA and PC testing CSVs
    echo ">>> Preparing SA and PC testing data..."
    python utils/prepare_data.py \
        --input_csv "${CSV_FILE}" \
        --output_folder "${OUTPUT_FOLDER}"

    # Step 3: Run SA (Semantic Adherence) evaluation
    echo ">>> Running SA (Semantic Adherence) evaluation..."
    python videocon/training/pipeline_video/entailment_inference.py \
        --input_csv "${OUTPUT_FOLDER}/sa_testing.csv" \
        --output_csv "${OUTPUT_FOLDER}/videocon_physics_sa_testing.csv" \
        --checkpoint "${VIDEOPHY_V1_CKPT}"

    # Step 4: Run PC (Physical Commonsense) evaluation
    echo ">>> Running PC (Physical Commonsense) evaluation..."
    python videocon/training/pipeline_video/entailment_inference.py \
        --input_csv "${OUTPUT_FOLDER}/physics_testing.csv" \
        --output_csv "${OUTPUT_FOLDER}/videocon_physics_pc_testing.csv" \
        --checkpoint "${VIDEOPHY_V1_CKPT}"

    # Step 5: Calculate final metrics
    echo ">>> Calculating metrics..."
    cp calculate_mean.py "${OUTPUT_FOLDER}/calculate_mean.py"
    cp videophy.txt "${OUTPUT_FOLDER}/videophy.txt"
    cp reference.csv "${OUTPUT_FOLDER}/reference.csv"
    cd "${OUTPUT_FOLDER}"
    python calculate_mean.py

    echo ""
    echo "========== VideoPhy v1 Results for Exp2 Temporal Diff =========="
    echo "Checkpoint: ${EXP_DIR}/${CHECKPOINT_STEP}"
    echo "Output directory: ${OUTPUT_FOLDER}"
    cat sa_*.txt 2>/dev/null || true
    cat pc_*.txt 2>/dev/null || true
    echo ""
    echo ">>> Stage 3 Complete!"
fi

echo ""
echo "============================================"
echo "All requested stages complete!"
echo "============================================"
