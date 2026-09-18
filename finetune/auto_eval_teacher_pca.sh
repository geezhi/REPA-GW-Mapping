#!/usr/bin/env bash
# Auto-evaluate teacher PCA experiment after training finishes.
# Usage: nohup bash auto_eval_teacher_pca.sh > auto_eval_teacher_pca.log 2>&1 &

TRAIN_LOG="/efs/zixianhuang/VideoREPA/finetune/train_teacher_pca.log"
TRAIN_PID=1223480

echo "[$(date)] Monitoring teacher PCA training (PID ${TRAIN_PID}) for completion..."

while true; do
    if ! ps -p ${TRAIN_PID} > /dev/null 2>&1; then
        echo "[$(date)] Training PID no longer running."
        break
    fi
    if strings "${TRAIN_LOG}" 2>/dev/null | grep -q "Training steps: 100%"; then
        echo "[$(date)] Training 100% detected."
        break
    fi
    sleep 60
done

echo "[$(date)] Waiting 30s for GPU release..."
sleep 30

# Check that checkpoint exists
EXP_DIR="/efs/zixianhuang/VideoREPA/finetune/output_dir_cogvideox-t2v-align_cosine_similarity_exp_teacher_pca_multilayer_3w2_4ep"
if [ ! -d "${EXP_DIR}/checkpoint-4000" ]; then
    echo "[$(date)] WARNING: checkpoint-4000 not found. Looking for latest checkpoint..."
    LATEST_CKPT=$(ls -d ${EXP_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "${LATEST_CKPT}" ]; then
        echo "[$(date)] ERROR: No checkpoint found in ${EXP_DIR}"
        exit 1
    fi
    CHECKPOINT_STEP=$(basename ${LATEST_CKPT})
    echo "[$(date)] Using ${CHECKPOINT_STEP}"
else
    CHECKPOINT_STEP="checkpoint-4000"
fi

echo "[$(date)] Starting evaluation pipeline..."

BASE_PATH="/efs/zixianhuang/VideoREPA"
ORIG_MODEL="/efs/zixianhuang/ckpt/cogvideox-2b"
INFERENCE_DIR="${BASE_PATH}/inference"
EVAL_V1_DIR="${BASE_PATH}/evaluation/videophy"
VIDEOPHY_V1_CKPT="/efs/zixianhuang/ckpt/videocon_physics"
EXP_NAME="teacher_pca_multilayer"
INFER_MODEL_DIR="${BASE_PATH}/infer_model/${EXP_NAME}"
OUTPUT_VIDEO_DIR="${INFERENCE_DIR}/output_dir/${EXP_NAME}_videophy"

source /efs/zixianhuang/VideoREPA/videorepa/bin/activate
export PATH="/efs/zixianhuang/VideoREPA/videorepa/bin:$PATH"

# ============ Stage 1: Merge Checkpoint ============
echo "[$(date)] Stage 1: Merging checkpoint..."
if [ ! -d "${EXP_DIR}/transformer" ]; then
    cd "${EXP_DIR}/${CHECKPOINT_STEP}"
    python zero_to_fp32.py ./ ../transformer --safe_serialization
    if [ -f "../transformer/model.safetensors.index.json" ]; then
        mv ../transformer/model.safetensors.index.json ../transformer/diffusion_pytorch_model.safetensors.index.json
    fi
    cp "${ORIG_MODEL}/transformer/config.json" "../transformer/config.json"
    echo "[$(date)] Checkpoint merged."
else
    echo "[$(date)] Transformer already exists, skipping."
fi

if [ ! -d "${INFER_MODEL_DIR}" ]; then
    mkdir -p "${INFER_MODEL_DIR}"
    for item in "${ORIG_MODEL}"/*; do
        bn=$(basename "$item")
        if [ "$bn" != "transformer" ]; then
            cp -r "$item" "${INFER_MODEL_DIR}/${bn}"
        fi
    done
    cp -r "${EXP_DIR}/transformer" "${INFER_MODEL_DIR}/transformer"
    echo "[$(date)] Inference model created."
fi

# ============ Stage 2: Generate Videos ============
echo "[$(date)] Stage 2: Generating 344 videos..."
cd "${INFERENCE_DIR}"
mkdir -p "${OUTPUT_VIDEO_DIR}"

NUM_GPUS=8
MAX_RETRIES=5
TOTAL_PROMPTS=344

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    (
        retry=0
        while [ $retry -lt ${MAX_RETRIES} ]; do
            find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
            generated=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
            if [ "${generated}" -ge "${TOTAL_PROMPTS}" ]; then break; fi
            env CUDA_VISIBLE_DEVICES=$gpu_id python generate.py \
                --input_file "./videophy.txt" \
                --output_dir "${OUTPUT_VIDEO_DIR}" \
                --model_path "${INFER_MODEL_DIR}" \
                --generate_type t2v \
                --upsampled --seed 42
            exit_code=$?
            retry=$((retry + 1))
            if [ $exit_code -ne 0 ]; then
                find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
                sleep 10
            else break; fi
        done
    ) &
    sleep 2
done
wait
find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
FINAL_COUNT=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
echo "[$(date)] Stage 2 done: ${FINAL_COUNT}/344 videos."

# ============ Stage 3: Evaluate ============
echo "[$(date)] Stage 3: VideoPhy v1 AutoEval..."
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

python utils/prepare_data.py --input_csv "${CSV_FILE}" --output_folder "${OUTPUT_FOLDER}"

echo "[$(date)] Running SA evaluation..."
python videocon/training/pipeline_video/entailment_inference.py \
    --input_csv "${OUTPUT_FOLDER}/sa_testing.csv" \
    --output_csv "${OUTPUT_FOLDER}/videocon_physics_sa_testing.csv" \
    --checkpoint "${VIDEOPHY_V1_CKPT}"

echo "[$(date)] Running PC evaluation..."
python videocon/training/pipeline_video/entailment_inference.py \
    --input_csv "${OUTPUT_FOLDER}/physics_testing.csv" \
    --output_csv "${OUTPUT_FOLDER}/videocon_physics_pc_testing.csv" \
    --checkpoint "${VIDEOPHY_V1_CKPT}"

cp calculate_mean.py "${OUTPUT_FOLDER}/"
cp videophy.txt "${OUTPUT_FOLDER}/"
cp reference.csv "${OUTPUT_FOLDER}/"
cd "${OUTPUT_FOLDER}"
python calculate_mean.py

echo ""
echo "========== VideoPhy v1 Results: Teacher PCA Multilayer =========="
cat sa_*.txt 2>/dev/null || true
cat pc_*.txt 2>/dev/null || true
echo ""
echo "[$(date)] All done!"
