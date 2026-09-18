#!/usr/bin/env bash
source /efs/zixianhuang/VideoREPA/videorepa/bin/activate
export PATH="/efs/zixianhuang/VideoREPA/videorepa/bin:$PATH"

BASE_PATH="/efs/zixianhuang/VideoREPA"
INFERENCE_DIR="${BASE_PATH}/inference"
EVAL_V1_DIR="${BASE_PATH}/evaluation/videophy"
VIDEOPHY_V1_CKPT="/efs/zixianhuang/ckpt/videocon_physics"
EXP_NAME="projector_finetune_lr0.1"
INFER_MODEL_DIR="${BASE_PATH}/infer_model/${EXP_NAME}"
OUTPUT_VIDEO_DIR="${INFERENCE_DIR}/output_dir/${EXP_NAME}_videophy"

# Stage 2: Generate
echo "[$(date)] Stage 2: Generating 344 videos..."
cd "${INFERENCE_DIR}"
mkdir -p "${OUTPUT_VIDEO_DIR}"

for gpu_id in $(seq 0 7); do
    (
        for retry in $(seq 1 5); do
            find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
            generated=$(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" 2>/dev/null | wc -l)
            [ "${generated}" -ge 344 ] && break
            env CUDA_VISIBLE_DEVICES=$gpu_id python generate.py \
                --input_file "./videophy.txt" --output_dir "${OUTPUT_VIDEO_DIR}" \
                --model_path "${INFER_MODEL_DIR}" --generate_type t2v --upsampled --seed 42
            [ $? -eq 0 ] && break
            find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
            sleep 10
        done
    ) &
    sleep 2
done
wait
find "${OUTPUT_VIDEO_DIR}" -name "*.lock" -type d -exec rmdir {} + 2>/dev/null || true
echo "[$(date)] Generated: $(find "${OUTPUT_VIDEO_DIR}" -name "*.mp4" | wc -l)/344"

# Stage 3: Evaluate
echo "[$(date)] Stage 3: Evaluating..."
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
python videocon/training/pipeline_video/entailment_inference.py --input_csv "${OUTPUT_FOLDER}/sa_testing.csv" --output_csv "${OUTPUT_FOLDER}/videocon_physics_sa_testing.csv" --checkpoint "${VIDEOPHY_V1_CKPT}"
python videocon/training/pipeline_video/entailment_inference.py --input_csv "${OUTPUT_FOLDER}/physics_testing.csv" --output_csv "${OUTPUT_FOLDER}/videocon_physics_pc_testing.csv" --checkpoint "${VIDEOPHY_V1_CKPT}"
cp calculate_mean.py videophy.txt reference.csv "${OUTPUT_FOLDER}/"
cd "${OUTPUT_FOLDER}" && python calculate_mean.py

echo ""
echo "========== VideoPhy v1 Results: Projector Finetune (lr_scale=0.1) =========="
cat sa_*.txt 2>/dev/null; cat pc_*.txt 2>/dev/null
echo "[$(date)] Done!"
