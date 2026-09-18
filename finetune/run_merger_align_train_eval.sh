#!/usr/bin/env bash
source /efs/zixianhuang/VideoREPA/videorepa/bin/activate
export PATH="/efs/zixianhuang/VideoREPA/videorepa/bin:$PATH"
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

BASE_PATH="/efs/zixianhuang/VideoREPA"
ORIG_MODEL="/efs/zixianhuang/ckpt/cogvideox-2b"
INFERENCE_DIR="${BASE_PATH}/inference"
EVAL_V1_DIR="${BASE_PATH}/evaluation/videophy"
VIDEOPHY_V1_CKPT="/efs/zixianhuang/ckpt/videocon_physics"
EXP_NAME="merger_align"
EXP_COMMENT="exp_merger_align_L18_3w2_4ep"

echo "[$(date)] ============ Training ============"
cd "${BASE_PATH}/finetune"

accelerate launch --main_process_port $((12000 + $RANDOM % 20000)) --num_processes 8 --config_file accelerate_config.yaml train.py \
    --model_path "${ORIG_MODEL}" \
    --model_name "cogvideox-t2v-merger-align" \
    --model_type "t2v" \
    --training_type "sft" \
    --output_dir "${BASE_PATH}/finetune/output_dir" \
    --report_to "none" \
    --train_data_path ${BASE_PATH}/finetune/openvid/openvid_3w2.csv \
    --data_root "${BASE_PATH}/finetune" \
    --train_resolution "49x480x720" \
    --caption_column "prompt.txt" \
    --video_column "videos.txt" \
    --train_epochs 4 \
    --seed 42 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --mixed_precision "bf16" \
    --num_workers 8 \
    --pin_memory True \
    --nccl_timeout 1800 \
    --checkpointing_steps 400 \
    --checkpointing_limit 2 \
    --do_validation true \
    --validation_dir "${BASE_PATH}/finetune/validation" \
    --validation_steps 400 \
    --validation_prompts "prompts.txt" \
    --gen_fps 8 \
    --loss merger_align \
    --align_models VideoMAEv2 \
    --align_layer 18 \
    --align_dims 768 \
    --proj_coeff 1.0 \
    --margin 0.1 \
    --comment "${EXP_COMMENT}" \
    --learning_rate 2e-6

echo "[$(date)] ============ Training Done ============"

# Find experiment directory
EXP_DIR=$(ls -dt ${BASE_PATH}/finetune/output_dir_*${EXP_COMMENT}* 2>/dev/null | head -1)
if [ -z "${EXP_DIR}" ]; then
    echo "[$(date)] ERROR: Cannot find experiment directory!"
    exit 1
fi
echo "[$(date)] Experiment dir: ${EXP_DIR}"

# ============ Stage 1: Merge Checkpoint (use latest checkpoint) ============
echo "[$(date)] Stage 1: Merging checkpoint..."
CHECKPOINT_STEP=$(ls -d ${EXP_DIR}/checkpoint-* 2>/dev/null | sed 's#.*/##' | sort -t- -k2 -n | tail -1)
echo "[$(date)] Using checkpoint: ${CHECKPOINT_STEP}"
if [ -z "${CHECKPOINT_STEP}" ]; then
    echo "[$(date)] ERROR: No checkpoint found!"
    exit 1
fi
if [ ! -d "${EXP_DIR}/transformer" ]; then
    cd "${EXP_DIR}/${CHECKPOINT_STEP}"
    python zero_to_fp32.py ./ ../transformer --safe_serialization
    if [ -f "../transformer/model.safetensors.index.json" ]; then
        mv ../transformer/model.safetensors.index.json ../transformer/diffusion_pytorch_model.safetensors.index.json
    fi
    cp "${ORIG_MODEL}/transformer/config.json" "../transformer/config.json"
fi

INFER_MODEL_DIR="${BASE_PATH}/infer_model/${EXP_NAME}"
rm -rf "${INFER_MODEL_DIR}"
mkdir -p "${INFER_MODEL_DIR}"
for item in "${ORIG_MODEL}"/*; do
    bn=$(basename "$item")
    [ "$bn" != "transformer" ] && cp -r "$item" "${INFER_MODEL_DIR}/${bn}"
done
cp -r "${EXP_DIR}/transformer" "${INFER_MODEL_DIR}/transformer"
echo "[$(date)] Inference model ready."

# ============ Stage 2: Generate Videos ============
OUTPUT_VIDEO_DIR="${INFERENCE_DIR}/output_dir/${EXP_NAME}_videophy"
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

# ============ Stage 3: Evaluate ============
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
echo "========== VideoPhy v1 Results: ${EXP_NAME} =========="
cat sa_*.txt 2>/dev/null; cat pc_*.txt 2>/dev/null
echo ""
echo "[$(date)] All done!"
