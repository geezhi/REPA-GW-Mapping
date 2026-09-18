#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
# VideoPhy2 评测 pipeline
# 使用方法:
#   1. 修改 INPUT_CSV 为你的实验名称对应的 csv
#   2. 确保 CHECKPOINT 指向已下载的 videophy2 模型
set -e

INPUT_CSV='VideoREPA_2B.csv'
CHECKPOINT="${CKPT_DIR}/videophy2"

OUTPUT_FOLDER="./output_dir/${INPUT_CSV%.csv}"

INPUT_CSV="./csv_file/$INPUT_CSV"

mkdir -p "$OUTPUT_FOLDER"

echo "output_dir:$OUTPUT_FOLDER"

python inference.py --input_csv $INPUT_CSV  \
--checkpoint "$CHECKPOINT" \
--output_csv "$OUTPUT_FOLDER/output_sa.csv" --task sa

echo "output_dir:$OUTPUT_FOLDER"  

python inference.py --input_csv $INPUT_CSV  \
--checkpoint "$CHECKPOINT" \
--output_csv "$OUTPUT_FOLDER/output_pc.csv" --task pc

echo "output_dir:$OUTPUT_FOLDER"

cp calculate_mean.py "$OUTPUT_FOLDER/calculate_mean.py"
cd "$OUTPUT_FOLDER"

echo "output_dir:$OUTPUT_FOLDER"

python calculate_mean.py