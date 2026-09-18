#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
source ${REPO_ROOT}/videorepa/bin/activate
export PATH="${REPO_ROOT}/videorepa/bin:$PATH"
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=4

cd ${REPO_ROOT}/finetune

torchrun --nproc_per_node=8 --master_port=$((12000 + $RANDOM % 20000)) \
    pretrain_projector.py \
    --model_path ${CKPT_DIR}/cogvideox-2b \
    --data_csv ${REPO_ROOT}/finetune/openvid/openvid_1w6.csv \
    --data_root ${REPO_ROOT}/finetune \
    --output_path ${REPO_ROOT}/finetune/pretrained_projector.pth \
    --align_layer 18 \
    --align_dims 768 \
    --projector_dim 2048 \
    --epochs 10 \
    --lr 1e-3 \
    --batch_size 1
