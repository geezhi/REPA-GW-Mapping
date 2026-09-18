#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
# 下载视觉基础模型权重 (Vision Foundation Model checkpoints)
# 参考: https://github.com/aHapBean/VideoREPA

set -e
cd ${CKPT_DIR}

# 1. VideoMAEv2 - vit_base distilled from giant (K710 pretrained)
# 来源: https://github.com/OpenGVLab/VideoMAEv2
mkdir -p VideoMAEv2
if [ ! -f "VideoMAEv2/vit_b_k710_dl_from_giant.pth" ]; then
    echo "==> Downloading VideoMAEv2 (vit_b_k710_dl_from_giant)..."
    wget -O VideoMAEv2/vit_b_k710_dl_from_giant.pth \
        "https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/videomaev2/vit_b_k710_dl_from_giant.pth"
fi

# 2. (可选) VideoMAE - vit_base K400 pretrained
mkdir -p VideoMAE
if [ ! -f "VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth" ]; then
    echo "==> Downloading VideoMAE (K400 pretrained base)..."
    wget -O VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth \
        "https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/videomae/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth"
fi

echo "==> Done! Downloaded VFM checkpoints to ${CKPT_DIR}/"
ls -lh VideoMAEv2/ VideoMAE/ 2>/dev/null
