#!/usr/bin/env bash
source /efs/zixianhuang/VideoREPA/videorepa/bin/activate
export PATH="/efs/zixianhuang/VideoREPA/videorepa/bin:$PATH"
export NCCL_DEBUG=ERROR
export OMP_NUM_THREADS=4

cd /efs/zixianhuang/VideoREPA/finetune

torchrun --nproc_per_node=8 --master_port=$((12000 + $RANDOM % 20000)) \
    pretrain_projector.py \
    --model_path /efs/zixianhuang/ckpt/cogvideox-2b \
    --data_csv /efs/zixianhuang/VideoREPA/finetune/openvid/openvid_1w6.csv \
    --data_root /efs/zixianhuang/VideoREPA/finetune \
    --output_path /efs/zixianhuang/VideoREPA/finetune/pretrained_projector.pth \
    --align_layer 18 \
    --align_dims 768 \
    --projector_dim 2048 \
    --epochs 10 \
    --lr 1e-3 \
    --batch_size 1
