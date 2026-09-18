#!/usr/bin/env bash
# Auto-launch temporal_diff_only training after TRD training finishes.
# Usage: nohup bash auto_launch_temporal_diff.sh > auto_launch_temporal_diff.log 2>&1 &

TRD_LOG="/efs/zixianhuang/VideoREPA/finetune/train_frozen_projector_TRD.log"
NEXT_SCRIPT="/efs/zixianhuang/VideoREPA/finetune/scripts/multigpu_VideoREPA_2B_sft_frozen_projector_temporal_diff.sh"
NEXT_LOG="/efs/zixianhuang/VideoREPA/finetune/train_frozen_projector_temporal_diff.log"

echo "[$(date)] Monitoring TRD training for completion..."

# Wait for TRD training to finish
while true; do
    # Check if any train.py process with TRD is running
    if ! pgrep -f "train.py.*exp_frozen_projector_TRD" > /dev/null 2>&1; then
        echo "[$(date)] TRD training process no longer running."
        break
    fi
    # Also check log for completion
    if grep -q "Training steps: 100%" "${TRD_LOG}" 2>/dev/null; then
        echo "[$(date)] TRD training 100% detected in log."
        break
    fi
    sleep 60
done

# Wait for GPU memory release
echo "[$(date)] Waiting 30s for GPU memory release..."
sleep 30

echo "[$(date)] Launching temporal_diff_only training..."
cd /efs/zixianhuang/VideoREPA/finetune
nohup bash "${NEXT_SCRIPT}" > "${NEXT_LOG}" 2>&1 &
NEXT_PID=$!
echo "[$(date)] Temporal diff training launched with PID: ${NEXT_PID}"
echo "[$(date)] Log: ${NEXT_LOG}"
echo "[$(date)] Done."
