#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
# Auto-launch TRD training after the current cosine_similarity training finishes.
# Usage: nohup bash auto_launch_TRD.sh > auto_launch_TRD.log 2>&1 &

CURRENT_PID=1651800  # PID of the current training launch script
LOG_FILE="${REPO_ROOT}/finetune/train_frozen_projector.log"
TRD_SCRIPT="${REPO_ROOT}/finetune/scripts/multigpu_VideoREPA_2B_sft_frozen_projector_TRD.sh"
TRD_LOG="${REPO_ROOT}/finetune/train_frozen_projector_TRD.log"

echo "[$(date)] Monitoring PID ${CURRENT_PID} for completion..."

# Wait for current training to finish
while true; do
    if ! ps -p ${CURRENT_PID} > /dev/null 2>&1; then
        echo "[$(date)] PID ${CURRENT_PID} is no longer running."
        break
    fi
    # Also check if the training log shows completion
    if grep -q "Training complete\|Training steps: 100%" "${LOG_FILE}" 2>/dev/null; then
        echo "[$(date)] Training complete detected in log."
        break
    fi
    sleep 60
done

# Wait a bit for GPU memory to be released
echo "[$(date)] Waiting 30s for GPU memory release..."
sleep 30

# Check if training actually completed successfully (not crashed)
if grep -q "Training steps:.*100%\|Saved checkpoint" "${LOG_FILE}" 2>/dev/null; then
    echo "[$(date)] Previous training completed successfully. Launching TRD training..."
    cd ${REPO_ROOT}/finetune
    nohup bash "${TRD_SCRIPT}" > "${TRD_LOG}" 2>&1 &
    TRD_PID=$!
    echo "[$(date)] TRD training launched with PID: ${TRD_PID}"
    echo "[$(date)] Log: ${TRD_LOG}"
else
    echo "[$(date)] WARNING: Previous training may have crashed. Check log: ${LOG_FILE}"
    echo "[$(date)] Launching TRD training anyway..."
    cd ${REPO_ROOT}/finetune
    nohup bash "${TRD_SCRIPT}" > "${TRD_LOG}" 2>&1 &
    TRD_PID=$!
    echo "[$(date)] TRD training launched with PID: ${TRD_PID}"
fi

echo "[$(date)] Auto-launch script done."
