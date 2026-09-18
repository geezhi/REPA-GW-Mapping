#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
while [ ! -d "${REPO_ROOT}/finetune" ] && [ "${REPO_ROOT}" != "/" ]; do REPO_ROOT="$(dirname "${REPO_ROOT}")"; done
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
# Auto-launch teacher PCA training after exp3 evaluation finishes.
# Usage: nohup bash auto_launch_teacher_pca.sh > auto_launch_teacher_pca.log 2>&1 &

EVAL_LOG="${REPO_ROOT}/eval_frozen_projector_exp3.log"
NEXT_SCRIPT="${REPO_ROOT}/finetune/scripts/multigpu_VideoREPA_2B_sft_teacher_pca.sh"
NEXT_LOG="${REPO_ROOT}/finetune/train_teacher_pca.log"

echo "[$(date)] Monitoring exp3 evaluation for completion..."

while true; do
    if grep -q "All evaluations complete" "${EVAL_LOG}" 2>/dev/null; then
        echo "[$(date)] Exp3 evaluation complete detected."
        break
    fi
    if ! pgrep -f "eval_frozen_projector_videophy.sh.*--exp 3" > /dev/null 2>&1; then
        echo "[$(date)] Exp3 evaluation process no longer running."
        break
    fi
    sleep 30
done

echo "[$(date)] Waiting 30s for GPU memory release..."
sleep 30

echo "[$(date)] Launching teacher PCA training..."
cd ${REPO_ROOT}/finetune
nohup bash "${NEXT_SCRIPT}" > "${NEXT_LOG}" 2>&1 &
NEXT_PID=$!
echo "[$(date)] Teacher PCA training launched with PID: ${NEXT_PID}"
echo "[$(date)] Log: ${NEXT_LOG}"
echo "[$(date)] Done."
