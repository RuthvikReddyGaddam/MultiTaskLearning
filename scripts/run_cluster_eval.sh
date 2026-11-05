#!/usr/bin/env bash
# ==============================================================================
# MultiTaskLearning - Cluster Evaluation Script
# Evaluates checkpoint across GLaS, MoNuSeg, and MoNuSAC
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================================="
echo " Starting Multi-Task Evaluation Job"
echo " Date: $(date +'%Y-%m-%d %H:%M:%S')"
echo " Working Dir: ${PROJECT_ROOT}"
echo "=========================================================="

ENV_NAME="${CONDA_DEFAULT_ENV:-multitask}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}" || echo "[Info] Using current conda environment: $(which python)"
elif [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

echo "Python Executable: $(which python)"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="${1:-configs/multitask/multitask_sdenetpointgradexp_512.yaml}"
CHECKPOINT_PATH="${2:-}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs/eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

CMD="python scripts/evaluate_multitask.py --config ${CONFIG_PATH}"
if [ -n "${CHECKPOINT_PATH}" ]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT_PATH}"
fi

echo "Executing: ${CMD}"
echo "Output Log: ${LOG_FILE}"
echo "----------------------------------------------------------"

eval "${CMD}" 2>&1 | tee "${LOG_FILE}"

echo "=========================================================="
echo " Evaluation Finished at $(date +'%Y-%m-%d %H:%M:%S')"
echo " Logs written to ${LOG_FILE}"
echo "=========================================================="

