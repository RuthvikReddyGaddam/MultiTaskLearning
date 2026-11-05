#!/usr/bin/env bash
# ==============================================================================
# MultiTaskLearning - Cluster Training Script
# Supports: Standalone servers / Cluster nodes (CUDA_VISIBLE_DEVICES, torchrun, logs)
# ==============================================================================
set -euo pipefail

# 1. Project directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================================="
echo " Starting Multi-Task Training Job"
echo " Date: $(date +'%Y-%m-%d %H:%M:%S')"
echo " Working Dir: ${PROJECT_ROOT}"
echo "=========================================================="

# 2. Python / Conda / Virtualenv environment activation
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

# 3. Hardware & environment settings
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="${1:-configs/multitask/multitask_sdenetpointgradexp_512.yaml}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs/train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "Config: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Output Log: ${LOG_FILE}"
echo "----------------------------------------------------------"

# 4. Run Training
# Single-GPU execution:
python scripts/train_multitask.py --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"

echo "=========================================================="
echo " Training Finished Successfully at $(date +'%Y-%m-%d %H:%M:%S')"
echo " Logs written to ${LOG_FILE}"
echo "=========================================================="

