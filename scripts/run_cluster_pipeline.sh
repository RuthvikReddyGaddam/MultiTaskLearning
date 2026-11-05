#!/usr/bin/env bash
# ==============================================================================
# MultiTaskLearning - Full Pipeline Runner (Train -> Evaluate)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/multitask/multitask_sdenetpointgradexp_512.yaml}"

echo ">>> [1/2] Launching Cluster Training..."
bash "${SCRIPT_DIR}/run_cluster_train.sh" "${CONFIG}"

echo ">>> [2/2] Launching Cluster Evaluation..."
bash "${SCRIPT_DIR}/run_cluster_eval.sh" "${CONFIG}"

echo ">>> Pipeline Completed Successfully!"

