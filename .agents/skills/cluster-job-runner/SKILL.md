---
name: cluster-job-runner
description: Write cluster training and evaluation bash scripts (supporting standalone multi-GPU remote servers, torchrun, nohup/tmux, conda/venv environments) and sync/push the repository changes to the GitHub remote repo. Use whenever the user asks to create or run cluster/remote training scripts, evaluation scripts, automate pipeline runs, or push repository code/scripts to the remote git repo.
---

# Cluster Job Runner & Remote Sync

This skill guides the agent in generating robust cluster/server execution bash scripts for training and evaluation workloads (such as PyTorch `torchrun`, single/multi-GPU `CUDA_VISIBLE_DEVICES`, nohup/tmux background jobs, and environment activation) and safely committing & pushing repository changes to the configured GitHub remote.

## 1. Workflow Overview

1. **Environment & Hardware Inspection**: Detect conda/virtual environments, available Python packages, configs, and target hardware (GPUs, CPUs).
2. **Script Generation**:
   - Training script (`scripts/run_train.sh` or customized name)
   - Evaluation script (`scripts/run_eval.sh` or customized name)
   - All-in-one / pipeline runner (`scripts/run_all.sh`)
3. **Script Quality & Robustness Standards**:
   - Proper bash safety flags (`set -euo pipefail`).
   - Absolute / project root path resolution (`cd "$(dirname "$0")/.." || exit 1`).
   - Virtual environment / conda activation logic with fallbacks.
   - Logging redirection (`tee` / timestamped log files in `logs/`).
   - GPU selection (`CUDA_VISIBLE_DEVICES` or `torchrun --nproc_per_node`).
   - Config argument propagation (`--config`, `--checkpoint`, overrides).
4. **Local Verification**:
   - Verify script syntax (`bash -n script.sh` or dry-run checks).
   - Ensure executable permissions (`chmod +x script.sh`).
5. **Git Commit & Remote Push**:
   - Inspect status (`git status`).
   - Stage changes (`git add .`).
   - Create clean, descriptive commit message.
   - Push to GitHub remote (`git push origin <branch>`).

---

## 2. Generating Bash Scripts

### A. Training Script Template (`scripts/run_train.sh`)

```bash
#!/usr/bin/env bash
# ==============================================================================
# Cluster Training Script
# ==============================================================================
set -euo pipefail

# 1. Project Directory Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] Starting Training Run ==="
echo "Project root: ${PROJECT_ROOT}"

# 2. Environment Activation (adjust conda env name / path as needed)
ENV_NAME="${CONDA_DEFAULT_ENV:-multitask}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}" || echo "Warning: Conda env ${ENV_NAME} could not be activated, using current environment."
elif [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

# 3. Hardware & Execution Configuration
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

CONFIG_PATH="${1:-configs/multitask/multitask_sdenetpointgradexp_512.yaml}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs/train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "Using Config: ${CONFIG_PATH}"
echo "Using GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "Logs saved to: ${LOG_FILE}"

# 4. Execute Training (Single GPU or multi-GPU torchrun)
# For single-GPU:
python scripts/train_multitask.py --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"

# For multi-GPU distributed data parallel (torchrun):
# NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
# torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" scripts/train_multitask.py --config "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] Training Finished Successfully ==="
```

### B. Evaluation Script Template (`scripts/run_eval.sh`)

```bash
#!/usr/bin/env bash
# ==============================================================================
# Cluster Evaluation Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] Starting Evaluation Run ==="
echo "Project root: ${PROJECT_ROOT}"

# Environment Activation
ENV_NAME="${CONDA_DEFAULT_ENV:-multitask}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}" || echo "Using active python environment."
elif [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/venv/bin/activate"
fi

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

echo "Running Command: ${CMD}"
echo "Logs saved to: ${LOG_FILE}"

eval "${CMD}" 2>&1 | tee "${LOG_FILE}"

echo "=== [$(date +'%Y-%m-%d %H:%M:%S')] Evaluation Completed ==="
```

### C. Pipeline Runner (`scripts/run_pipeline.sh`)

Allows running train followed by eval, or launching detached in tmux/nohup:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-configs/multitask/multitask_sdenetpointgradexp_512.yaml}"

echo ">>> Stage 1: Running Training"
bash "${SCRIPT_DIR}/run_train.sh" "${CONFIG}"

echo ">>> Stage 2: Running Evaluation"
bash "${SCRIPT_DIR}/run_eval.sh" "${CONFIG}"
```

---

## 3. Git Commit and Remote Push Process

When the user requests updating the repo or pushing scripts to GitHub:

1. **Verify Git Remote**:
   Confirm remote configuration (`git remote -v`). Ensure destination is correct (e.g. `origin` -> `git@github.com:...`).
2. **Review Untracked & Modified Files**:
   Run `git status` to verify modified and untracked files. Do not commit unnecessary large checkpoints (`*.pt`, `*.pth`), virtualenv directories, or temporary scratch logs unless explicitly intended. Check `.gitignore` if needed.
3. **Stage Changes**:
   ```bash
   git add .
   ```
4. **Commit with Structured Message**:
   Use conventional, clear commit message:
   ```bash
   git commit -m "feat(cluster): add cluster training and evaluation bash scripts"
   ```
5. **Push to Remote**:
   ```bash
   git push origin main
   ```
   *(Or the current active branch via `git push origin HEAD`)*
6. **Confirm Remote Status**:
   Ensure `git status` shows branch is clean and up-to-date with `origin/main`.

