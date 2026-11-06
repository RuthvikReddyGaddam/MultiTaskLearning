# AI Agent Guidelines & Repository Rules

This file is automatically loaded as a workspace rule for all AI agents working in this repository. Follow these rules strictly:

---

## 1. Mandatory README Synchronization
Whenever you add, modify, or refactor any code in this repository:
1. **Audit Impact**: Check if your modification changes:
   - CLI flags, script arguments, or execution commands (`scripts/`)
   - YAML config options or defaults (`configs/`)
   - Batch sampling, dataset splits, task ratios, or tensor schemas (`datasets/`)
   - Model prediction head signatures, keys, or output shapes (`models/`)
   - Loss terms, regularization components, or loss weights (`losses/`)
   - Metric definitions or evaluation flows (`metrics/`, `utils/`)
2. **Update README.md**: You **MUST** update `README.md` before claiming a task is complete. Keep the codebase map, architectural contracts, and execution commands synchronized with the actual implementation.
3. **Use the `sync-readme` Skill**: Consult `.agents/skills/sync-readme/SKILL.md` for the exact verification checklist.

---

## 2. Invariants & Architectural Rules
1. **Batch Sampling Contract**:
   - Batch size must remain a multiple of 8 (e.g. 8, 16, 24).
   - The 2:1:1 ratio ($B/2$ GLaS, $B/4$ MoNuSeg, $B/4$ MoNuSAC) is strict.
   - Patches are randomly shuffled within the batch. Never use hardcoded positional slices like `batch[:4]`; always use boolean task masks (`is_gland = task_ids == 0`, `is_monuseg = task_ids == 1`, `is_monusac = task_ids == 2`).
2. **Deterministic FP32**:
   - Maintain pure FP32 numerical stability; avoid adding unverified mixed precision (`torch.cuda.amp`) that causes gradient underflow on rare gap pixels.
3. **Environment**:
   - Always run Python commands using the project environment:
     `/usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python`
   - Prefix commands with `PYTHONPATH=.` when executing from the repository root.
4. **Verification Before Completion**:
   - Always run the relevant test suite (`PYTHONPATH=. /usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python -m unittest ...`) before concluding your turn to verify that modifications do not break existing contracts.

