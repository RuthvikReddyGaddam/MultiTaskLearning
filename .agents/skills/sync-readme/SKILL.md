---
name: sync-readme
description: Guides updating and auditing README.md whenever codebase features, contracts, configs, or scripts are modified. Use before concluding tasks to ensure documentation stays synchronized.
---

# Sync README Skill

Use this skill whenever you modify, add, or refactor components in this multi-task histopathology repository to ensure `README.md` stays 100% synchronized with the code.

---

## When to Activate
Activate this skill whenever you have:
1. Added or modified a CLI script in `scripts/` (e.g. new arguments, flags, default paths).
2. Changed YAML configuration options in `configs/`.
3. Updated model architectures, decoder blocks, or prediction head signatures in `models/`.
4. Modified dataset loaders, batch ratios, augmentations, or sampler logic in `datasets/`.
5. Adjusted loss functions, weighting, or regularization terms in `losses/`.
6. Added or updated post-processing or evaluation routines in `utils/` or `metrics/`.

---

## Step-by-Step Workflow

### Step 1: Identify Changed Files
Run:
```bash
git status -s
git diff --name-only
```
Categorize which subsystems were impacted:
- **Configs / Scripts**: Check if CLI invocation examples in `README.md` need updating.
- **Datasets**: Check if batch size, 2:1:1 ratio, task IDs, or label schemas (`target_4class`, `class_map`) changed.
- **Models / Losses**: Check if returned prediction keys (`preds["..."]`) or loss terms changed.
- **New Files**: Check if `README.md`'s Directory Map needs a new entry.

### Step 2: Audit Against README.md Sections
Review each section in `README.md` against your code changes:

| Section in README.md | Check |
| :--- | :--- |
| **1. Codebase Directory Map** | Are all newly added modules or files represented? |
| **2. Invariants & Data Contracts** | Did any batch shape, ratio, shuffling behavior, or label indices change? |
| **3. Environment & Execution Commands** | Are the example commands exact drop-in copy-pasteable strings with the right Python binary? |
| **4. Agent Guidelines** | Were any new constraints or invariants introduced? |

### Step 3: Edit README.md
Update the affected sections using `edit_file` to keep the descriptions concise, accurate, and structured.

### Step 4: Verification
Confirm that markdown links and bash commands are valid and syntax-highlighted.

