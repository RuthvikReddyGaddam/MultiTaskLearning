# Multi-Task Histopathology Learning (SDENet-PointGrad)

Dedicated, streamlined multi-task deep learning framework across three benchmark histopathology datasets:
- **GLaS**: Gland instance & semantic segmentation (colon histology)
- **MoNuSeg**: Multi-organ cell nuclei segmentation
- **MoNuSAC**: Cell nuclei segmentation + 4-class cell phenotype classification (Epithelial, Lymphocyte, Neutrophil, Macrophage)

---

## 1. Codebase Directory Map

| Directory | Responsibility | Primary Files |
| :--- | :--- | :--- |
| `configs/` | YAML configuration files for models, datasets, training, and losses. | `configs/multitask/multitask_sdenetpointgradexp_512.yaml`, `configs/multitask_sdenetpointgradexp_512.yaml` |
| `datasets/` | Multi-task datasets, samplers, collators, and geometric augmentations. | `multitask_dataset.py`, `multitask_sampler.py`, `multitask_collate.py`, `multitask_augment.py`, `monusac_dataset.py` (`ResampledMoNuSACDataset`) |
| `models/` | Dual encoder (image + geometry), fusion decoder, and multi-task prediction heads. | `models/SDENetPointGrad/MultiTask/Model.py`, `.../Decoder/PredictionHead.py` |
| `losses/` | Decoupled 4-class Youden's J-regularization, CrossEntropy, and Generalized Dice loss. | `multitask_loss.py`, `j_regularization_loss.py` |
| `training/` | Training loop, checkpointing (`best.pt`, `latest.pt`), and metric tracking. | `training/trainer_multitask.py` |
| `metrics/` | Instance metrics (Dice, IoU, AJI, PQ) and phenotype multiclass metrics. | `metrics/monusac_training_metrics.py` |
| `utils/` | Watershed post-processing, Bottom-Hat instance extraction, lumen reclamation. | `utils/test_utils.py` (`class4_to_instances`, `extract_model_outputs`) |
| `scripts/` | Main entry points for training, evaluation, cluster runs, and batch inspection. | `train_multitask.py`, `eval_multitask_inference.py`, `test_glas.py`, `test_monusac.py` |
| `tests/` | Unit and end-to-end integration test suites. | `test_monusac_stratified_sampling.py`, `test_multitask_sampler_8batch.py`, `test_trainer_multitask_8batch.py` |

---

## 2. Invariants & Data Contracts

### A. Batch Composition & In-Batch Shuffling
- **Batch Size Constraint**: Must be a multiple of 8 (default: `8`).
- **Exact 2:1:1 Task Allocation**:
  - 4 GLaS patches (`task_id = 0`)
  - 2 MoNuSeg patches (`task_id = 1`)
  - 2 MoNuSAC patches (`task_id = 2`)
- **Randomized Order**: Within each batch of 8, patches are **randomly shuffled** during training via `EqualTaskBatchSampler`.
- **Mask-Based Routing**: **Never assume fixed positional slices** (e.g., `batch[:4]`). Always route using boolean indicator masks:
  ```python
  is_gland   = (task_ids == 0)
  is_monuseg = (task_ids == 1)
  is_monusac = (task_ids == 2)
  is_cell    = is_monuseg | is_monusac
  ```

### B. Target Schemas
1. **4-Class Semantic Segmentation Target (`target_4class`)**:
   - `0`: Background
   - `1`: Instance Interior (Core)
   - `2`: Instance Boundary (Contour ring)
   - `3`: Inter-Instance Gap / Contact Seam (Bottom-Hat morphological transform: $\text{Closing}(M) - M$)
2. **MoNuSAC Cell Phenotype Classification (`class_map`)**:
   - `-1`: Ignored / Non-MoNuSAC patches (GLaS and MoNuSeg)
   - `0`: Background
   - `1`: Epithelial
   - `2`: Lymphocyte
   - `3`: Neutrophil
   - `4`: Macrophage

### C. Multi-Task Prediction Head Contract (`MultiTaskPredictionHead`)
- `preds["gland_seg"]`: Gland predictions evaluated on `x[is_gland]` (4 classes).
- `preds["gland_neg"]`: Cell patches `x[is_cell]` evaluated through the gland head for **negative background supervision** (target: class 0), actively suppressing false-positive cell detections in glands.
- `preds["cell_seg"]`: Shared cell segmentation evaluated on `x[is_cell]`.
- `preds["monuseg_seg"]`: Sliced from cell predictions for MoNuSeg patches (`task_ids[is_cell] == 1`).
- `preds["monusac_seg"]`: Sliced from cell predictions for MoNuSAC patches (`task_ids[is_cell] == 2`).
- `preds["cell_class"]`: MoNuSAC phenotype classification evaluated on `x[is_monusac]` (5 classes).

### D. Multi-Task Loss Formulation (`MultiTaskLoss` & `SoftYoudenJLoss`)
- Integrates differentiable **Soft Youden's J-statistic regularization** ($J = \text{Sensitivity} + \text{Specificity} - 1$) alongside weighted Cross-Entropy and Generalized Dice loss to separate contacting cells and rare gap pixels.
- Total multi-task loss formulation:
  $$\mathcal{L} = w_{\text{glas}} \cdot (\mathcal{L}_{\text{glas}} + \lambda_{\text{neg}} \mathcal{L}_{\text{neg}}) + w_{\text{monuseg}} \cdot \mathcal{L}_{\text{monuseg}} + w_{\text{monusac}} \cdot (\mathcal{L}_{\text{monusac\_seg}} + \mathcal{L}_{\text{monusac\_cls}})$$

### E. Class-Stratified Hierarchical MoNuSAC Resampling
- **Hierarchical Priority Bucket Assignment**:
  1. **Macrophage**: Any patch containing $\ge 1$ macrophage nucleus.
  2. **Neutrophil**: Else any patch containing $\ge 1$ neutrophil nucleus.
  3. **Epithelial / Lymphocyte**: Else majority foreground area.
- **Equalized Virtual Pool & Cyclic Interleaving**:
  Minority classes are equalized to the maximum class count $N_{\text{max}} = \max(|B_c|)$ via replacement sampling. Each epoch shuffles class buckets independently and draws equal class quotas cyclically interleaved: `[C0, C1, C2, C3, C0, C1, ...]`, ensuring balanced gradient distribution.
- **Throughput**: Set to **160 steps per epoch** (`steps_per_epoch: 160`), completing in **~7.6 minutes per epoch** with 4 workers and `torch.backends.cudnn.benchmark = True`.

---

## 3. Environment & Execution Commands

### Dedicated Python Environment
Use the active environment:
```bash
/usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python
```

### Multi-Task Training
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python scripts/train_multitask.py \
    --config configs/multitask/multitask_sdenetpointgradexp_512.yaml
```

### Cluster Runner Scripts (nohup / Slurm)
- **Train**:
  ```bash
  bash scripts/run_cluster_train.sh configs/multitask/multitask_sdenetpointgradexp_512.yaml
  ```
- **Evaluate**:
  ```bash
  bash scripts/run_cluster_eval.sh checkpoints/multitask/multitask_sdenetpointgradexp_4class_512/best.pt
  ```
- **End-to-End Pipeline**:
  ```bash
  bash scripts/run_cluster_pipeline.sh configs/multitask/multitask_sdenetpointgradexp_512.yaml
  ```

### Full Multi-Task Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python scripts/eval_multitask_inference.py \
    --config configs/multitask/multitask_sdenetpointgradexp_512.yaml \
    --checkpoint checkpoints/multitask/multitask_sdenetpointgradexp_4class_512/best.pt
```

### Single Dataset Evaluation
- **GLaS Evaluation**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python scripts/test_glas.py \
      --checkpoint checkpoints/multitask/multitask_sdenetpointgradexp_4class_512/best.pt
  ```
- **MoNuSAC Evaluation**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python scripts/test_monusac.py \
      --checkpoint checkpoints/multitask/multitask_sdenetpointgradexp_4class_512/best.pt
  ```

### Running Test Suite
```bash
PYTHONPATH=. /usr/mvl2/rgnm9/miniconda3/envs/sdenet-meru/bin/python -m unittest discover -s tests -p "test_*.py"
```

---

## 4. Key Invariant Rules
1. **Preserve the 2:1:1 batch contract** ($B/2$ GLaS, $B/4$ MoNuSeg, $B/4$ MoNuSAC).
2. **Dynamic In-Batch Masking**: Always route task patches using boolean indicator masks (`is_gland`, `is_monuseg`, `is_monusac`, `is_cell`); never assume static positional indices.
3. **Deterministic FP32 Precision**: Keep execution in deterministic FP32 precision.
4. **Synchronized Documentation**: Keep this `README.md` updated with exact configuration paths, script arguments, and model signatures.
