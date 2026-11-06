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

### A. Batch Composition, Gradient Accumulation & In-Batch Shuffling
- **Batch Size Constraint**: Must be a multiple of 4 (e.g., `4`, `8`, `16`, `24`).
  - Standard training uses `batch_size: 8`.
  - For memory-constrained environments or large patch sizes, `batch_size: 4` with `gradient_accumulation_steps: 2` preserves the exact effective batch size of 8.
- **Exact 2:1:1 Task Allocation**:
  - $B=8$: 4 GLaS patches (`task_id = 0`), 2 MoNuSeg patches (`task_id = 1`), 2 MoNuSAC patches (`task_id = 2`).
  - $B=4$: 2 GLaS patches (`task_id = 0`), 1 MoNuSeg patch (`task_id = 1`), 1 MoNuSAC patch (`task_id = 2`).
- **Randomized Order**: Within each batch, patches are **randomly shuffled** during training via `EqualTaskBatchSampler`.
- **Mask-Based Routing**: **Never assume fixed positional slices** (e.g., `batch[:4]`). Always route using boolean indicator masks:
  ```python
  is_gland   = (task_ids == 0)
  is_monuseg = (task_ids == 1)
  is_monusac = (task_ids == 2)
  is_cell    = is_monuseg | is_monusac
  ```

### B. Target Schemas
1. **4-Class J4 Semantic Segmentation Target (`target_4class`)**:
   - `0`: Background (Outside cells/glands)
   - `1`: Cell/Gland Interior ($M_{\text{cell}}$, non-touching cell body pixels)
   - `2`: Touching Region ($\Gamma$, cell pixels directly adjacent/touching a different cell instance in 8-neighborhood)
   - `3`: Gap (Background pixels in narrow crevices between cells via Bottom-Hat: $\text{Closing}(M) - M$)
2. **MoNuSAC Cell Phenotype Classification (`class_map`)**:
   - `-1`: Ignored / Non-MoNuSAC patches (GLaS and MoNuSeg)
   - `0`: Background
   - `1`: Epithelial
   - `2`: Lymphocyte
   - `3`: Neutrophil
   - `4`: Macrophage

### C. Multi-Task Prediction Head Contract (`MultiTaskPredictionHead` & `ModelMaskedDecoders`)
- **Training Mode (`model.train()`) - Task-Sliced Feature Routing**:
  - `image_features` and `geometric_features` are sliced before being routed to each respective decoder.
  - `preds["gland_seg"]`: Gland predictions evaluated ONLY on `x[is_gland]` (4 classes).
  - `preds["cell_seg"]`: Shared cell segmentation evaluated on `x[is_cell]` (4 classes).
  - `preds["monuseg_seg"]`: Evaluated on MoNuSeg patches (`is_monuseg`).
  - `preds["monusac_seg"]`: Evaluated on MoNuSAC patches (`is_monusac`).
  - `preds["cell_class"]`: MoNuSAC phenotype classification evaluated ONLY on `x[is_monusac]` (5 classes).
  - Cuts intermediate memory consumption by >50% (from 45.6 GB down to ~20.8 GB peak during backward pass).
- **Inference Mode (`model.eval()`) - Full Dense Prediction**:
  - When `task_ids` is None or during evaluation (`model.eval()`), all three decoders (`gland_decoder`, `cell_seg_decoder`, `cell_class_decoder`) evaluate densely across all input patches, yielding full-batch predictions of shape `[B, C, H, W]` across all heads simultaneously.

### D. Multi-Task Loss Formulation (`MultiTaskLoss`)
- **4-Class Segmentation (GLaS, MoNuSeg, MoNuSAC seg)**: Integrates differentiable **Pairwise Soft Youden's J-statistic regularization** ($J = \text{Sensitivity} + \text{Specificity} - 1$) alongside Cross-Entropy and Generalized Dice loss.
- **5-Class Phenotype Classification (MoNuSAC cls)**: Supervised over the whole patch using strictly **unweighted Cross-Entropy** (`weight=None`) and Multi-Class Dice loss without J-regularization. Valid pixel existence is strictly guarded (`(sac_cls_gt >= 0).any()`) to return `0.0` gracefully and prevent NaN loss when batches contain non-labeled pixels.
- Negative supervision is completely removed in favor of clean task gradient isolation via masked decoders.
- Total multi-task loss formulation:
  $$\mathcal{L} = w_{\text{glas}} \cdot \mathcal{L}_{\text{glas}} + w_{\text{monuseg}} \cdot \mathcal{L}_{\text{monuseg}} + w_{\text{monusac}} \cdot (\mathcal{L}_{\text{monusac\_seg}} + \mathcal{L}_{\text{monusac\_cls}})$$

### E. Memory Management & Multi-Decoder Scaling
- **VRAM Allocation Configuration**: When running training on dense multi-decoder architectures with 512x512 patches, configure PyTorch's caching allocator to prevent heap fragmentation across hundreds of accumulation steps:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
  ```
- **Intermediate Activation Lifecycle**: In `trainer_multitask.py`, explicit intermediate tensor deletion (`del preds, batch_gpu, loss, loss_scaled`) and periodic cache clearing (`torch.cuda.empty_cache()` every 50 iterations) ensure stable long-running training within single 48GB Quadro GPUs.

### F. Parameter-Free J4 MAP Instance Reconstruction Pipeline (`j4_reconstruct_instances`)
1. **Four-Channel Softmax Prediction**: $Z \in \mathbb{R}^{4 \times H \times W}$ for classes $\{0: \text{BG}, 1: \text{Cell}, 2: \text{Touching}, 3: \text{Gap}\}$.
2. **Maximum A Posteriori (MAP)**: $\hat{h}(p) = \arg\max_{k \in \{0, 1, 2, 3\}} Z_k(p)$.
3. **Gap Resolution**: Confident gaps remain background ($0$); uncertain gaps ($Z_3 < \tau$) are reassigned to $\arg\max_{k \in \{0, 1, 2\}} Z_k(p)$.
4. **Connected-Component Labeling**: Applied strictly to cell interiors $M_{\text{cell}} = [\hat{h} == 1]$ to generate distinct integer instance seeds $1..N$.
5. **Touching Pixel Assignment**: Touching pixels ($\hat{h} == 2$) are assigned the integer ID of their closest labeled cell interior instance via Euclidean distance transform.
6. **Mask-Guided Phenotype Majority Voting**: Phenotype classification is evaluated on the segmented cell instances ($\hat{g} > 0$), where instance pixels vote among foreground classes $1..4$ to assign a single uniform phenotype per instance, eliminating false positives on stroma.

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
