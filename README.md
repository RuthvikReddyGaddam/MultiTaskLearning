# Multi-Task Histopathology Learning Implementation

Dedicated, streamlined implementation of multi-task learning across GLaS (gland segmentation), MoNuSeg (nuclear segmentation), and MoNuSAC (nuclear segmentation & multi-class phenotype classification).

## Architecture & Features
- **Strict Multiple-of-8 Balanced Sampling**: Exactly 2:1:1 patch ratio per batch ($B/2$ GLaS, $B/4$ MoNuSeg, $B/4$ MoNuSAC) with zero dynamic shape patching.
- **Atrous Spatial Pyramid Pooling (ASPP) GLaS Head**: Multi-scale dilated convolutions ($r \in \{6, 12, 18\}$) and global image pooling to capture macroscopic glandular context and resolve lumen interiors.
- **Cell Negative Background Supervision**: The GLaS head receives negative supervision on cell patches against Class 0 (Background) to actively suppress false-positive cell detections.
- **Safe Gated Gland Lumen Reclamation**: Post-processing boundary closure + enclosed hole-filling gated by optical-white saturation ($V > 210, S < 35$), preventing false positives on eosin-pink stroma.
- **Pure FP32 Execution**: Deterministic numerical precision without mixed precision drift.

## Python Environment
Use the dedicated environment at:
`/cluster/VAST/civalab/users/rgnm9/newimp/sdenet-mmcv-loc/bin/python`

## Training
```bash
CUDA_VISIBLE_DEVICES=1 /cluster/VAST/civalab/users/rgnm9/newimp/sdenet-mmcv-loc/bin/python scripts/train_multitask.py --config configs/multitask_sdenetpointgradexp_512.yaml
```

## Inference & Evaluation
```bash
CUDA_VISIBLE_DEVICES=1 /cluster/VAST/civalab/users/rgnm9/newimp/sdenet-mmcv-loc/bin/python scripts/eval_multitask_inference.py --config configs/multitask_sdenetpointgradexp_512.yaml --checkpoint checkpoints/multitask/best.pt
```

## Running Tests
```bash
CUDA_VISIBLE_DEVICES=1 /cluster/VAST/civalab/users/rgnm9/newimp/sdenet-mmcv-loc/bin/python -m unittest discover -s tests -p "test_*.py"
```

