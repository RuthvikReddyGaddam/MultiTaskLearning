# models/SDENetPointGrad/MultiTask/Decoder/PredictionHead.py

from typing import Dict, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(channels: int) -> int:
    """Finds an optimal divisor for GroupNorm without exceeding 32 groups."""
    num_groups = min(32, max(1, channels // 8))
    while channels % num_groups != 0:
        num_groups -= 1
    return num_groups


class ASPPConv(nn.Sequential):
    """Atrous (dilated) 3x3 convolution block with GroupNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    """Image-level global average pooling branch for whole-patch histological context."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = super().forward(x)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class AtrousSpatialPyramidHead(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) prediction head designed specifically for
    macroscopic tissue structures (e.g. GLaS gland segmentation and white lumen context).

    Multi-scale branches:
      1. 1x1 convolution for localized boundary and cleft features.
      2. 3x3 dilated convolutions with rates (6, 12, 18) to capture epithelial wall
         enclosure and contextual lumen surround without downsampling resolution.
      3. Global average pooling branch for whole-patch histological context.
      4. Channel projection with GroupNorm, ReLU, and 2D dropout.
      5. Final 1x1 classification layer producing `out_classes` logits.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_classes: int = 4,
        dilations: Sequence[int] = (6, 12, 18),
        mid_channels: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        mid = max(32, in_channels // 2) if mid_channels is None else mid_channels

        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
                nn.GroupNorm(_get_num_groups(mid), mid),
                nn.ReLU(inplace=True),
            )
        ]
        for d in dilations:
            branches.append(ASPPConv(in_channels, mid, dilation=d))
        branches.append(ASPPPooling(in_channels, mid))
        self.branches = nn.ModuleList(branches)

        total_branch_channels = len(branches) * mid
        self.project = nn.Sequential(
            nn.Conv2d(total_branch_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_get_num_groups(in_channels), in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Conv2d(in_channels, out_classes, kernel_size=1)

        # Kaiming normal initialization for stable FP32 convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.classifier.weight, std=0.001)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Expose classifier weight for gradient tracking and drop-in compatibility."""
        return self.classifier.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Expose classifier bias for gradient tracking and drop-in compatibility."""
        return self.classifier.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [b(x) for b in self.branches]
        cat = torch.cat(res, dim=1)
        proj = self.project(cat)
        return self.classifier(proj)


# ASPPHead alias for convenience
ASPPHead = AtrousSpatialPyramidHead


class MultiTaskPredictionHead(nn.Module):
    """
    Dynamic Multi-Task Prediction Head configured according to the training approach:
      1. '4class_j' / '4class': 4-class J-regularized segmentation
         - gland_seg: 4 classes (bg, inside, boundary, bottom-hat gap)
         - cell_seg:  4 classes (bg, inside, boundary, bottom-hat gap)
         - cell_class: 5 classes (MoNuSAC: bg + 4 phenotypes)
      2. '3class_j' / '3class': 3-class J-regularized segmentation
         - gland_seg: 3 classes (bg, inside, boundary)
         - cell_seg:  3 classes (bg, inside, boundary)
         - cell_class: 5 classes
      3. 'flow': Segmentation + 2D flow vector field regression
         - gland_seg:  1 class (binary)
         - gland_flow: 2 channels (Fx, Fy)
         - cell_seg:   1 class (binary)
         - cell_flow:  2 channels (Fx, Fy)
         - cell_class: 5 classes
      4. 'binary': Binary segmentation
         - gland_seg:  1 class
         - cell_seg:   1 class
         - cell_class: 5 classes
    """

    def __init__(
        self,
        in_channels: int = 64,
        approach: str = "4class_j",
        num_gland_classes: Optional[int] = None,
        num_cell_classes: Optional[int] = None,
        num_classification_classes: int = 5,
        predict_flow: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.approach = approach.lower().strip()
        self.num_classification_classes = num_classification_classes

        # Infer class counts and flow prediction based on approach
        if self.approach in ("4class_j", "4class", "4_class_j", "4_class", "4ch", "4channel"):
            self.num_gland_classes = 4 if num_gland_classes is None else num_gland_classes
            self.num_cell_classes = 4 if num_cell_classes is None else num_cell_classes
            self.predict_flow = False if predict_flow is None else bool(predict_flow)
        elif self.approach in ("3class_j", "3class", "3_class_j", "3_class", "3ch", "3channel"):
            self.num_gland_classes = 3 if num_gland_classes is None else num_gland_classes
            self.num_cell_classes = 3 if num_cell_classes is None else num_cell_classes
            self.predict_flow = False if predict_flow is None else bool(predict_flow)
        elif self.approach in ("flow", "hovernet", "cellpose"):
            self.num_gland_classes = 1 if num_gland_classes is None else num_gland_classes
            self.num_cell_classes = 1 if num_cell_classes is None else num_cell_classes
            self.predict_flow = True if predict_flow is None else bool(predict_flow)
        elif self.approach in ("binary", "single_channel", "single"):
            self.num_gland_classes = 1 if num_gland_classes is None else num_gland_classes
            self.num_cell_classes = 1 if num_cell_classes is None else num_cell_classes
            self.predict_flow = False if predict_flow is None else bool(predict_flow)
        else:
            # Fallback custom configuration
            self.num_gland_classes = 4 if num_gland_classes is None else num_gland_classes
            self.num_cell_classes = 4 if num_cell_classes is None else num_cell_classes
            self.predict_flow = False if predict_flow is None else bool(predict_flow)

        # 1. GLaS Gland Segmentation Head (Atrous Spatial Pyramid Pooling for lumen and context)
        self.gland_seg = AtrousSpatialPyramidHead(
            in_channels=in_channels,
            out_classes=self.num_gland_classes,
            dilations=kwargs.get("gland_dilations", (6, 12, 18)),
            mid_channels=kwargs.get("gland_aspp_channels", None),
        )

        # 2. Shared Cell Nuclei Segmentation Head (MoNuSeg + MoNuSAC)
        self.cell_seg = nn.Conv2d(in_channels, self.num_cell_classes, kernel_size=1)

        # 3. MoNuSAC Multi-Class Cell Phenotype Head (5 classes: bg + 4 phenotypes)
        self.cell_class = nn.Conv2d(in_channels, self.num_classification_classes, kernel_size=1)

        # 4. Optional Flow Vector Field Heads [2, H, W] for GLaS and Cell datasets
        if self.predict_flow:
            self.gland_flow = nn.Conv2d(in_channels, 2, kernel_size=1)
            self.cell_flow = nn.Conv2d(in_channels, 2, kernel_size=1)

    # Dataset-specific property aliases
    @property
    def glas_seg(self) -> AtrousSpatialPyramidHead:
        return self.gland_seg

    @property
    def monuseg_seg(self) -> nn.Conv2d:
        return self.cell_seg

    @property
    def monusac_seg(self) -> nn.Conv2d:
        return self.cell_seg

    @property
    def monusac_class(self) -> nn.Conv2d:
        return self.cell_class

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-task predictions.

        Args:
            x: Decoder feature map [B, in_channels, H, W].
            task: Optional single-task indicator ('glas', 'monuseg', 'monusac', 'cell', 'gland').
            task_ids: Optional [B] tensor with dataset indices:
                      0 = GLaS
                      1 = MoNuSeg
                      2 = MoNuSAC

        Returns:
            Dictionary containing prediction logits and flow maps.
        """
        # =========================================================
        # 1. Single-task evaluation mode
        # =========================================================
        if task in ("glas", "gland"):
            g_seg = self.gland_seg(x)
            out = {
                "gland_seg": g_seg,
                "glas_seg": g_seg,
                "segmentation": g_seg,
            }
            if self.predict_flow:
                g_flow = self.gland_flow(x)
                out["gland_flow"] = g_flow
                out["glas_flow"] = g_flow
                out["flow"] = g_flow
            return out

        elif task in ("monuseg", "cell", "nuclei"):
            c_seg = self.cell_seg(x)
            out = {
                "cell_seg": c_seg,
                "monuseg_seg": c_seg,
                "segmentation": c_seg,
            }
            if self.predict_flow:
                c_flow = self.cell_flow(x)
                out["cell_flow"] = c_flow
                out["monuseg_flow"] = c_flow
                out["flow"] = c_flow
            return out

        elif task == "monusac":
            c_seg = self.cell_seg(x)
            c_cls = self.cell_class(x)
            out = {
                "cell_seg": c_seg,
                "monusac_seg": c_seg,
                "segmentation": c_seg,
                "cell_class": c_cls,
                "monusac_class": c_cls,
                "classification": c_cls,
            }
            if self.predict_flow:
                c_flow = self.cell_flow(x)
                out["cell_flow"] = c_flow
                out["monusac_flow"] = c_flow
                out["flow"] = c_flow
            return out

        # =========================================================
        # 2. Mask-based conditional evaluation during training
        # =========================================================
        if task_ids is not None:
            is_gland = (task_ids == 0)
            is_cell = (task_ids == 1) | (task_ids == 2)
            is_monusac = (task_ids == 2)
            preds: Dict[str, torch.Tensor] = {}

            if is_gland.any():
                x_g = x[is_gland]
                g_seg = self.gland_seg(x_g)
                preds["gland_seg"] = g_seg
                preds["glas_seg"] = g_seg
                if self.predict_flow:
                    g_flow = self.gland_flow(x_g)
                    preds["gland_flow"] = g_flow
                    preds["glas_flow"] = g_flow

            if is_cell.any():
                x_c = x[is_cell]
                preds["gland_neg"] = self.gland_seg(x_c)
                preds["glas_neg"] = preds["gland_neg"]

                c_seg = self.cell_seg(x_c)
                preds["cell_seg"] = c_seg

                is_monuseg_in_cell = (task_ids[is_cell] == 1)
                is_monusac_in_cell = (task_ids[is_cell] == 2)
                if is_monuseg_in_cell.any():
                    preds["monuseg_seg"] = c_seg[is_monuseg_in_cell]
                if is_monusac_in_cell.any():
                    preds["monusac_seg"] = c_seg[is_monusac_in_cell]

                if self.predict_flow:
                    c_flow = self.cell_flow(x_c)
                    preds["cell_flow"] = c_flow
                    if is_monuseg_in_cell.any():
                        preds["monuseg_flow"] = c_flow[is_monuseg_in_cell]
                    if is_monusac_in_cell.any():
                        preds["monusac_flow"] = c_flow[is_monusac_in_cell]

            if is_monusac.any():
                x_s = x[is_monusac]
                c_cls = self.cell_class(x_s)
                preds["cell_class"] = c_cls
                preds["monusac_class"] = c_cls

            return preds

        # =========================================================
        # 3. Dense evaluation across entire batch
        # =========================================================
        g_seg = self.gland_seg(x)
        c_seg = self.cell_seg(x)
        c_cls = self.cell_class(x)
        out = {
            "gland_seg": g_seg,
            "glas_seg": g_seg,
            "cell_seg": c_seg,
            "monuseg_seg": c_seg,
            "monusac_seg": c_seg,
            "cell_class": c_cls,
            "monusac_class": c_cls,
            "segmentation": c_seg,
            "classification": c_cls,
        }
        if self.predict_flow:
            g_flow = self.gland_flow(x)
            c_flow = self.cell_flow(x)
            out["gland_flow"] = g_flow
            out["glas_flow"] = g_flow
            out["cell_flow"] = c_flow
            out["monuseg_flow"] = c_flow
            out["monusac_flow"] = c_flow
            out["flow"] = c_flow

        return out


# PredictionHead alias for drop-in compatibility
PredictionHead = MultiTaskPredictionHead

__all__ = [
    "MultiTaskPredictionHead",
    "PredictionHead",
    "AtrousSpatialPyramidHead",
    "ASPPHead",
]
