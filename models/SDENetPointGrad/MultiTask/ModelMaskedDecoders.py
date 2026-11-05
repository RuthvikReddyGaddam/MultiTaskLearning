# models/SDENetPointGrad/MultiTask/ModelMaskedDecoders.py

from typing import Optional, Union, Tuple, Dict, Any
import torch
from torch import nn

from models.SDENetPointGrad.MultiTask.ImageEncoder.ImageEncoder import ImageEncoder
from models.SDENetPointGrad.MultiTask.GeometricEncoder.GeometricEncoder import GeometricEncoder
from models.SDENetPointGrad.MultiTask.Decoder.Decoder import Decoder
from models.SDENetPointGrad.MultiTask.Decoder.PredictionHead import AtrousSpatialPyramidHead


class ModelMaskedDecoders(nn.Module):
    """
    MultiTask SDENetPointGrad Architecture with Shared Encoders and Task-Specific Decoders.
    
    Structure:
      - Shared Encoders:
        * ImageEncoder (SDE multi-scale convolutions)
        * GeometricEncoder (PointGrad multi-scale convolutions)
      - Three Task-Specific Decoders (all receive the full batch feature representation):
        * gland_decoder: Dedicated decoder for GLaS gland segmentation.
        * cell_seg_decoder: Dedicated decoder for shared cell nuclei segmentation (MoNuSeg + MoNuSAC).
        * cell_class_decoder: Dedicated decoder for MoNuSAC phenotype multiclass classification.
      - Three Dedicated Prediction Heads:
        * gland_head: Atrous Spatial Pyramid Pooling (ASPP) producing 4 gland classes.
        * cell_seg_head: 1x1 Conv producing 4 cell segmentation classes.
        * cell_class_head: 1x1 Conv producing 5 cell phenotype classes.
      - Masked Output Execution:
        At the end of prediction, broadcastable task masks zero out non-task-specific predictions:
        * gland_mask zeroes out non-GLaS patches -> zero gradient into gland_decoder from cell tasks.
        * cell_mask zeroes out GLaS patches -> zero gradient into cell_seg_decoder from gland task.
        * monusac_mask zeroes out non-MoNuSAC patches -> zero gradient into cell_class_decoder from other tasks.
    """
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels=(
            64,
            128,
            128,
            256,
        ),
        decoder_channels=(
            256,
            128,
            128,
            64,
        ),
        skip_channels=(
            128,
            128,
            64,
        ),
        embed_dim: int = 128,
        num_heads: int = 4,
        sde_block=(
            "SE_only",
            "Full",
            "Full",
        ),
        r=(
            16,
            16,
            8,
        ),
        pooled_sizes=(
            (32, 32),    # Bottleneck
            (64, 64),    # skip_l1
            (64, 64),    # skip_l2
            (64, 64),    # skip_l3
        ),
        dropout: float = 0.0,
        image_sde_r: int = 3,
        geometric_in_channels: int = 3,
        geometric_encoder_channels=None,
        approach: str = "4class_j",
        num_gland_classes: int = 4,
        num_cell_classes: int = 4,
        num_classification_classes: int = 5,
        multitask: bool = True,
        **kwargs,
    ):
        super().__init__()

        if geometric_encoder_channels is None:
            geometric_encoder_channels = encoder_channels

        self.multitask = multitask
        self.approach = approach
        self.num_gland_classes = num_gland_classes
        self.num_cell_classes = num_cell_classes
        self.num_classification_classes = num_classification_classes

        # =====================================================
        # 1. SHARED ENCODERS
        # =====================================================
        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels,
            r=image_sde_r,
        )

        self.geometric_encoder = GeometricEncoder(
            in_channels=geometric_in_channels,
            out_channels=geometric_encoder_channels,
        )

        # =====================================================
        # 2. THREE TASK-SPECIFIC DECODERS
        # =====================================================
        # Gland segmentation decoder
        self.gland_decoder = Decoder(
            num_heads=num_heads,
            channels=decoder_channels,
            skip_channels=skip_channels,
            geometry_channels=(
                geometric_encoder_channels[3],
                geometric_encoder_channels[2],
                geometric_encoder_channels[1],
                geometric_encoder_channels[0],
            ),
            embed_dim=embed_dim,
            sde_block=sde_block,
            r=r,
            pooled_sizes=pooled_sizes,
            dropout=dropout,
        )

        # Cell nuclei segmentation decoder (MoNuSeg + MoNuSAC)
        self.cell_seg_decoder = Decoder(
            num_heads=num_heads,
            channels=decoder_channels,
            skip_channels=skip_channels,
            geometry_channels=(
                geometric_encoder_channels[3],
                geometric_encoder_channels[2],
                geometric_encoder_channels[1],
                geometric_encoder_channels[0],
            ),
            embed_dim=embed_dim,
            sde_block=sde_block,
            r=r,
            pooled_sizes=pooled_sizes,
            dropout=dropout,
        )

        # Cell phenotype classification decoder (MoNuSAC)
        self.cell_class_decoder = Decoder(
            num_heads=num_heads,
            channels=decoder_channels,
            skip_channels=skip_channels,
            geometry_channels=(
                geometric_encoder_channels[3],
                geometric_encoder_channels[2],
                geometric_encoder_channels[1],
                geometric_encoder_channels[0],
            ),
            embed_dim=embed_dim,
            sde_block=sde_block,
            r=r,
            pooled_sizes=pooled_sizes,
            dropout=dropout,
        )

        # =====================================================
        # 3. TASK-SPECIFIC PREDICTION HEADS
        # =====================================================
        # GLaS ASPP head for multi-scale glandular context
        self.gland_head = AtrousSpatialPyramidHead(
            in_channels=decoder_channels[-1],
            out_classes=self.num_gland_classes,
            dilations=kwargs.get("gland_dilations", (6, 12, 18)),
            mid_channels=kwargs.get("gland_aspp_channels", None),
        )

        # 1x1 Conv head for shared cell nuclei segmentation
        self.cell_seg_head = nn.Conv2d(
            decoder_channels[-1],
            self.num_cell_classes,
            kernel_size=1,
        )

        # 1x1 Conv head for MoNuSAC phenotype multiclass classification
        self.cell_class_head = nn.Conv2d(
            decoder_channels[-1],
            self.num_classification_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"Input must have shape [B, C, H, W], but received {x.shape}."
            )

        # 1. Shared Feature Extraction (all B patches)
        image_features = self.image_encoder(x)
        geometric_features = self.geometric_encoder(x)

        # 2. Single-task inference branch (skips inactive decoders to accelerate test inference)
        if task is not None and task_ids is None:
            if task in ("glas", "gland"):
                feat = self.gland_decoder(image_features, geometric_features)
                pred = self.gland_head(feat)
                return {
                    "gland_seg": pred,
                    "glas_seg": pred,
                    "segmentation": pred,
                }
            elif task in ("monuseg", "cell", "nuclei"):
                feat = self.cell_seg_decoder(image_features, geometric_features)
                pred = self.cell_seg_head(feat)
                return {
                    "cell_seg": pred,
                    "monuseg_seg": pred,
                    "segmentation": pred,
                }
            elif task == "monusac":
                feat_seg = self.cell_seg_decoder(image_features, geometric_features)
                feat_cls = self.cell_class_decoder(image_features, geometric_features)
                p_seg = self.cell_seg_head(feat_seg)
                p_cls = self.cell_class_head(feat_cls)
                return {
                    "cell_seg": p_seg,
                    "monusac_seg": p_seg,
                    "cell_class": p_cls,
                    "monusac_class": p_cls,
                    "segmentation": p_seg,
                    "classification": p_cls,
                }

        # 3. Dense execution: ALL 3 decoders receive ALL features across the full batch
        feat_gland = self.gland_decoder(image_features, geometric_features)
        feat_cell_seg = self.cell_seg_decoder(image_features, geometric_features)
        feat_cell_class = self.cell_class_decoder(image_features, geometric_features)

        # 4. Dense raw predictions
        raw_gland = self.gland_head(feat_gland)
        raw_cell_seg = self.cell_seg_head(feat_cell_seg)
        raw_cell_class = self.cell_class_head(feat_cell_class)

        # 5. Masking non-task predictions to guarantee zero gradient on non-task patches
        if task_ids is not None:
            mask_gland = (task_ids == 0).view(-1, 1, 1, 1).to(dtype=raw_gland.dtype)
            mask_cell = ((task_ids == 1) | (task_ids == 2)).view(-1, 1, 1, 1).to(dtype=raw_cell_seg.dtype)
            mask_monuseg = (task_ids == 1).view(-1, 1, 1, 1).to(dtype=raw_cell_seg.dtype)
            mask_monusac = (task_ids == 2).view(-1, 1, 1, 1).to(dtype=raw_cell_class.dtype)

            # Multiply by task mask to zero out non-task predictions
            gland_seg = raw_gland * mask_gland
            cell_seg = raw_cell_seg * mask_cell
            monuseg_seg = raw_cell_seg * mask_monuseg
            monusac_seg = raw_cell_seg * mask_monusac
            cell_class = raw_cell_class * mask_monusac

            return {
                "gland_seg": gland_seg,
                "glas_seg": gland_seg,
                "cell_seg": cell_seg,
                "monuseg_seg": monuseg_seg,
                "monusac_seg": monusac_seg,
                "cell_class": cell_class,
                "monusac_class": cell_class,
                "segmentation": cell_seg,
                "classification": cell_class,
            }

        # Unmasked fallback (e.g. dense feature inspection)
        return {
            "gland_seg": raw_gland,
            "glas_seg": raw_gland,
            "cell_seg": raw_cell_seg,
            "monuseg_seg": raw_cell_seg,
            "monusac_seg": raw_cell_seg,
            "cell_class": raw_cell_class,
            "monusac_class": raw_cell_class,
            "segmentation": raw_cell_seg,
            "classification": raw_cell_class,
        }

