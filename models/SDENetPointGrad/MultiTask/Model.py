from typing import Optional, Union, Tuple, Dict, Any
import torch
from torch import nn

from models.SDENetPointGrad.MultiTask.ImageEncoder.ImageEncoder import ImageEncoder
from models.SDENetPointGrad.MultiTask.GeometricEncoder.GeometricEncoder import GeometricEncoder
from models.SDENetPointGrad.MultiTask.Decoder.Decoder import Decoder
from models.SDENetPointGrad.MultiTask.Decoder.PredictionHead import MultiTaskPredictionHead, PredictionHead


class Model(nn.Module):
    """
    MultiTask SDENetPointGradEXP Model.
    Integrates SDE image feature extraction, PointGrad geometric encoder,
    multimodal attention fusion decoder, and dynamic multi-task prediction heads:
      - 4-class J-regularized segmentation ('4class_j')
      - 3-class J-regularized segmentation ('3class_j')
      - 2D gradient flow regression ('flow')
      - Binary segmentation ('binary')
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
            "Full"
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
        num_gland_classes: Optional[int] = None,
        num_cell_classes: Optional[int] = None,
        num_classification_classes: int = 5,
        predict_flow: Optional[bool] = None,
        multitask: bool = True,
        classification: bool = True,
        num_classes: int = 4,
        **kwargs,
    ):
        super().__init__()

        if geometric_encoder_channels is None:
            geometric_encoder_channels = encoder_channels

        self.multitask = multitask
        self.approach = approach
        self.classification = classification
        self.num_classes = num_classes
        self.num_classification_classes = num_classification_classes

        # =====================================================
        # 1. IMAGE ENCODER (Depthwise SDE Block)
        # =====================================================
        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels,
            r=image_sde_r,
        )

        # =====================================================
        # 2. GEOMETRIC ENCODER (PointGrad Convolutions)
        # =====================================================
        self.geometric_encoder = GeometricEncoder(
            in_channels=geometric_in_channels,
            out_channels=geometric_encoder_channels,
        )

        # =====================================================
        # 3. DECODER (Multimodal Attention Fusion)
        # =====================================================
        self.decoder = Decoder(
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
        # 4. PREDICTION HEAD (Dynamic Multi-Task Heads)
        # =====================================================
        self.prediction_head = MultiTaskPredictionHead(
            in_channels=decoder_channels[-1],
            approach=approach,
            num_gland_classes=num_gland_classes,
            num_cell_classes=num_cell_classes,
            num_classification_classes=num_classification_classes,
            predict_flow=predict_flow,
        )

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [B, C, H, W], "
                f"but received {x.shape}."
            )

        # 1. Feature extraction
        image_features = self.image_encoder(x)
        geometric_features = self.geometric_encoder(x)

        # 2. Decoder fusion
        decoder_features = self.decoder(
            image_features=image_features,
            geometry_features=geometric_features,
        )

        # 3. Multi-task prediction
        return self.prediction_head(
            decoder_features,
            task=task,
            task_ids=task_ids,
        )
