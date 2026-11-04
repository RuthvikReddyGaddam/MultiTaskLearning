from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from models.SDENetPointGrad.MultiTask.Decoder.FusionBlock import FusionBlock
from models.SDENetPointGrad.MultiTask.Decoder.DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    FEATURE_NAMES = (
        "skip_l1",
        "skip_l2",
        "skip_l3",
    )

    def __init__(
        self,
        num_heads: int = 8,
        channels: Tuple[int, ...] = (
            256,
            128,
            128,
            64,
        ),
        skip_channels: Tuple[int, ...] = (
            128,
            128,
            64,
        ),

        geometry_channels: Tuple[int, ...] = (
            256,
            128,
            128,
            64,
        ),
        embed_dim: int = 128,
        sde_block=(
            "SE_only",
            "Full",
            "Full"
        ),
        r: Tuple[int, ...] = (
            16,
            16,
            8,
        ),
        dropout: float = 0.0,
        pooled_sizes: Tuple[Tuple[int, int], ...] = (
            (32,32),    # Bottleneck
            (64,64),    # skip_l1
            (64,64),    # skip_l2
            (64,64),    # skip_l3
        ),
    ):
        super().__init__()

        self.bottleneck_fusion = FusionBlock(
            image_in_channels=channels[0],
            geometry_in_channels=geometry_channels[0],
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooled_height=pooled_sizes[0][0],
            pooled_width=pooled_sizes[0][1],
            dropout=dropout,
        )
        decoder_blocks = {}

        for index, feature_name in enumerate(
            self.FEATURE_NAMES
        ):
            pooled_height, pooled_width = (
                pooled_sizes[index + 1]
            )

            decoder_blocks[feature_name] = DecoderBlock(
                skip_channels=skip_channels[index],
                decoder_channels=channels[index],
                out_channels=channels[index + 1],
                geometry_channels=geometry_channels[index+1],
                sde_block=sde_block[index],
                r=r[index],
                embed_dim=embed_dim,
                num_heads=num_heads,
                pooled_height=pooled_height,
                pooled_width=pooled_width,
                dropout=dropout,
                k_size=3,
                stride=1,
                padding=1,
            )

        self.decoder_blocks = nn.ModuleDict(
            decoder_blocks
        )

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        geometry_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        x = self.bottleneck_fusion(
            image_feature=image_features["bottleneck"],
            geometry_feature=geometry_features["bottleneck"],
        )
        for feature_name in self.FEATURE_NAMES:
            x = self.decoder_blocks[feature_name](
                x=x,
                image=image_features[feature_name],
                geometry=geometry_features[feature_name],
            )

        return x

