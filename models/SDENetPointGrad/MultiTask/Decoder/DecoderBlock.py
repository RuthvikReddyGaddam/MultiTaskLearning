from typing import Optional
import torch
import torch.nn as nn

from models.SDENetPointGrad.MultiTask.ImageEncoder.SDEBlock.SDEBlock import SDEBlock
from models.SDENetPointGrad.MultiTask.ImageEncoder.ImageEncoder import ConvBlock
from models.SDENetPointGrad.MultiTask.Decoder.FusionBlock import FusionBlock


class UpBlock(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        out_channels: int,
        sde_block: str = "Full",
        r: int = 16,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.sde_block = SDEBlock(
            in_channels=skip_channels,
            dec_in_channels=decoder_channels,
            sde_block=sde_block,
            r=r,
            k_size=k_size,
            padding=padding,
            stride=stride,
            input=False,
        )

        self.sde_block_out_channels = (
            self.sde_block.calculate_C_sd()
        )

        self.conv_block = nn.Sequential(
            ConvBlock(
                in_channels=self.sde_block_out_channels,
                out_channels=out_channels,
                kernel_size=k_size,
                stride=stride,
                padding=padding,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=k_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image:
                CNN skip feature map.

            x:
                Decoder feature map from the previous/deeper stage.

        Returns:
            Upsampled and fused decoder feature map.
        """

        x = self.sde_block(image, x)
        x = self.conv_block(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        out_channels: int,
        geometry_channels: int,
        r: int,
        embed_dim: int,
        num_heads: int,
        pooled_height: int,
        pooled_width: int,
        dropout: float,
        sde_block: str = "Full",
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.upblock = UpBlock(
            skip_channels=skip_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            sde_block=sde_block,
            r=r,
            k_size=k_size,
            stride=stride,
            padding=padding,
        )

        self.fusion_block = FusionBlock(
            image_in_channels=out_channels,
            geometry_in_channels=geometry_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooled_height=pooled_height,
            pooled_width=pooled_width,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        image: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upblock(
            image=image,
            x=x,
        )

        x = self.fusion_block(
            image_feature=x,
            geometry_feature=geometry,
        )

        return x