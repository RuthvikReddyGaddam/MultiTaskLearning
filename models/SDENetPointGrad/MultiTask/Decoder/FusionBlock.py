from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalAttentionLayer(nn.Module):
    """
    Transformer-style joint self-attention over:

        [image tokens | appearance tokens | geometry tokens]

    All valid tokens can interact with one another.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        feedforward_multiplier: int = 4,
    ):
        super().__init__()

        hidden_dim = embed_dim * feedforward_multiplier

        # Pre-normalization for self-attention.
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)

        # Pre-normalization for feed-forward network.
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens:
                [B, N_total, embed_dim]

        Returns:
            Updated tokens:
                [B, N_total, embed_dim]
        """

        # ------------------------------------------------------
        # Multi-head self-attention
        # ------------------------------------------------------
        normalized_tokens = self.norm1(tokens)

        attention_output, _ = self.self_attention(
            query=normalized_tokens,
            key=normalized_tokens,
            value=normalized_tokens,
            need_weights=False,
        )

        tokens = tokens + self.attention_dropout(attention_output)

        # ------------------------------------------------------
        # Feed-forward network
        # ------------------------------------------------------
        normalized_tokens = self.norm2(tokens)

        feed_forward_output = self.feed_forward(normalized_tokens)

        tokens = tokens + self.feed_forward_dropout(
            feed_forward_output
            )

        return tokens


class FusionBlock(nn.Module):
    def __init__(
        self,
        image_in_channels: int,
        geometry_in_channels: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        pooled_height: int = 16,
        pooled_width: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.num_spatial_tokens = pooled_height * pooled_width


        self.image_projection = nn.Sequential(
            nn.Conv2d(
                image_in_channels,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
        )

        self.geometry_projection = nn.Sequential(
            nn.Conv2d(
                geometry_in_channels,
                embed_dim,
                kernel_size=1,
                bias=False,
            )
        )

        # ------------------------------------------------------
        # Adaptive pooling
        # ------------------------------------------------------
        self.image_pool = nn.AdaptiveAvgPool2d(
            (pooled_height, pooled_width)
        )

        self.geometry_pool = nn.AdaptiveAvgPool2d(
            (pooled_height, pooled_width)
        )

        # ------------------------------------------------------
        # Positional embeddings
        # ------------------------------------------------------
        self.image_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_spatial_tokens,
                embed_dim,
            )
        )

        self.geometry_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_spatial_tokens,
                embed_dim,
            )
        )

        # ------------------------------------------------------
        # Modality embeddings
        # ------------------------------------------------------
        self.image_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.geometry_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # ------------------------------------------------------
        # Multimodal attention
        # ------------------------------------------------------
        self.multimodal_attention = MultimodalAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                image_in_channels,
                kernel_size=1,
                bias=False,
            )
        )


        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(
            self.image_position_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.geometry_position_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.image_modality_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.geometry_modality_embedding,
            std=0.02,
        )

    @staticmethod
    def _get_num_groups(channels: int) -> int:
        num_groups = max(1, channels // 8)
        while channels % num_groups != 0:
            num_groups -= 1
        return num_groups

    @staticmethod
    def feature_map_to_tokens(
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts:
            [B, C, H, W] -> [B, H*W, C]
        """
        if feature_map.ndim != 4:
            raise ValueError(
                "feature_map must have shape [B, C, H, W], "
                f"but received {feature_map.shape}."
            )

        return feature_map.flatten(2).transpose(1, 2)

    def tokens_to_feature_map(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts:
            [B, pooled_height*pooled_width, embed_dim]
            -> [B, embed_dim, pooled_height, pooled_width]
        """
        batch_size, num_tokens, channels = tokens.shape

        if num_tokens != self.num_spatial_tokens:
            raise ValueError(
                f"Expected {self.num_spatial_tokens} image tokens, "
                f"but received {num_tokens}."
            )

        if channels != self.embed_dim:
            raise ValueError(
                f"Expected token dimension {self.embed_dim}, "
                f"but received {channels}."
            )

        return (
            tokens.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                self.embed_dim,
                self.pooled_height,
                self.pooled_width,
            )
        )

    def forward(
        self,
        image_feature: torch.Tensor,
        geometry_feature: torch.Tensor,
    ) -> torch.Tensor:
    

        original_height = image_feature.shape[-2]
        original_width = image_feature.shape[-1]

        # ------------------------------------------------------
        # 1. Project feature dimensions
        # ------------------------------------------------------
        projected_image = self.image_projection(image_feature)
        projected_geometry = self.geometry_projection(geometry_feature)

        # ------------------------------------------------------
        # 2. Pool spatial features
        # ------------------------------------------------------
        
        if not (self.pooled_height == original_height and self.pooled_width == original_width):
            pooled_image = self.image_pool(projected_image)
            pooled_geometry = self.geometry_pool(projected_geometry)
        else:
            pooled_image = projected_image
            pooled_geometry = projected_geometry

        # ------------------------------------------------------
        # 3. Convert spatial feature maps to tokens
        # ------------------------------------------------------
        image_tokens = self.feature_map_to_tokens(pooled_image)
        geometry_tokens = self.feature_map_to_tokens(pooled_geometry)

        # ------------------------------------------------------
        # 4. Add positional embeddings
        # ------------------------------------------------------
        image_tokens = (
            image_tokens
            + self.image_position_embedding
        )

        geometry_tokens = (
            geometry_tokens
            + self.geometry_position_embedding
        )

        # ------------------------------------------------------
        # 5. Add modality embeddings
        # ------------------------------------------------------
        image_tokens = (
            image_tokens
            + self.image_modality_embedding
        )

        geometry_tokens = (
            geometry_tokens
            + self.geometry_modality_embedding
        )

        # ------------------------------------------------------
        # 6. Concatenate modalities
        # ------------------------------------------------------
        combined_tokens = torch.cat(
            [
                image_tokens,
                geometry_tokens,
            ],
            dim=1,
        )

        combined_tokens = self.multimodal_attention(combined_tokens)

        image_token_count = image_tokens.shape[1]

        fused_image_tokens = combined_tokens[
            :,
            :image_token_count,
            :,
        ]


        # ------------------------------------------------------
        # 10. Convert tokens back into feature map
        # ------------------------------------------------------
        fused_image_map = self.tokens_to_feature_map(
            fused_image_tokens
        )

        if not (self.pooled_height == original_height and self.pooled_width == original_width):
            fused_image_map = F.interpolate(
                fused_image_map,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

        fused_image_map = self.output_projection(
            fused_image_map
        )

        # ------------------------------------------------------
        # 13. Residual fusion
        # ------------------------------------------------------
        output = image_feature + fused_image_map
        return output

