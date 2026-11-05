# scripts/build_models.py
"""
Dedicated Model Builder for Multi-Task SDENetPointGradEXP Architecture.
Constructs the multi-task model with ASPP GLaS head, shared cell segmentation,
and MoNuSAC phenotype classification.
"""

from typing import Dict, Any
import torch.nn as nn

from models.SDENetPointGrad.MultiTask.Model import Model as MultiTaskSDENetPointGradEXP
from models.SDENetPointGrad.MultiTask.ModelMaskedDecoders import ModelMaskedDecoders


def build_multitask_masked_decoders(config: Dict[str, Any]) -> ModelMaskedDecoders:
    """Instantiates ModelMaskedDecoders with 3 task-specific decoders and masked outputs."""
    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder", model_config.get("image_encoder", {}))
    geom_config = model_config.get("geometric_encoder", {})
    decoder_config = model_config.get("decoder", {})
    heads_config = model_config.get("heads", {})

    in_channels = encoder_config.get("in_channels", 3)
    encoder_channels = tuple(encoder_config.get("encoder_channels", (64, 128, 128, 256)))

    pooled_sizes = decoder_config.get("pooled_sizes", ((32, 32), (64, 64), (64, 64), (64, 64)))
    if isinstance(pooled_sizes, (list, tuple)):
        pooled_sizes = tuple(tuple(size) for size in pooled_sizes)

    sde_block = decoder_config.get("sde_block", ("SE_only", "Full", "Full"))
    if isinstance(sde_block, list):
        sde_block = tuple(sde_block)

    approach = (
        model_config.get("approach")
        or heads_config.get("approach")
        or config.get("approach", "4class_j")
    )

    num_gland_classes = heads_config.get("segmentation", {}).get("num_gland_classes", 4)
    num_cell_classes = heads_config.get("segmentation", {}).get("num_cell_classes", 4)
    num_classification_classes = heads_config.get("classification", {}).get("num_classes", 5)

    return ModelMaskedDecoders(
        in_channels=in_channels,
        encoder_channels=encoder_channels,
        image_sde_r=encoder_config.get("sde_r", 3),
        geometric_in_channels=geom_config.get("in_channels", in_channels),
        geometric_encoder_channels=tuple(geom_config.get("encoder_channels", encoder_channels)),
        decoder_channels=tuple(decoder_config.get("decoder_channels", (256, 128, 128, 64))),
        skip_channels=tuple(decoder_config.get("skip_channels", (128, 128, 64))),
        embed_dim=decoder_config.get("embed_dim", 128),
        num_heads=decoder_config.get("num_heads", 4),
        sde_block=sde_block,
        r=tuple(decoder_config.get("r", (16, 16, 8))),
        pooled_sizes=pooled_sizes,
        dropout=decoder_config.get("dropout", 0.0),
        approach=approach,
        num_gland_classes=num_gland_classes,
        num_cell_classes=num_cell_classes,
        num_classification_classes=num_classification_classes,
        multitask=True,
    )


def build_multitask_sdenetpointgradexp(config: Dict[str, Any]) -> MultiTaskSDENetPointGradEXP:
    """Instantiates MultiTaskSDENetPointGradEXP from configuration dictionary."""
    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder", model_config.get("image_encoder", {}))
    geom_config = model_config.get("geometric_encoder", {})
    decoder_config = model_config.get("decoder", {})
    heads_config = model_config.get("heads", {})

    in_channels = encoder_config.get("in_channels", 3)
    encoder_channels = tuple(encoder_config.get("encoder_channels", (64, 128, 128, 256)))

    pooled_sizes = decoder_config.get("pooled_sizes", ((32, 32), (64, 64), (64, 64), (64, 64)))
    if isinstance(pooled_sizes, (list, tuple)):
        pooled_sizes = tuple(tuple(size) for size in pooled_sizes)

    sde_block = decoder_config.get("sde_block", ("SE_only", "Full", "Full"))
    if isinstance(sde_block, list):
        sde_block = tuple(sde_block)

    approach = (
        model_config.get("approach")
        or heads_config.get("approach")
        or config.get("approach", "4class_j")
    )

    num_gland_classes = heads_config.get("segmentation", {}).get("num_gland_classes", 4)
    num_cell_classes = heads_config.get("segmentation", {}).get("num_cell_classes", 4)
    num_classification_classes = heads_config.get("classification", {}).get("num_classes", 5)
    predict_flow = heads_config.get("flow", {}).get("enabled", False)

    return MultiTaskSDENetPointGradEXP(
        in_channels=in_channels,
        encoder_channels=encoder_channels,
        image_sde_r=encoder_config.get("sde_r", 3),
        geometric_in_channels=geom_config.get("in_channels", in_channels),
        geometric_encoder_channels=tuple(geom_config.get("encoder_channels", encoder_channels)),
        decoder_channels=tuple(decoder_config.get("decoder_channels", (256, 128, 128, 64))),
        skip_channels=tuple(decoder_config.get("skip_channels", (128, 128, 64))),
        embed_dim=decoder_config.get("embed_dim", 128),
        num_heads=decoder_config.get("num_heads", 4),
        sde_block=sde_block,
        r=tuple(decoder_config.get("r", (16, 16, 8))),
        pooled_sizes=pooled_sizes,
        dropout=decoder_config.get("dropout", 0.0),
        approach=approach,
        num_gland_classes=num_gland_classes,
        num_cell_classes=num_cell_classes,
        num_classification_classes=num_classification_classes,
        predict_flow=predict_flow,
        multitask=True,
    )


MODEL_BUILDERS = {
    "multitask_sdenetpointgradexp": build_multitask_sdenetpointgradexp,
    "multitask_masked_decoders": build_multitask_masked_decoders,
    "masked_decoders": build_multitask_masked_decoders,
    "split_decoders": build_multitask_masked_decoders,
    "sdenetpointgradexp": build_multitask_sdenetpointgradexp,
    "multitask": build_multitask_sdenetpointgradexp,
    "default": build_multitask_sdenetpointgradexp,
}


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Universal model builder entrypoint for multi-task training and inference."""
    model_name = config.get("model", {}).get("name", "multitask_sdenetpointgradexp").lower().strip()
    builder = MODEL_BUILDERS.get(model_name, build_multitask_sdenetpointgradexp)
    return builder(config)


__all__ = ["build_model", "build_multitask_sdenetpointgradexp", "build_multitask_masked_decoders", "MODEL_BUILDERS"]
