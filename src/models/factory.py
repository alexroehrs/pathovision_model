"""Model factory — build any model from YAML configuration."""

import logging
from typing import Any

import torch.nn as nn

from src.models.pathovision import PathoVision
from src.models.pathovision_configurable import ConfigurablePathoVision

logger = logging.getLogger(__name__)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model from configuration dictionary.

    Args:
        config: Full configuration dictionary with 'model' key.

    Returns:
        Constructed model (nn.Module).

    Examples:
        # Baseline (EfficientNetB2)
        config = {"model": {"backbone": "efficientnet_b2", ...}}
        model = build_model(config)

        # Other backbones (e.g. EfficientNetV2-S)
        config = {"model": {"backbone": "efficientnetv2_s", ...}}
        model = build_model(config)
    """
    model_cfg = config["model"]
    backbone = model_cfg["backbone"]
    num_classes = model_cfg.get("num_classes", 7)
    pretrained = model_cfg.get("pretrained", True)
    freeze_strategy = model_cfg.get("freeze_strategy", "last_n")
    freeze_except_last_n = model_cfg.get("freeze_except_last_n", 10)

    head_cfg = model_cfg.get("head", {})
    head_type = head_cfg.get("type", "pathovision")
    head_hidden_dims = head_cfg.get("hidden_dims", [512, 256])
    head_dropout_rates = head_cfg.get("dropout_rates", [0.5, 0.3])

    # Baseline PathoVision architecture
    if backbone == "efficientnet_b2" and head_type == "pathovision":
        model = PathoVision(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_except_last_n=freeze_except_last_n,
            head_hidden_dims=head_hidden_dims,
            head_dropout_rates=head_dropout_rates,
        )
    else:
        # Configurable model for any other combination
        head_kwargs = {
            "hidden_dims": head_hidden_dims,
            "dropout_rates": head_dropout_rates,
        }
        model = ConfigurablePathoVision(
            backbone_name=backbone,
            head_type=head_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_strategy=freeze_strategy,
            freeze_except_last_n=freeze_except_last_n,
            head_kwargs=head_kwargs,
        )

    logger.info("Built model: %s", type(model).__name__)
    return model
