"""PathoVision model with configurable backbone and head.

Differences from the baseline model (src/models/pathovision.py):
- Configurable backbone (EfficientNetV2, ConvNeXt, Swin, ViT)
- Optional SE attention in classification head
- Optional attention pooling
- Better regularization options

Version: 1.0.0
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import freeze_backbone, get_backbone
from src.models.heads import build_head

logger = logging.getLogger(__name__)


class ConfigurablePathoVision(nn.Module):
    """PathoVision model with configurable architecture.

    Args:
        backbone_name: Name of the backbone (from BACKBONE_REGISTRY).
        head_type: Type of classification head ('pathovision', 'attention_pooling', 'se_head').
        num_classes: Number of output classes.
        pretrained: Load pretrained weights.
        freeze_strategy: Layer freezing strategy.
        freeze_except_last_n: Layers to keep trainable.
        head_kwargs: Additional kwargs for the head constructor.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnetv2_s",
        head_type: str = "pathovision",
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_strategy: str = "last_n",
        freeze_except_last_n: int = 20,
        head_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if head_kwargs is None:
            head_kwargs = {}

        # Backbone
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Freeze layers
        freeze_backbone(
            self.backbone,
            strategy=freeze_strategy,
            freeze_except_last_n=freeze_except_last_n,
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Classification head
        self.head = build_head(
            head_type=head_type,
            in_features=self.feature_dim,
            num_classes=num_classes,
            **head_kwargs,
        )

        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ConfigurablePathoVision [%s + %s]: %.2fM total, %.2fM trainable",
            backbone_name, head_type, total_params / 1e6, trainable_params / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = probabilities.max(dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.backbone(x)
