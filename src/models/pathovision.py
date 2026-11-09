"""PathoVision baseline model (M1).

Architecture: EfficientNetB2 (ImageNet) + GAP + Dense(512) + Dense(256) + Dense(7)

DO NOT MODIFY this file for experiments. Create new model files instead.

Version: 1.0.0
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import freeze_backbone, get_backbone
from src.models.heads import PathoVisionHead

logger = logging.getLogger(__name__)


class PathoVision(nn.Module):
    """PathoVision baseline model.

    Architecture:
        - Backbone: EfficientNetB2 with ImageNet pretrained weights
        - Frozen layers: all except last 10 parameter groups
        - Global Average Pooling (from backbone)
        - Dense(512, ReLU) + Dropout(0.5)
        - Dense(256, ReLU) + Dropout(0.3)
        - Dense(7, Softmax) [softmax applied at inference only]

    Args:
        num_classes: Number of output classes (default: 7).
        pretrained: Load ImageNet pretrained weights.
        freeze_except_last_n: Number of parameter groups to keep trainable.
        head_hidden_dims: Hidden layer dimensions.
        head_dropout_rates: Dropout rates per hidden layer.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_except_last_n: int = 10,
        head_hidden_dims: list[int] | None = None,
        head_dropout_rates: list[float] | None = None,
    ) -> None:
        super().__init__()

        if head_hidden_dims is None:
            head_hidden_dims = [512, 256]
        if head_dropout_rates is None:
            head_dropout_rates = [0.5, 0.3]

        # Backbone: EfficientNetB2 as feature extractor (num_classes=0 removes classifier)
        self.backbone = get_backbone(
            name="efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,
        )

        # Freeze layers (all except the last 10)
        freeze_backbone(
            self.backbone,
            strategy="last_n",
            freeze_except_last_n=freeze_except_last_n,
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features

        # Classification head
        self.head = PathoVisionHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dims=head_hidden_dims,
            dropout_rates=head_dropout_rates,
        )

        self.num_classes = num_classes

        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "PathoVision model: %.2fM total params, %.2fM trainable",
            total_params / 1e6, trainable_params / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input images of shape (batch, 3, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.backbone(x)  # (batch, feature_dim)
        logits = self.head(features)  # (batch, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict with probabilities and class indices.

        Args:
            x: Input images of shape (batch, 3, H, W).

        Returns:
            Dictionary with:
                - logits: Raw logits (batch, num_classes)
                - probabilities: Softmax probabilities (batch, num_classes)
                - predicted_class: Predicted class indices (batch,)
                - confidence: Confidence of prediction (batch,)
        """
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
        """Extract backbone features (before classification head).

        Useful for hybrid CNN + classical ML pipeline.

        Args:
            x: Input images of shape (batch, 3, H, W).

        Returns:
            Features of shape (batch, feature_dim).
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
        return features
