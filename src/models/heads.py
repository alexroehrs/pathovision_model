"""Classification heads for skin disease models."""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PathoVisionHead(nn.Module):
    """PathoVision classification head (baseline).

    Architecture: GAP -> Dense(512, ReLU) -> Dropout(0.5)
                       -> Dense(256, ReLU) -> Dropout(0.3)
                       -> Dense(num_classes, Softmax)

    Args:
        in_features: Number of input features from backbone.
        num_classes: Number of output classes.
        hidden_dims: List of hidden layer dimensions.
        dropout_rates: List of dropout rates after each hidden layer.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        hidden_dims: list[int] | None = None,
        dropout_rates: list[float] | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]
        if dropout_rates is None:
            dropout_rates = [0.5, 0.3]

        assert len(hidden_dims) == len(dropout_rates), (
            f"hidden_dims ({len(hidden_dims)}) and dropout_rates ({len(dropout_rates)}) "
            "must have the same length"
        )

        layers: list[nn.Module] = []
        prev_dim = in_features

        for dim, drop in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
            ])
            prev_dim = dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        logger.info(
            "PathoVisionHead: %d -> %s -> %d (dropouts: %s)",
            in_features, hidden_dims, num_classes, dropout_rates,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (batch, in_features).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)


class AttentionPoolingHead(nn.Module):
    """Classification head with attention-weighted pooling.

    Replaces GAP with a learnable attention mechanism that weights
    spatial locations before pooling.

    Args:
        in_features: Number of input feature channels.
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension after pooling.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.Tanh(),
            nn.Linear(in_features // 4, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention pooling.

        Args:
            x: Input features. If 4D (B, C, H, W), reshapes to (B, H*W, C).
                If 2D (B, C), passes through directly.

        Returns:
            Logits of shape (batch, num_classes).
        """
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.reshape(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)

            # Compute attention weights
            attn_weights = self.attention(x)  # (B, H*W, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)

            # Weighted average
            x = (x * attn_weights).sum(dim=1)  # (B, C)
        elif x.dim() == 2:
            pass  # Already pooled
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return self.classifier(x)


class SEHead(nn.Module):
    """Classification head with Squeeze-and-Excitation channel attention.

    Args:
        in_features: Number of input features.
        num_classes: Number of output classes.
        hidden_dims: List of hidden layer dimensions.
        dropout_rates: List of dropout rates.
        reduction: SE reduction ratio.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 7,
        hidden_dims: list[int] | None = None,
        dropout_rates: list[float] | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]
        if dropout_rates is None:
            dropout_rates = [0.4, 0.3]

        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid(),
        )

        # Classification layers
        layers: list[nn.Module] = []
        prev_dim = in_features
        for dim, drop in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
            ])
            prev_dim = dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply SE attention
        se_weights = self.se(x)
        x = x * se_weights

        x = self.features(x)
        return self.classifier(x)


HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "pathovision": PathoVisionHead,
    "attention_pooling": AttentionPoolingHead,
    "se_head": SEHead,
}


def build_head(
    head_type: str,
    in_features: int,
    num_classes: int,
    **kwargs: Any,
) -> nn.Module:
    """Build a classification head from the registry.

    Args:
        head_type: Name of the head type.
        in_features: Number of input features.
        num_classes: Number of output classes.
        **kwargs: Additional keyword arguments for the head constructor.

    Returns:
        Classification head module.
    """
    if head_type not in HEAD_REGISTRY:
        available = ", ".join(sorted(HEAD_REGISTRY.keys()))
        raise ValueError(f"Unknown head type '{head_type}'. Available: {available}")

    return HEAD_REGISTRY[head_type](
        in_features=in_features,
        num_classes=num_classes,
        **kwargs,
    )
