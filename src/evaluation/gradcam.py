"""Grad-CAM explainability visualizations for PathoVision."""

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_target_layer(model: nn.Module) -> nn.Module:
    """Get the last convolutional layer from the model's backbone.

    Args:
        model: PathoVision or ConfigurablePathoVision model.

    Returns:
        The target layer for Grad-CAM.
    """
    backbone = model.backbone

    # For EfficientNet models (timm)
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    if hasattr(backbone, "blocks"):
        # Last block of EfficientNet
        return backbone.blocks[-1]
    # For ConvNeXt
    if hasattr(backbone, "stages"):
        return backbone.stages[-1]
    # Fallback: last module with conv layers
    last_conv = None
    for module in backbone.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            last_conv = module
    if last_conv is not None:
        return last_conv

    raise ValueError("Could not automatically find target layer for Grad-CAM")


def generate_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_class: int | None = None,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmaps using pytorch-grad-cam library.

    Args:
        model: Trained model.
        images: Input images tensor (B, C, H, W).
        target_class: Target class index (None = predicted class).
        target_layer: Layer to compute CAM from (None = auto-detect).

    Returns:
        Heatmaps array of shape (B, H, W) with values in [0, 1].
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        logger.error("pytorch-grad-cam not installed. Run: pip install grad-cam")
        raise

    if target_layer is None:
        target_layer = get_target_layer(model)

    cam = GradCAM(model=model, target_layers=[target_layer])

    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)] * images.shape[0]
    else:
        targets = None

    grayscale_cams = cam(input_tensor=images, targets=targets)

    # Clean up (API varies by version)
    if hasattr(cam, 'release'):
        cam.release()
    del cam

    return grayscale_cams


def visualize_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "",
    save_path: str | Path | None = None,
    alpha: float = 0.4,
) -> plt.Figure:
    """Overlay Grad-CAM heatmap on original image.

    Args:
        image: Original image (H, W, 3) in uint8 [0, 255].
        heatmap: Grad-CAM heatmap (H, W) in [0, 1].
        title: Plot title.
        save_path: Path to save figure.
        alpha: Overlay transparency.

    Returns:
        Matplotlib Figure.
    """
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_gradcam_batch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    save_dir: str | Path,
    num_samples: int = 20,
    misclassified_only: bool = False,
) -> None:
    """Generate Grad-CAM visualizations for a batch of images.

    Args:
        model: Trained model.
        dataloader: DataLoader with test images.
        class_names: Ordered class names.
        device: Compute device.
        save_dir: Directory to save visualizations.
        num_samples: Number of samples to visualize.
        misclassified_only: Only show misclassified examples.
    """
    from src.data.transforms import denormalize

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    target_layer = get_target_layer(model)
    count = 0

    for batch in dataloader:
        if count >= num_samples:
            break

        images = batch["image"].to(device)
        labels = batch["label"]

        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()

        # Generate Grad-CAM
        heatmaps = generate_gradcam(model, images, target_layer=target_layer)

        for i in range(images.shape[0]):
            if count >= num_samples:
                break

            true_label = labels[i].item()
            pred_label = preds[i].item()

            if misclassified_only and true_label == pred_label:
                continue

            # Denormalize image for display
            img_np = denormalize(images[i].cpu().numpy())
            correct = "CORRECT" if true_label == pred_label else "WRONG"

            title = (
                f"{correct} | True: {class_names[true_label]} | "
                f"Pred: {class_names[pred_label]}"
            )

            visualize_gradcam(
                img_np, heatmaps[i],
                title=title,
                save_path=save_dir / f"gradcam_{count:03d}_{correct.lower()}.png",
            )
            count += 1

    plt.close("all")
    logger.info("Generated %d Grad-CAM visualizations in %s", count, save_dir)
