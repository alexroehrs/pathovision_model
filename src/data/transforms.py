"""Image augmentation and preprocessing pipelines for PathoVision."""

import logging
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

# EfficientNet ImageNet normalization (same as TF preprocessing)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(config: dict[str, Any]) -> A.Compose:
    """Build training augmentation pipeline matching the project.

    Project augmentation: rotation(±20°), shift(±20%), horizontal flip,
    zoom(±20%), shear(±20°), EfficientNet normalization, resize 500x500.

    Args:
        config: Configuration dictionary with 'augmentation' and 'data' keys.

    Returns:
        Albumentations Compose pipeline.
    """
    aug = config.get("augmentation", {})
    input_size = config["data"]["input_size"]
    h, w = input_size[0], input_size[1]

    transforms_list = [
        # Resize to target size
        A.Resize(height=h, width=w),

        # Project augmentation
        A.Rotate(limit=aug.get("rotation_limit", 20), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=aug.get("shift_limit", 0.2),
            scale_limit=aug.get("zoom_limit", 0.2),
            rotate_limit=0,  # rotation handled above
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5 if aug.get("horizontal_flip", True) else 0.0),
        A.Affine(
            shear={"x": (-aug.get("shear_limit", 20), aug.get("shear_limit", 20)),
                   "y": (-aug.get("shear_limit", 20), aug.get("shear_limit", 20))},
            p=0.3,
        ),
    ]

    # Enhanced augmentation (only if configured)
    if aug.get("color_jitter"):
        cj = aug["color_jitter"]
        transforms_list.append(
            A.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.05),
                p=0.5,
            )
        )

    if aug.get("clahe"):
        cl = aug["clahe"]
        transforms_list.append(
            A.CLAHE(
                clip_limit=cl.get("clip_limit", 2.0),
                p=cl.get("probability", 0.3),
            )
        )

    if aug.get("elastic"):
        el = aug["elastic"]
        transforms_list.append(
            A.ElasticTransform(
                alpha=el.get("alpha", 50),
                sigma=el.get("sigma", 5),
                p=el.get("probability", 0.1),
            )
        )

    # Normalization and tensor conversion (always applied)
    transforms_list.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    # Random erasing (applied after tensor conversion)
    pipeline = A.Compose(transforms_list)

    logger.info(
        "Train transforms: %d augmentation steps, input=%dx%d",
        len(transforms_list), h, w,
    )
    return pipeline


def get_val_transforms(config: dict[str, Any]) -> A.Compose:
    """Build validation/test preprocessing pipeline (no augmentation).

    Only applies: resize + EfficientNet normalization.

    Args:
        config: Configuration dictionary with 'data' key.

    Returns:
        Albumentations Compose pipeline.
    """
    input_size = config["data"]["input_size"]
    h, w = input_size[0], input_size[1]

    pipeline = A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    logger.info("Val/Test transforms: resize + normalize, input=%dx%d", h, w)
    return pipeline


def get_tta_transforms(config: dict[str, Any]) -> list[A.Compose]:
    """Build Test-Time Augmentation transforms.

    Returns multiple transform pipelines for TTA averaging.

    Args:
        config: Configuration dictionary.

    Returns:
        List of Albumentations Compose pipelines.
    """
    input_size = config["data"]["input_size"]
    h, w = input_size[0], input_size[1]

    base_normalize = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    tta_list = [
        # Original (no augmentation)
        A.Compose([A.Resize(h, w)] + base_normalize),
        # Horizontal flip
        A.Compose([A.Resize(h, w), A.HorizontalFlip(p=1.0)] + base_normalize),
        # Slight rotation
        A.Compose([A.Resize(h, w), A.Rotate(limit=10, p=1.0)] + base_normalize),
        # Slight zoom
        A.Compose([
            A.Resize(h, w),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=0, p=1.0),
        ] + base_normalize),
        # Center crop + resize
        A.Compose([
            A.Resize(int(h * 1.1), int(w * 1.1)),
            A.CenterCrop(h, w),
        ] + base_normalize),
    ]

    return tta_list


def denormalize(tensor: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image array (H, W, 3) or (3, H, W).

    Returns:
        Denormalized image in [0, 255] uint8 range.
    """
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    if tensor.shape[0] == 3:  # CHW -> HWC
        tensor = tensor.transpose(1, 2, 0)

    img = tensor * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
