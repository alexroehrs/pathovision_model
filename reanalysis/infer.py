"""Inference helpers shared by the prediction, hybrid, Grad-CAM and CV stages.

All functions operate on folder-per-class image trees (the layout used by
``datasets/data/<split>/<class>/`` and ``datasets/external_unified/<class>/``)
and return pandas DataFrames with one row per image: ``path``, ``class``
(ground truth), ``pred`` (argmax class), ``confidence`` and ``proba_<class>``
columns.  Image-level perturbations used for the sensitivity analyses
(brightness normalisation, central crop) are implemented here so that every
stage applies exactly the same transformation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from common import CLASSES, PROJECT, log

sys.path.insert(0, str(PROJECT))

from src.data.dataset import SkinDiseaseDataset          # noqa: E402
from src.data.transforms import get_val_transforms       # noqa: E402
from src.models.factory import build_model               # noqa: E402

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(ckpt_path: Path, device: torch.device | None = None):
    """Return (model, config) with the model in eval mode on ``device``."""
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config


# --------------------------------------------------------------------------
# Image-level perturbations
# --------------------------------------------------------------------------
def brightness_normalise(image_rgb: np.ndarray, target_mean_l: float) -> np.ndarray:
    """Shift the CIELAB L* channel so the image mean L* equals ``target_mean_l``.

    L* is handled as float in [0, 100] (OpenCV's 8-bit Lab stores L*·255/100).
    Chromatic channels are left unchanged.  This removes the global-brightness
    cue identified in the Normal class while preserving texture and colour.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_star = lab[:, :, 0] * (100.0 / 255.0)
    l_star = np.clip(l_star + (target_mean_l - l_star.mean()), 0, 100)
    lab[:, :, 0] = l_star * (255.0 / 100.0)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def central_crop(image_rgb: np.ndarray, keep: float = 0.70) -> np.ndarray:
    """Keep the central ``keep`` fraction of height and width.

    Watermarks and captions in web-scraped dermatology images sit near the
    borders, so a central crop removes them while retaining the lesion.
    """
    h, w = image_rgb.shape[:2]
    dh, dw = int(round(h * (1 - keep) / 2)), int(round(w * (1 - keep) / 2))
    return image_rgb[dh:h - dh, dw:w - dw]


def mean_l_star(image_rgb: np.ndarray) -> float:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean() * (100.0 / 255.0))


def dataset_mean_l_star(root: Path, max_per_class: int = 200) -> float:
    """Mean L* of (a subsample of) a folder-per-class tree, used as the target
    brightness for ``brightness_normalise``."""
    vals = []
    for cls in CLASSES:
        d = root / cls
        if not d.exists():
            continue
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in VALID_EXTENSIONS)[:max_per_class]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                vals.append(mean_l_star(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return float(np.mean(vals))


class PerturbedDataset(SkinDiseaseDataset):
    """SkinDiseaseDataset with an optional RGB-level perturbation applied
    before the validation transform."""

    def __init__(self, root_dir, transform, classes, perturb=None):
        super().__init__(root_dir, transform, classes)
        self.perturb = perturb

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.perturb is not None:
            image = self.perturb(image)
        image = self.transform(image=image)["image"]
        return {"image": image, "label": label, "class_name": self.idx_to_class[label], "path": str(img_path)}


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_folder(model, config, root: Path, perturb=None, batch_size: int = 32,
                   num_workers: int = 2, with_features: bool = False,
                   device: torch.device | None = None):
    """Predict every image under ``root/<class>/``.

    Returns a DataFrame (and, if ``with_features``, an (N, D) embedding array
    in the same row order).  Also records mean per-image latency in
    ``df.attrs['latency_ms']``.
    """
    device = device or get_device()
    transform = get_val_transforms(config)
    ds = PerturbedDataset(root, transform, CLASSES, perturb=perturb)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=device.type == "cuda")
    rows, feats = [], []
    t_infer = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer += time.perf_counter() - t0
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        if with_features:
            feats.append(model.extract_features(x).float().cpu().numpy())
        for i in range(len(probs)):
            row = {"path": batch["path"][i], "class": batch["class_name"][i],
                   "pred": CLASSES[int(probs[i].argmax())], "confidence": float(probs[i].max())}
            for j, c in enumerate(CLASSES):
                row[f"proba_{c}"] = float(probs[i, j])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["latency_ms"] = 1000.0 * t_infer / max(len(df), 1)
    if with_features:
        return df, (np.concatenate(feats) if feats else np.zeros((0, 0)))
    return df


def relative_paths(df: pd.DataFrame, root: Path) -> pd.Series:
    """Paths relative to the dataset root (``<split>/<class>/<file>`` or
    ``<class>/<file>``), which are stable across machines."""
    return df["path"].map(lambda p: str(Path(p).resolve().relative_to(root.resolve())))
