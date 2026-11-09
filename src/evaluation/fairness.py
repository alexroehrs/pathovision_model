"""Fairness analysis — skin tone stratification and equity reporting.

Analyzes model performance across skin tones
(Fitzpatrick I-VI or Monk Skin Tone scale).
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fitzpatrick Skin Type approximate ITA (Individual Typology Angle) thresholds
# ITA = arctan((L - 50) / b) * (180 / pi), where L and b are from CIELab
# Higher ITA = lighter skin
FITZPATRICK_ITA_THRESHOLDS = {
    "FST I-II (Light)": (41, 90),      # Very light to light
    "FST III (Medium)": (28, 41),       # Medium
    "FST IV (Olive)": (19, 28),         # Olive
    "FST V-VI (Dark)": (-90, 19),       # Dark to very dark
}


def compute_ita_from_image(image: np.ndarray) -> float:
    """Compute Individual Typology Angle (ITA) from an image.

    ITA is a proxy for skin tone based on the CIELab color space.
    Higher ITA values indicate lighter skin.

    Args:
        image: RGB image array (H, W, 3).

    Returns:
        ITA value (float).
    """
    # Convert to CIELab
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Use center region (avoid borders/artifacts)
    h, w = lab.shape[:2]
    margin = 0.2
    y1, y2 = int(h * margin), int(h * (1 - margin))
    x1, x2 = int(w * margin), int(w * (1 - margin))
    center = lab[y1:y2, x1:x2]

    # Median L and b values
    L = np.median(center[:, :, 0])  # Lightness
    b = np.median(center[:, :, 2])  # Yellow-blue

    # ITA formula
    ita = np.arctan2(L - 50, b) * (180.0 / np.pi)
    return float(ita)


def classify_skin_tone(ita: float) -> str:
    """Classify skin tone based on ITA value.

    Args:
        ita: Individual Typology Angle.

    Returns:
        Fitzpatrick group string.
    """
    for group, (low, high) in FITZPATRICK_ITA_THRESHOLDS.items():
        if low <= ita < high:
            return group
    return "Unknown"


def analyze_dataset_skin_tones(
    dataset_dir: str | Path,
    classes: list[str],
    max_per_class: int = 200,
) -> pd.DataFrame:
    """Analyze skin tone distribution across the dataset.

    Args:
        dataset_dir: Root of dataset split (e.g., datasets/merged/test).
        classes: List of class names.
        max_per_class: Max images to sample per class.

    Returns:
        DataFrame with columns: path, class, ita, skin_tone_group.
    """
    dataset_dir = Path(dataset_dir)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    records = []

    for cls in classes:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue

        images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in valid_ext])
        if max_per_class > 0:
            images = images[:max_per_class]

        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ita = compute_ita_from_image(img_rgb)
                tone = classify_skin_tone(ita)
                records.append({
                    "path": str(img_path),
                    "class": cls,
                    "ita": ita,
                    "skin_tone_group": tone,
                })
            except Exception as e:
                logger.warning("Failed to process %s: %s", img_path, e)

    df = pd.DataFrame(records)
    logger.info("Analyzed %d images for skin tone distribution", len(df))
    return df


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    skin_tones: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    """Compute per-group accuracy and F1 stratified by skin tone.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        skin_tones: Skin tone group for each sample.
        class_names: Ordered class names.

    Returns:
        DataFrame with metrics per skin tone group.
    """
    from sklearn.metrics import accuracy_score, f1_score

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "skin_tone": skin_tones,
    })

    rows = []
    for group in sorted(df["skin_tone"].unique()):
        mask = df["skin_tone"] == group
        group_df = df[mask]

        if len(group_df) < 5:
            continue

        acc = accuracy_score(group_df["y_true"], group_df["y_pred"]) * 100
        f1_w = f1_score(group_df["y_true"], group_df["y_pred"],
                        average="weighted", zero_division=0) * 100
        f1_m = f1_score(group_df["y_true"], group_df["y_pred"],
                        average="macro", zero_division=0) * 100

        rows.append({
            "Skin Tone Group": group,
            "N Samples": len(group_df),
            "Accuracy (%)": acc,
            "F1 Weighted (%)": f1_w,
            "F1 Macro (%)": f1_m,
        })

    return pd.DataFrame(rows)


def plot_skin_tone_distribution(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot skin tone distribution by class.

    Args:
        df: DataFrame from analyze_dataset_skin_tones.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution across dataset
    tone_counts = df["skin_tone_group"].value_counts().sort_index()
    tone_counts.plot(kind="bar", ax=axes[0], color=["#ffd93d", "#f9a825", "#c17817", "#6d4c41"])
    axes[0].set_title("Skin Tone Distribution (Overall)")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    # Distribution per class
    ct = pd.crosstab(df["class"], df["skin_tone_group"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=axes[1],
            color=["#ffd93d", "#f9a825", "#c17817", "#6d4c41"])
    axes[1].set_title("Skin Tone Distribution per Class (%)")
    axes[1].set_ylabel("Percentage")
    axes[1].legend(title="Tone", bbox_to_anchor=(1.05, 1), fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Skin tone distribution saved to %s", save_path)

    return fig


def plot_fairness_comparison(
    fairness_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot accuracy/F1 comparison across skin tone groups.

    Args:
        fairness_df: DataFrame from compute_fairness_metrics.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(fairness_df))
    width = 0.25

    ax.bar(x - width, fairness_df["Accuracy (%)"], width, label="Accuracy", color="#3498db")
    ax.bar(x, fairness_df["F1 Weighted (%)"], width, label="F1 Weighted", color="#2ecc71")
    ax.bar(x + width, fairness_df["F1 Macro (%)"], width, label="F1 Macro", color="#e74c3c")

    ax.set_xlabel("Skin Tone Group")
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance by Skin Tone Group")
    ax.set_xticks(x)
    ax.set_xticklabels(fairness_df["Skin Tone Group"], rotation=15)
    ax.legend()
    ax.set_ylim(0, 105)

    # Add sample counts
    for i, row in fairness_df.iterrows():
        ax.text(i, 2, f"n={row['N Samples']}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
