"""Full evaluation report generation."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.confusion import plot_confusion_matrices
from src.evaluation.curves import plot_precision_recall_curves, plot_roc_curves
from src.evaluation.metrics import (
    compare_with_reference_baseline,
    compute_full_report,
)
from src.utils.device import get_amp_dtype

logger = logging.getLogger(__name__)


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_mixed_precision: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on a dataloader.

    Args:
        model: Trained model.
        dataloader: Test/val DataLoader.
        device: Compute device.
        use_mixed_precision: Whether to use AMP.

    Returns:
        Tuple of (y_true, y_pred, y_proba) as numpy arrays.
    """
    model.eval()
    amp_dtype = get_amp_dtype(device)

    all_labels = []
    all_preds = []
    all_probas = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_mixed_precision):
                logits = model(images)

            logits = logits.float()  # ensure float32 for softmax/numpy
            probas = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.append(labels.numpy())
            all_preds.append(preds)
            all_probas.append(probas)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probas)

    return y_true, y_pred, y_proba


def generate_full_report(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """Generate complete evaluation report with all metrics and visualizations.

    Args:
        model: Trained model.
        dataloader: Test DataLoader.
        class_names: Ordered class names.
        config: Configuration dictionary.
        device: Compute device.
        output_dir: Directory to save outputs.

    Returns:
        Full report dictionary.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = config["experiment"].get("name", "pathovision")
    use_mp = config["training"].get("mixed_precision", True)

    # Run inference
    logger.info("Running inference on test set...")
    y_true, y_pred, y_proba = run_inference(model, dataloader, device, use_mp)

    # Compute metrics
    logger.info("Computing metrics...")
    report = compute_full_report(y_true, y_pred, class_names)

    # Comparison with reference values
    comparison = compare_with_reference_baseline(report["global"], report["per_class"])
    report["comparison"] = comparison

    # Print report
    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {experiment_name}")
    print("=" * 70)
    print(f"\nTest samples: {len(y_true)}")
    print(f"\nGlobal Metrics:")
    for k, v in report["global"].items():
        print(f"  {k}: {v:.2f}%")

    print(f"\nPer-Class Metrics:")
    print(report["per_class"].to_string(index=False, float_format="%.2f"))

    print(f"\nComparison with reference values:")
    print(comparison.to_string(index=False, float_format="%.2f"))

    print(f"\nClassification Report (sklearn):\n{report['classification_report']}")

    # Save CSV
    report["per_class"].to_csv(reports_dir / f"{experiment_name}_per_class.csv", index=False)
    comparison.to_csv(reports_dir / f"{experiment_name}_comparison.csv", index=False)

    # Save LaTeX table
    latex_str = report["per_class"][
        ["Class", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"]
    ].to_latex(index=False, float_format="%.2f", caption="Per-class results", label="tab:results")
    with open(reports_dir / f"{experiment_name}_table.tex", "w") as f:
        f.write(latex_str)

    # Generate plots
    if config.get("evaluation", {}).get("generate_plots", True):
        logger.info("Generating plots...")

        plot_confusion_matrices(
            report["confusion_matrix"], class_names,
            save_dir=figures_dir, experiment_name=experiment_name,
        )

        auc_scores = plot_roc_curves(
            y_true, y_proba, class_names,
            save_path=figures_dir / f"{experiment_name}_roc_curves.png",
        )
        report["auc_scores"] = auc_scores

        ap_scores = plot_precision_recall_curves(
            y_true, y_proba, class_names,
            save_path=figures_dir / f"{experiment_name}_pr_curves.png",
        )
        report["ap_scores"] = ap_scores

    logger.info("Report saved to %s", reports_dir)
    return report
