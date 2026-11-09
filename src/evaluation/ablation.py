"""Ablation study orchestrator.

Runs multiple configs and collects results into a comparison table.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


ABLATION_CONFIGS = {
    "Baseline": "configs/base.yaml",
    "No Augmentation": "configs/ablation/no_augmentation.yaml",
    "No Dropout": "configs/ablation/no_dropout.yaml",
    "No Freeze (full fine-tune)": "configs/ablation/no_freeze.yaml",
    "Resolution 260×260": "configs/ablation/resolution_260.yaml",
    "Resolution 384×384": "configs/ablation/resolution_384.yaml",
    "CE Loss Only (no weights)": "configs/ablation/ce_loss_only.yaml",
}


def run_ablation_experiment(
    config_path: str,
    experiment_name: str,
    python_exe: str = sys.executable,
) -> dict[str, Any] | None:
    """Run a single ablation experiment via subprocess.

    Args:
        config_path: Path to the YAML config.
        experiment_name: Name for logging.
        python_exe: Python executable path.

    Returns:
        Dictionary with metrics, or None on failure.
    """
    logger.info("Running ablation: %s (%s)", experiment_name, config_path)

    cmd = [
        python_exe, "scripts/train.py",
        "--config", config_path,
        f"experiment.name={experiment_name.replace(' ', '_').lower()}",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
        )
        if result.returncode != 0:
            logger.error("Ablation '%s' failed: %s", experiment_name, result.stderr[-500:])
            return None
        logger.info("Ablation '%s' completed", experiment_name)
        return {"status": "ok"}
    except subprocess.TimeoutExpired:
        logger.error("Ablation '%s' timed out (2h limit)", experiment_name)
        return None


def collect_ablation_results(
    results_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """Collect metrics from all ablation experiment checkpoints.

    Args:
        results_dir: Base output directory.

    Returns:
        DataFrame with ablation comparison.
    """
    import torch

    results_dir = Path(results_dir)
    rows = []

    for ckpt_path in sorted(results_dir.glob("**/best.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            metrics = ckpt.get("metrics", {})
            config = ckpt.get("config", {})
            name = config.get("experiment", {}).get("name", ckpt_path.parent.name)

            rows.append({
                "Experiment": name,
                "Val Accuracy (%)": metrics.get("val_accuracy", 0) * 100,
                "Val Loss": metrics.get("val_loss", 0),
                "Epoch": ckpt.get("epoch", 0),
                "Checkpoint": str(ckpt_path),
            })
        except Exception as e:
            logger.warning("Could not load %s: %s", ckpt_path, e)

    return pd.DataFrame(rows)


def generate_ablation_latex(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> str:
    """Generate LaTeX table from ablation results.

    Args:
        df: DataFrame with ablation results.
        save_path: Optional path to save .tex file.

    Returns:
        LaTeX table string.
    """
    latex = df[["Experiment", "Val Accuracy (%)", "Val Loss"]].to_latex(
        index=False,
        float_format="%.2f",
        caption="Ablation study results. Each row removes or modifies one component.",
        label="tab:ablation",
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex)
        logger.info("Ablation LaTeX table saved to %s", save_path)

    return latex
