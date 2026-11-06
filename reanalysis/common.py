"""Shared constants and helpers for the PathoVision re-execution package.

Every script in ``reanalysis/`` imports from here so that model identifiers,
seeds, directory conventions and statistical helpers are defined once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent          # repository root
REANALYSIS = PROJECT / "reanalysis"
# Large, persistent artefacts (images, checkpoints, predictions, results) live
# under REANALYSIS_ROOT when that variable is set (e.g. a Google Drive folder
# in Colab, so that they survive session resets); otherwise inside the repo.
_ROOT = Path(os.environ["REANALYSIS_ROOT"]).expanduser().resolve() if os.environ.get("REANALYSIS_ROOT") else None
RUNS = (_ROOT / "runs") if _ROOT else REANALYSIS / "runs"            # checkpoints and predictions
RESULTS = (_ROOT / "results") if _ROOT else REANALYSIS / "results"   # tables, figures, reports
# Images are read thousands of times per epoch, so they should sit on a fast
# local disk: REANALYSIS_DATA_ROOT overrides their location independently.
_DROOT = Path(os.environ["REANALYSIS_DATA_ROOT"]).expanduser().resolve() if os.environ.get("REANALYSIS_DATA_ROOT") else None
DATASETS = _DROOT if _DROOT else ((_ROOT / "datasets") if _ROOT else PROJECT / "datasets")
DATA_DIR = DATASETS / "data"                              # train/val/test after split
MERGED_DIR = DATASETS / "merged"                          # inherited split, deduplicated
RAW_DIR = DATASETS / "raw"
EXTERNAL_DIR = DATASETS / "external_unified"
EXTERNAL_RAW_DIR = DATASETS / "external_raw"

CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]
CLASS_LABELS = {  # display names used in tables and figures
    "Acne": "Acne", "Candidiasis": "Candidiasis", "Eczema": "Eczema",
    "NailFungus": "Nail fungus", "Normal": "Normal", "Psoriasis": "Psoriasis", "Tinea": "Tinea",
}

# Model identifiers of the paper -> training configuration (unchanged from the
# original repository).  The order matters for tables.
MODEL_CONFIGS = {
    "M1":  "configs/base.yaml",
    "M1b": "configs/balanced_ce.yaml",
    "M5":  "configs/balanced.yaml",
    "M2b": "configs/improved_nobal.yaml",
    "M2":  "configs/improved.yaml",
    "M3b": "configs/swin_tiny_nobal.yaml",
    "M3":  "configs/swin_tiny.yaml",
    "M4b": "configs/convnext_tiny_nobal.yaml",
    "M4":  "configs/convnext_tiny.yaml",
}
MODEL_BACKBONE = {
    "M1": "EfficientNetB2", "M1b": "EfficientNetB2", "M5": "EfficientNetB2",
    "M2b": "EfficientNetV2-S", "M2": "EfficientNetV2-S",
    "M3b": "Swin-Tiny", "M3": "Swin-Tiny",
    "M4b": "ConvNeXt-Tiny", "M4": "ConvNeXt-Tiny",
}
MODEL_BALANCING = {
    "M1": "No", "M1b": "CE+Bal", "M5": "Focal+Bal", "M2b": "No", "M2": "Focal+Bal",
    "M3b": "No", "M3": "Focal+Bal", "M4b": "No", "M4": "Focal+Bal",
}
ENSEMBLE_MEMBERS = ["M2", "M3", "M4"]      # soft-voting ensemble M6
HYBRID_BASE = "M4"                          # H1: random forest on M4 embeddings
PRIMARY = "M4"
BASELINE = "M1"
CV_MODELS = ["M1", "M4"]

SEEDS = [42, 43, 44]

# Evaluation splits produced by 04_predict_all.py.  "variants" are
# re-evaluations of an existing split with a controlled perturbation.
SPLITS = ["val", "test", "external"]
VARIANTS = ["test_brightnorm", "test_centercrop", "external_centercrop"]


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(model_id: str, seed: int) -> Path:
    return RUNS / model_id / f"seed{seed}"


def checkpoint_path(model_id: str, seed: int) -> Path:
    return run_dir(model_id, seed) / "checkpoints" / "best.pt"


def pred_path(model_id: str, seed: int, split: str) -> Path:
    return run_dir(model_id, seed) / f"pred_{split}.csv"


def available_runs(model_ids=None, seeds=None):
    """Yield (model_id, seed) for every run whose checkpoint exists."""
    for mid in (model_ids or list(MODEL_CONFIGS)):
        for s in (seeds or SEEDS):
            if checkpoint_path(mid, s).exists():
                yield mid, s


def available_predictions(split: str, model_ids=None, seeds=None):
    """Yield (model_id, seed, DataFrame) for every prediction file of a split.

    Ensemble (M6) and hybrid (H1) predictions are written by 05 to the same
    layout, so they are picked up here as well when their ids are requested.
    """
    for mid in (model_ids or list(MODEL_CONFIGS) + ["M6", "H1"]):
        for s in (seeds or SEEDS):
            p = pred_path(mid, s, split)
            if p.exists():
                yield mid, s, pd.read_csv(p)


def sh(cmd: list[str], cwd: Path = PROJECT, env: dict | None = None) -> float:
    """Run a command, stream its output, return the elapsed time in seconds."""
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    full_env = dict(os.environ)
    full_env["PYTHONPATH"] = str(PROJECT)
    if env:
        full_env.update(env)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True, env=full_env)
    return time.time() - t0


def write_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_table(df: pd.DataFrame, name: str, subdir: str = "tables", float_format: str = "%.2f") -> Path:
    """Save a table as CSV and LaTeX (booktabs) under results/<subdir>/."""
    out = ensure_dir(RESULTS / subdir)
    df.to_csv(out / f"{name}.csv", index=False)
    try:
        with open(out / f"{name}.tex", "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, float_format=float_format, escape=True))
    except Exception as exc:  # pragma: no cover - LaTeX export is best effort
        print(f"[warn] LaTeX export failed for {name}: {exc}")
    return out / f"{name}.csv"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * (centre - half), 100 * (centre + half))


def bootstrap_ci(values_fn, n: int, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic computed by ``values_fn(idx)``.

    ``values_fn`` receives an index array (resampled with replacement) and
    returns the statistic.  Returns the 2.5 and 97.5 percentiles.
    """
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = values_fn(idx)
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)) * 100)


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, classes=CLASSES) -> pd.DataFrame:
    """Per-class support, recall, precision, F1 (percent) with Wilson CI of recall."""
    from sklearn.metrics import precision_recall_fscore_support
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    rows = []
    for i, c in enumerate(classes):
        k = int(np.sum((y_true == c) & (y_pred == c)))
        lo, hi = wilson_ci(k, int(s[i]))
        rows.append({"class": CLASS_LABELS[c], "support": int(s[i]), "recall": 100 * r[i],
                     "recall_ci_lo": lo, "recall_ci_hi": hi,
                     "precision": 100 * p[i], "f1": 100 * f[i]})
    return pd.DataFrame(rows)


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes=CLASSES) -> dict:
    from sklearn.metrics import f1_score, balanced_accuracy_score
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true); k = int(np.sum(y_true == y_pred))
    lo, hi = wilson_ci(k, n)
    if n == 0:
        return {"n": 0, "accuracy": np.nan, "acc_ci_lo": np.nan, "acc_ci_hi": np.nan,
                "balanced_accuracy": np.nan, "f1_weighted": np.nan, "f1_macro": np.nan}
    return {
        "n": n,
        "accuracy": 100 * k / n,
        "acc_ci_lo": lo, "acc_ci_hi": hi,
        "balanced_accuracy": 100 * balanced_accuracy_score(y_true, y_pred),
        "f1_weighted": 100 * f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        "f1_macro": 100 * f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
    }


def mean_sd(values) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.2f}"
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"


def proba_matrix(df: pd.DataFrame, classes=CLASSES) -> np.ndarray:
    return df[[f"proba_{c}" for c in classes]].to_numpy(dtype=float)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
