"""Per-class and global evaluation metrics for skin disease classification."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Compute per-class metrics matching the paper's Table B.3.

    Metrics: Accuracy, Recall (Sensitivity), Specificity, Precision, F1-Score,
    Balanced Accuracy.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        class_names: Ordered list of class names.

    Returns:
        DataFrame with one row per class and metric columns.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    n_classes = len(class_names)
    total = len(y_true)

    rows = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2

        rows.append({
            "Class": cls,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": accuracy * 100,
            "Sensitivity": sensitivity * 100,
            "Specificity": specificity * 100,
            "Precision": precision * 100,
            "F1-Score": f1 * 100,
            "Balanced Accuracy": balanced_acc * 100,
        })

    df = pd.DataFrame(rows)
    return df


def compute_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute global (aggregate) metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of global metric values (as percentages).
    """
    return {
        "global_accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100,
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    }


def compute_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute complete evaluation report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Ordered list of class names.

    Returns:
        Dictionary with 'per_class' (DataFrame), 'global' (dict),
        'confusion_matrix' (ndarray), and 'classification_report' (str).
    """
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)
    global_metrics = compute_global_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    sklearn_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report = {
        "per_class": per_class,
        "global": global_metrics,
        "confusion_matrix": cm,
        "classification_report": sklearn_report,
    }

    # Log summary
    logger.info("Global Accuracy: %.2f%%", global_metrics["global_accuracy"])
    logger.info("Weighted F1: %.2f%%", global_metrics["f1_weighted"])

    return report


def compare_with_reference_baseline(
    global_metrics: dict[str, float],
    per_class_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare per-class F1 with reference values of the EfficientNetB2 baseline.

    Args:
        global_metrics: Computed global metrics.
        per_class_df: Computed per-class DataFrame.

    Returns:
        Comparison DataFrame.
    """
    reference_f1 = {
        "Acne": 91.88,
        "Candidiasis": 52.17,
        "Eczema": 81.82,
        "NailFungus": 87.32,
        "Normal": 100.00,
        "Psoriasis": 75.34,
        "Tinea": 55.92,
    }

    comparison_rows = []
    for _, row in per_class_df.iterrows():
        cls = row["Class"]
        ref = reference_f1.get(cls, None)
        ours = row["F1-Score"]
        diff = ours - ref if ref is not None else None
        comparison_rows.append({
            "Class": cls,
            "Reference F1": ref,
            "Ours F1": ours,
            "Diff": diff,
        })

    # Add global
    comparison_rows.append({
        "Class": "GLOBAL (weighted)",
        "Reference F1": 83.12,
        "Ours F1": global_metrics["f1_weighted"],
        "Diff": global_metrics["f1_weighted"] - 83.12,
    })

    return pd.DataFrame(comparison_rows)


# ============================================================
# Statistical rigor: confidence intervals and tests
# ============================================================


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        metric_fn: Function(y_true, y_pred) -> float.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_estimate = metric_fn(y_true, y_pred)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(bootstrap_scores, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_scores, (1 - alpha) * 100))

    return float(point_estimate), ci_lower, ci_upper


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict[str, dict[str, float]]:
    """Compute global metrics with 95% bootstrap confidence intervals.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dictionary mapping metric name to {value, ci_lower, ci_upper}.
    """
    metrics_fns = {
        "accuracy": lambda yt, yp: accuracy_score(yt, yp) * 100,
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp) * 100,
        "f1_weighted": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0) * 100,
        "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0) * 100,
    }

    results = {}
    for name, fn in metrics_fns.items():
        value, ci_lower, ci_upper = bootstrap_confidence_interval(
            y_true, y_pred, fn, n_bootstrap=n_bootstrap,
        )
        results[name] = {
            "value": value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
        logger.info(
            "%s: %.2f%% (95%% CI: [%.2f, %.2f])",
            name, value, ci_lower, ci_upper,
        )

    return results


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """McNemar's test for comparing two classifiers.

    Tests whether two models make significantly different errors.

    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.

    Returns:
        Dictionary with 'statistic', 'p_value', 'significant' (at alpha=0.05).
    """
    # Build contingency table
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # McNemar's test with continuity correction
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # Chi-squared distribution with 1 df
    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(statistic, df=1))

    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "b_correct_a_only": b,
        "c_correct_b_only": c,
    }


def format_cv_results(
    fold_results: list[dict[str, float]],
) -> pd.DataFrame:
    """Format cross-validation results with mean ± std and 95% CI.

    Args:
        fold_results: List of metric dictionaries, one per fold.

    Returns:
        DataFrame with formatted results for the paper.
    """
    df = pd.DataFrame(fold_results)

    rows = []
    for col in df.columns:
        values = df[col].values
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ci_lower = mean - 1.96 * std / np.sqrt(len(values))
        ci_upper = mean + 1.96 * std / np.sqrt(len(values))

        rows.append({
            "Metric": col,
            "Mean": mean,
            "Std": std,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            "Formatted": f"{mean:.2f} ± {std:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]",
        })

    return pd.DataFrame(rows)
