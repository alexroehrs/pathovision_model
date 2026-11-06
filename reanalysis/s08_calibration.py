"""Stage 08 - calibration and selective prediction (CPU, from prediction CSVs).

For every model/seed with val, test (and external) predictions:

* expected and maximum calibration error (ECE/MCE, 15 equal-width bins),
  Brier score and negative log-likelihood, before and after temperature
  scaling (temperature fitted on the validation split by minimising NLL);
* selective prediction: accuracy among images whose confidence exceeds a
  threshold, and the fraction retained (coverage), for thresholds 0.5-0.95;
* reliability-diagram data (per-bin confidence and accuracy) for figures.

Outputs ``results/tables/calibration.csv``, ``selective_prediction.csv`` and
``results/calibration/reliability_<split>_<model>.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, MODEL_CONFIGS, RESULTS, SEEDS, available_predictions, ensure_dir, log, mean_sd, proba_matrix, save_table  # noqa: E402

MODELS = list(MODEL_CONFIGS) + ["M6", "H1"]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def ece_mce(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece, mce, rows = 0.0, 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap; mce = max(mce, gap)
            rows.append({"bin_lo": bins[b], "bin_hi": bins[b + 1], "n": int(m.sum()),
                         "mean_confidence": conf[m].mean(), "accuracy": correct[m].mean()})
    return 100 * ece, 100 * mce, pd.DataFrame(rows)


def nll_brier(P, y_idx):
    p_true = np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, 1)
    onehot = np.eye(P.shape[1])[y_idx]
    return float(-np.log(p_true).mean()), float(((P - onehot) ** 2).sum(1).mean())


def temperature_scale(P, T):
    logp = np.log(np.clip(P, 1e-12, 1)) / T
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def fit_temperature(P_val, y_val):
    res = minimize_scalar(lambda T: nll_brier(temperature_scale(P_val, T), y_val)[0], bounds=(0.25, 10), method="bounded")
    return float(res.x)


def main() -> None:
    val = {(m, s): d for m, s, d in available_predictions("val", MODELS)}
    rows, sel_rows = [], []
    out_dir = ensure_dir(RESULTS / "calibration")
    for split in ("test", "external"):
        for mid, seed, df in available_predictions(split, MODELS):
            P = proba_matrix(df); y = df["class"].map(CLASSES.index).to_numpy()
            T = fit_temperature(proba_matrix(val[(mid, seed)]), val[(mid, seed)]["class"].map(CLASSES.index).to_numpy()) if (mid, seed) in val else np.nan
            for tag, PP in (("raw", P), ("temp_scaled", temperature_scale(P, T) if np.isfinite(T) else None)):
                if PP is None:
                    continue
                conf = PP.max(1); correct = (PP.argmax(1) == y).astype(float)
                ece, mce, rel = ece_mce(conf, correct)
                nll, brier = nll_brier(PP, y)
                rows.append({"split": split, "model": mid, "seed": seed, "calibration": tag, "temperature": T if tag == "temp_scaled" else 1.0,
                             "ece": ece, "mce": mce, "nll": nll, "brier": brier, "mean_confidence": 100 * conf.mean(), "accuracy": 100 * correct.mean()})
                if tag == "raw":
                    rel.to_csv(out_dir / f"reliability_{split}_{mid}_seed{seed}.csv", index=False)
                for thr in THRESHOLDS:
                    keep = conf >= thr
                    sel_rows.append({"split": split, "model": mid, "seed": seed, "calibration": tag, "threshold": thr,
                                     "coverage": 100 * keep.mean(), "accuracy_retained": 100 * correct[keep].mean() if keep.any() else np.nan,
                                     "accuracy_rejected": 100 * correct[~keep].mean() if (~keep).any() else np.nan})
    cal = pd.DataFrame(rows)
    save_table(cal, "calibration_per_run")
    agg = cal.groupby(["split", "model", "calibration"], sort=False).agg(
        n_seeds=("seed", "count"), ece=("ece", mean_sd), mce=("mce", mean_sd), nll=("nll", mean_sd), brier=("brier", mean_sd),
        mean_confidence=("mean_confidence", mean_sd), accuracy=("accuracy", mean_sd), temperature=("temperature", mean_sd)).reset_index()
    save_table(agg, "calibration")
    sel = pd.DataFrame(sel_rows)
    save_table(sel, "selective_prediction_per_run")
    sagg = sel.groupby(["split", "model", "calibration", "threshold"], sort=False).agg(
        coverage=("coverage", mean_sd), accuracy_retained=("accuracy_retained", mean_sd), accuracy_rejected=("accuracy_rejected", mean_sd)).reset_index()
    save_table(sagg, "selective_prediction")
    log("stage 08 done")


if __name__ == "__main__":
    main()
