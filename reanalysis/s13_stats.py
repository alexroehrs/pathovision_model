"""Stage 13 - statistical tests (CPU, from CSV outputs of stages 04-07).

* ``stats_pairwise``  - for the key model pairs (M4 vs M1, M4 vs M4b, M6 vs M4,
  H1 vs M4, each balanced vs unbalanced pair): per-seed accuracy difference
  (mean +- SD, paired t-test across seeds) and McNemar's exact test on the
  paired per-image predictions of each seed;
* ``stats_cv``        - cross-validation summary: mean +- SD, t-based 95 % CI,
  paired t-test across folds and the Nadeau-Bengio corrected resampled t-test
  for M4 vs M1;
* ``stats_external``  - internal vs external accuracy of the primary model with
  a bootstrap 95 % CI of the difference (seeds pooled) and the class-matched
  version; chi-square test of class distribution internal test vs external;
* ``stats_seed_variability`` - range and SD of test accuracy across seeds per
  model (how much a single-seed comparison can be trusted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASELINE, CLASSES, MODEL_CONFIGS, PRIMARY, RESULTS, SEEDS, available_predictions,  # noqa: E402
                    bootstrap_ci, global_metrics, log, mean_sd, save_table)

PAIRS = [(PRIMARY, BASELINE), (PRIMARY, "M4b"), ("M6", PRIMARY), ("H1", PRIMARY), ("M1b", "M1"), ("M5", "M1"), ("M2", "M2b"), ("M3", "M3b")]


def mcnemar_exact(a_correct, b_correct):
    b = int(np.sum(a_correct & ~b_correct)); c = int(np.sum(~a_correct & b_correct))
    n = b + c
    p = min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)) if n > 0 else 1.0
    return b, c, p


def pairwise(test_p):
    rows = []
    for a, b in PAIRS:
        if a not in test_p or b not in test_p:
            continue
        seeds = sorted(set(test_p[a]) & set(test_p[b]))
        diffs, mc = [], []
        for s in seeds:
            da = test_p[a][s].sort_values("relative_path"); db = test_p[b][s].sort_values("relative_path")
            ca = (da["class"] == da["pred"]).to_numpy(); cb = (db["class"] == db["pred"]).to_numpy()
            diffs.append(100 * (ca.mean() - cb.mean()))
            mc.append(mcnemar_exact(ca, cb))
        t_p = stats.ttest_rel([100 * (test_p[a][s]["class"] == test_p[a][s]["pred"]).mean() for s in seeds],
                              [100 * (test_p[b][s]["class"] == test_p[b][s]["pred"]).mean() for s in seeds]).pvalue if len(seeds) > 1 else np.nan
        rows.append({"model_a": a, "model_b": b, "n_seeds": len(seeds), "acc_a": mean_sd([100 * (test_p[a][s]["class"] == test_p[a][s]["pred"]).mean() for s in seeds]),
                     "acc_b": mean_sd([100 * (test_p[b][s]["class"] == test_p[b][s]["pred"]).mean() for s in seeds]),
                     "diff_pp": mean_sd(diffs), "paired_t_p_across_seeds": t_p,
                     "mcnemar_discordant_a_only": ";".join(str(m[0]) for m in mc), "mcnemar_discordant_b_only": ";".join(str(m[1]) for m in mc),
                     "mcnemar_p_per_seed": ";".join(f"{m[2]:.4f}" for m in mc), "mcnemar_p_median": float(np.median([m[2] for m in mc]))})
    save_table(pd.DataFrame(rows), "stats_pairwise", float_format="%.4f")


def cv_stats():
    p = RESULTS / "tables" / "cv_folds.csv"
    if not p.exists():
        log("cv_folds.csv not found; skipping CV statistics")
        return
    cv = pd.read_csv(p)
    rows = []
    for mid, g in cv.groupby("model", sort=False):
        acc = g.sort_values("fold")["accuracy"].to_numpy(); k = len(acc)
        half = stats.t.ppf(0.975, k - 1) * acc.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
        rows.append({"model": mid, "k": k, "accuracy": mean_sd(acc), "ci95_lo": acc.mean() - half, "ci95_hi": acc.mean() + half,
                     "f1_weighted": mean_sd(g["f1_weighted"]), "f1_macro": mean_sd(g["f1_macro"])})
    models = cv["model"].unique()
    if PRIMARY in models and BASELINE in models:
        a = cv[cv.model == PRIMARY].sort_values("fold")["accuracy"].to_numpy(); b = cv[cv.model == BASELINE].sort_values("fold")["accuracy"].to_numpy()
        d = a - b; k = len(d)
        t_naive, p_naive = stats.ttest_rel(a, b)
        # Nadeau & Bengio (2003) corrected resampled t-test for k-fold CV: variance inflated by (1/k + test_frac/train_frac)
        test_frac, train_frac = 1 / k, (k - 1) / k
        var = d.var(ddof=1) * (1 / k + test_frac / train_frac)
        t_nb = d.mean() / np.sqrt(var); p_nb = 2 * stats.t.sf(abs(t_nb), k - 1)
        rows.append({"model": f"{PRIMARY} - {BASELINE}", "k": k, "accuracy": mean_sd(d), "ci95_lo": np.nan, "ci95_hi": np.nan,
                     "f1_weighted": f"paired t = {t_naive:.2f}, p = {p_naive:.4f}", "f1_macro": f"Nadeau-Bengio t = {t_nb:.2f}, p = {p_nb:.4f}"})
    save_table(pd.DataFrame(rows), "stats_cv", float_format="%.4f")


def external_stats(test_p, ext_p):
    if PRIMARY not in ext_p:
        return
    rows = []
    seeds = sorted(set(test_p[PRIMARY]) & set(ext_p[PRIMARY]))
    ti = pd.concat([test_p[PRIMARY][s] for s in seeds]); te = pd.concat([ext_p[PRIMARY][s] for s in seeds])
    ci_ = (ti["class"] == ti["pred"]).to_numpy().astype(float); ce_ = (te["class"] == te["pred"]).to_numpy().astype(float)
    rng = np.random.default_rng(0)
    boots = [ci_[rng.integers(0, len(ci_), len(ci_))].mean() - ce_[rng.integers(0, len(ce_), len(ce_))].mean() for _ in range(2000)]
    rows.append({"analysis": "all classes", "model": PRIMARY, "n_internal": len(ti) // len(seeds), "n_external": len(te) // len(seeds),
                 "internal_acc": 100 * ci_.mean(), "external_acc": 100 * ce_.mean(), "drop_pp": 100 * (ci_.mean() - ce_.mean()),
                 "drop_ci_lo": 100 * np.percentile(boots, 2.5), "drop_ci_hi": 100 * np.percentile(boots, 97.5)})
    support = te["class"].value_counts() / len(seeds)
    matched = [c for c in CLASSES if support.get(c, 0) >= 30] or [c for c in CLASSES if support.get(c, 0) > 0]
    tm, em = ti[ti["class"].isin(matched)], te[te["class"].isin(matched)]
    gm_i = global_metrics(tm["class"], tm["pred"], matched); gm_e = global_metrics(em["class"], em["pred"], matched)
    rows.append({"analysis": f"class-matched ({';'.join(matched)}), balanced accuracy", "model": PRIMARY,
                 "n_internal": len(tm) // len(seeds), "n_external": len(em) // len(seeds), "internal_acc": gm_i["balanced_accuracy"],
                 "external_acc": gm_e["balanced_accuracy"], "drop_pp": gm_i["balanced_accuracy"] - gm_e["balanced_accuracy"],
                 "drop_ci_lo": np.nan, "drop_ci_hi": np.nan})
    # chi-square of class distribution (internal test vs external)
    ct = pd.DataFrame({"internal": ti["class"].value_counts().reindex(CLASSES).fillna(0) / len(seeds),
                       "external": te["class"].value_counts().reindex(CLASSES).fillna(0) / len(seeds)}).round().astype(int)
    chi2, p, dof, _ = stats.chi2_contingency(ct.T.to_numpy())
    rows.append({"analysis": f"chi-square class distribution internal test vs external: chi2={chi2:.1f}, dof={dof}, p={p:.2e}", "model": "-",
                 "n_internal": int(ct['internal'].sum()), "n_external": int(ct['external'].sum()), "internal_acc": np.nan, "external_acc": np.nan,
                 "drop_pp": np.nan, "drop_ci_lo": np.nan, "drop_ci_hi": np.nan})
    save_table(pd.DataFrame(rows), "stats_external", float_format="%.3f")


def seed_variability(test_p):
    rows = []
    for mid in list(MODEL_CONFIGS) + ["M6", "H1"]:
        if mid not in test_p:
            continue
        acc = np.array([100 * (d["class"] == d["pred"]).mean() for d in test_p[mid].values()])
        rows.append({"model": mid, "n_seeds": len(acc), "accuracy": mean_sd(acc), "min": acc.min(), "max": acc.max(), "range_pp": acc.max() - acc.min()})
    save_table(pd.DataFrame(rows), "stats_seed_variability")


def main() -> None:
    test_p, ext_p = {}, {}
    for mid, s, d in available_predictions("test"):
        test_p.setdefault(mid, {})[s] = d
    for mid, s, d in available_predictions("external"):
        ext_p.setdefault(mid, {})[s] = d
    if not test_p:
        sys.exit("no predictions found")
    pairwise(test_p); cv_stats(); external_stats(test_p, ext_p); seed_variability(test_p)
    log("stage 13 done")


if __name__ == "__main__":
    main()
