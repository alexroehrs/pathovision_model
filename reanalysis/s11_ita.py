"""Stage 11 - image-derived skin-tone (ITA) stratification, corrected.

The original ``src/evaluation/fairness.py`` computed the Individual Typology
Angle from OpenCV's 8-bit Lab values without rescaling (L in 0-255 and b
offset by 128), which shifts every ITA value.  Here ITA is computed on the
proper CIELAB scale (L* in 0-100, b* centred on 0) with two pixel-selection
rules:

* ``center``  - median L*/b* of the central 60 % of the image (as before);
* ``skin``    - median over pixels that pass a loose skin-colour gate in HSV
  and whose a* (redness) is below the image's 60th percentile, to favour
  lesion-free skin.

Strata follow Del Bino et al.: very light (>55), light (41-55),
intermediate (28-41), tan (19-28), brown (10-19), dark (<10); the paper's
four groups (light >41, intermediate 28-41, tan 19-28, dark <19) are also
reported.  For each model/seed and split the stage writes accuracy per stratum
with Wilson CIs, the class composition of each stratum, an omnibus chi-square
test of accuracy across strata, and a class-adjusted logistic regression
(correct ~ stratum + class) whose stratum coefficients isolate the skin-tone
effect from the class mix.

Outputs ``results/ita/<split>_ita.csv`` (per image) and tables
``ita_class_composition``, ``ita_accuracy_by_stratum``, ``ita_tests``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, CLASS_LABELS, DATA_DIR, EXTERNAL_DIR, MODEL_CONFIGS, PRIMARY, RESULTS,  # noqa: E402
                    available_predictions, ensure_dir, log, mean_sd, save_table, wilson_ci)
from infer import VALID_EXTENSIONS  # noqa: E402

MODELS = list(MODEL_CONFIGS) + ["M6", "H1"]
SIX = [("very light", 55, np.inf), ("light", 41, 55), ("intermediate", 28, 41), ("tan", 19, 28), ("brown", 10, 19), ("dark", -np.inf, 10)]
FOUR = [("light", 41, np.inf), ("intermediate", 28, 41), ("tan", 19, 28), ("dark", -np.inf, 19)]


def strata(ita: pd.Series, scheme) -> pd.Series:
    out = pd.Series(index=ita.index, dtype=object)
    for name, lo, hi in scheme:
        out[(ita >= lo) & (ita < hi)] = name
    return out


def ita_of(path: Path) -> dict | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100 / 255); a = lab[:, :, 1] - 128; b = lab[:, :, 2] - 128
    hh, ww = L.shape
    c = (slice(int(hh * .2), int(hh * .8)), slice(int(ww * .2), int(ww * .8)))
    ita_center = np.degrees(np.arctan2(np.median(L[c]) - 50, np.median(b[c])))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gate = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] >= 20) & (hsv[:, :, 1] <= 200) & (hsv[:, :, 2] >= 40)
    gate &= a <= np.percentile(a, 60)
    if gate.sum() >= 0.02 * gate.size:
        ita_skin = np.degrees(np.arctan2(np.median(L[gate]) - 50, np.median(b[gate])))
        frac = gate.mean()
    else:
        ita_skin, frac = np.nan, gate.mean()
    return {"ita_center": ita_center, "ita_skin": ita_skin, "skin_fraction": frac, "mean_L": L.mean()}


def collect(root: Path, split: str) -> pd.DataFrame:
    out = ensure_dir(RESULTS / "ita") / f"{split}_ita.csv"
    if out.exists():
        return pd.read_csv(out)
    rows = []
    for cls in CLASSES:
        d = root / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in VALID_EXTENSIONS:
                r = ita_of(f)
                if r:
                    base = root.parent if split in ("train", "val", "test") else root
                    r.update(relative_path=str(f.relative_to(base)), split=split, **{"class": cls})
                    rows.append(r)
    df = pd.DataFrame(rows)
    for col, scheme, tag in (("ita_center", FOUR, "stratum4_center"), ("ita_center", SIX, "stratum6_center"),
                             ("ita_skin", FOUR, "stratum4_skin"), ("ita_skin", SIX, "stratum6_skin")):
        df[tag] = strata(df[col], scheme)
    df.to_csv(out, index=False)
    log(f"{split}: ITA for {len(df)} images")
    return df


MIN_N = 20  # minimum images per stratum for the tests (overridable with --min-n)


def analyse(ita_df: pd.DataFrame, split: str, stratum_col: str, comp_rows, acc_rows, test_rows):
    order = [s[0] for s in (FOUR if "4" in stratum_col else SIX)]
    comp = ita_df.groupby([stratum_col, "class"]).size().unstack(fill_value=0).reindex(order).fillna(0).astype(int)
    comp = comp.reindex(columns=CLASSES, fill_value=0)
    for s in order:
        if s in comp.index:
            comp_rows.append({"split": split, "scheme": stratum_col, "stratum": s, "n": int(comp.loc[s].sum()),
                              **{CLASS_LABELS[c]: int(comp.loc[s, c]) for c in CLASSES},
                              "normal_share_pct": 100 * comp.loc[s, "Normal"] / max(comp.loc[s].sum(), 1)})
    key = ita_df.set_index("relative_path")[stratum_col]
    for mid, seed, pred in available_predictions(split, MODELS):
        pred = pred.assign(stratum=pred["relative_path"].map(key), correct=(pred["class"] == pred["pred"]).astype(int)).dropna(subset=["stratum"])
        for s, g in pred.groupby("stratum"):
            lo, hi = wilson_ci(int(g["correct"].sum()), len(g))
            acc_rows.append({"split": split, "scheme": stratum_col, "model": mid, "seed": seed, "stratum": s, "n": len(g),
                             "accuracy": 100 * g["correct"].mean(), "ci_lo": lo, "ci_hi": hi})
        # omnibus chi-square across strata (n >= MIN_N per stratum)
        tab = pred.groupby("stratum")["correct"].agg(["sum", "count"])
        tab = tab[tab["count"] >= MIN_N]
        if len(tab) >= 2:
            ct = np.array([tab["sum"], tab["count"] - tab["sum"]])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            test_rows.append({"split": split, "scheme": stratum_col, "model": mid, "seed": seed, "test": "chi2 accuracy across strata",
                              "statistic": chi2, "dof": dof, "p_value": p, "strata": ";".join(tab.index)})
        # class-adjusted logistic regression: correct ~ C(stratum) + C(class)
        if len(tab) < 2:
            continue
        try:
            import statsmodels.formula.api as smf
            sub = pred[pred["stratum"].isin(tab.index)].copy()
            ref = order[0] if order[0] in tab.index else tab.index[0]
            sub["stratum"] = pd.Categorical(sub["stratum"], categories=[ref] + [s for s in order if s in tab.index and s != ref])
            fit = smf.logit("correct ~ C(stratum) + C(Q('class'))", data=sub).fit(disp=0, maxiter=200)
            for name in fit.params.index:
                if name.startswith("C(stratum)"):
                    test_rows.append({"split": split, "scheme": stratum_col, "model": mid, "seed": seed,
                                      "test": f"logit class-adjusted OR vs {ref}: {name.split('[T.')[1].rstrip(']')}",
                                      "statistic": float(np.exp(fit.params[name])), "dof": np.nan, "p_value": float(fit.pvalues[name]),
                                      "strata": f"OR 95% CI {np.exp(fit.conf_int().loc[name, 0]):.2f}-{np.exp(fit.conf_int().loc[name, 1]):.2f}"})
            lr = fit.llr_pvalue
            full_ll, red = fit.llf, smf.logit("correct ~ C(Q('class'))", data=sub).fit(disp=0, maxiter=200).llf
            lrt = 2 * (full_ll - red)
            test_rows.append({"split": split, "scheme": stratum_col, "model": mid, "seed": seed, "test": "LRT stratum effect adjusted for class",
                              "statistic": lrt, "dof": len(tab) - 1, "p_value": float(stats.chi2.sf(lrt, len(tab) - 1)), "strata": ";".join(tab.index)})
        except ImportError:
            log("statsmodels not installed; class-adjusted model skipped (pip install statsmodels)")
        except Exception as exc:  # singular designs on tiny smoke data
            log(f"class-adjusted logit failed for {mid} seed {seed} ({exc})")


def main() -> None:
    global MIN_N
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--min-n", type=int, default=MIN_N); MIN_N = ap.parse_args().min_n
    comp_rows, acc_rows, test_rows = [], [], []
    for split, root in (("test", DATA_DIR / "test"), ("external", EXTERNAL_DIR), ("train", DATA_DIR / "train"), ("val", DATA_DIR / "val")):
        if not root.exists():
            continue
        df = collect(root, split)
        if split in ("test", "external"):
            for col in ("stratum4_center", "stratum6_center", "stratum4_skin", "stratum6_skin"):
                analyse(df, split, col, comp_rows, acc_rows, test_rows)
    # per-class ITA distribution (documents that ITA is not a skin-tone measure for nail images etc.)
    all_int = pd.concat([pd.read_csv(RESULTS / "ita" / f"{s}_ita.csv") for s in ("train", "val", "test") if (RESULTS / "ita" / f"{s}_ita.csv").exists()])
    dist = all_int.groupby("class")[["ita_center", "ita_skin", "skin_fraction"]].describe(percentiles=[.25, .5, .75])
    dist.columns = ["_".join(c) for c in dist.columns]
    save_table(dist.reindex(CLASSES).reset_index(), "ita_distribution_by_class")
    save_table(pd.DataFrame(comp_rows), "ita_class_composition", float_format="%.1f")
    acc = pd.DataFrame(acc_rows)
    if len(acc):
        save_table(acc, "ita_accuracy_by_stratum_per_run")
        agg = acc.groupby(["split", "scheme", "model", "stratum"], sort=False).agg(n=("n", "first"), accuracy=("accuracy", mean_sd),
                                                                                    accuracy_mean=("accuracy", "mean"),
                                                                                    ci_lo_mean=("ci_lo", "mean"), ci_hi_mean=("ci_hi", "mean")).reset_index()
        save_table(agg, "ita_accuracy_by_stratum")
    if test_rows:
        save_table(pd.DataFrame(test_rows), "ita_tests", float_format="%.4f")
    log("stage 11 done")


if __name__ == "__main__":
    main()
