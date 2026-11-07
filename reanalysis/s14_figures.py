"""Stage 14 - publication figures (CPU, from CSV outputs).

Figures follow the journal's artwork rules: 84 mm (single column) or 174 mm
(double column) wide, 9-pt sans-serif lettering, no titles inside the figure,
saved as EPS (vector), TIFF (600 dpi, LZW) and PDF (for the LaTeX source).
Colors use a color-vision-deficiency-safe categorical set (validated), a
single-hue sequential map for the confusion matrix, and every series is also
identified by a direct label or legend text.

Fig_class_distribution   - images per class and split (stacked bars);
Fig_confusion_<model>    - row-normalized confusion matrix, seeds pooled;
Fig_roc_pr_<model>       - one-vs-rest ROC and precision-recall curves (seed 42);
Fig_reliability_<model>  - reliability diagrams, test and external, raw vs
                           temperature-scaled;
Fig_ita_<model>          - accuracy per ITA stratum with Wilson CIs, test and
                           external, with the stratum's Normal-class share;
Fig_external_recall_<model> - per-class recall internal vs external.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, CLASS_LABELS, DATA_DIR, PRIMARY, RESULTS, available_predictions, ensure_dir, log, proba_matrix  # noqa: E402

MM = 1 / 25.4
SINGLE, DOUBLE = 84 * MM, 174 * MM
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#B5306F", "#A0661A", "#3FA3D6", "#6A3D9A"]  # validated CVD-safe
CLASS_COLOR = dict(zip(CLASSES, PALETTE))
INK, MUTED, GRID = "#222222", "#555555", "#DDDDDD"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
                     "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})
OUT = ensure_dir(RESULTS / "figures")


def save(fig, name):
    # exact journal widths: no tight bounding box, constrained layout inside the canvas
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.eps")
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    # flatten to RGB (journals do not accept an alpha channel in halftone artwork)
    from PIL import Image
    im = Image.open(OUT / f"{name}.tif")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[3])
        bg.save(OUT / f"{name}.tif", dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)
    log(f"figure {name} written")


def fig_class_distribution():
    prov = DATA_DIR / "provenance.csv"
    if prov.exists():
        c = pd.read_csv(prov).groupby(["class", "split"]).size().unstack(fill_value=0)
    else:
        c = pd.DataFrame({s: {cls: len(list((DATA_DIR / s / cls).glob("*"))) for cls in CLASSES} for s in ("train", "val", "test")})
    c = c.reindex(CLASSES)[["train", "val", "test"]]
    fig, ax = plt.subplots(layout="constrained", figsize=(SINGLE, 2.4))
    x = np.arange(len(CLASSES)); bottom = np.zeros(len(CLASSES))
    shades = ["#0072B2", "#5BA3D0", "#B8D8EC"]
    for (split, col), label in zip(zip(c.columns, shades), ("Training", "Validation", "Test")):
        ax.bar(x, c[split], bottom=bottom, color=col, width=0.7, label=label, edgecolor="white", linewidth=0.8)
        bottom += c[split].to_numpy()
    for i, tot in enumerate(bottom):
        ax.text(i, tot + max(bottom) * 0.01, f"{int(tot)}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([CLASS_LABELS[k] for k in CLASSES], rotation=30, ha="right")
    ax.set_ylabel("Images"); ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.set_ylim(0, max(bottom) * 1.12)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1, columnspacing=1.2)
    save(fig, "Fig_class_distribution")


def fig_confusion(mid, split="test"):
    p = RESULTS / "tables" / f"confusion_{split}_{mid}.csv"
    if not p.exists():
        return
    cm = pd.read_csv(p, index_col=0).to_numpy(dtype=float)
    norm = np.divide(cm * 100, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0)
    fig, ax = plt.subplots(layout="constrained", figsize=(SINGLE, SINGLE * 0.82))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    labels = [CLASS_LABELS[k] for k in CLASSES]
    ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(labels, rotation=40, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for i in range(7):
        for j in range(7):
            if cm[i, j] > 0:
                ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 55 else INK)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("Row-normalized (%)"); cb.outline.set_visible(False)
    save(fig, f"Fig_confusion_{split}_{mid}")


def fig_roc_pr(mid, seed=42, split="test"):
    dfs = {s: d for m, s, d in available_predictions(split, [mid]) if s == seed}
    if seed not in dfs:
        return
    df = dfs[seed]; P = proba_matrix(df); Y = df["class"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, layout="constrained", figsize=(DOUBLE, DOUBLE / 2.0))
    for j, c in enumerate(CLASSES):
        y = (Y == c).astype(int)
        if y.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, P[:, j]); prec, rec, _ = precision_recall_curve(y, P[:, j])
        a1.plot(fpr, tpr, color=CLASS_COLOR[c], lw=1.2, label=f"{CLASS_LABELS[c]} (AUC {auc(fpr, tpr):.2f})")
        a2.plot(rec, prec, color=CLASS_COLOR[c], lw=1.2, label=f"{CLASS_LABELS[c]} (AP {average_precision_score(y, P[:, j]):.2f})")
    a1.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=0.8)
    a1.set_xlabel("False-positive rate"); a1.set_ylabel("True-positive rate"); a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    for ax, lab in ((a1, "a"), (a2, "b")):
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.text(-0.18, 1.02, lab, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.legend(loc="lower right" if ax is a1 else "lower left", handlelength=1.2)
    save(fig, f"Fig_roc_pr_{split}_{mid}")


def fig_reliability(mid):
    files = sorted((RESULTS / "calibration").glob(f"reliability_*_{mid}_seed*.csv"))
    if not files:
        return
    cal = pd.read_csv(RESULTS / "tables" / "calibration.csv")
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(DOUBLE, DOUBLE / 2.0), sharey=True)
    for ax, split, lab in zip(axes, ("test", "external"), ("a", "b")):
        fs = [f for f in files if f.name.startswith(f"reliability_{split}_")]
        if not fs:
            ax.set_visible(False); continue
        rel = pd.concat([pd.read_csv(f) for f in fs]).groupby(["bin_lo", "bin_hi"]).apply(
            lambda g: pd.Series({"n": g["n"].sum(), "conf": np.average(g["mean_confidence"], weights=g["n"]), "acc": np.average(g["accuracy"], weights=g["n"])})).reset_index()
        ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=0.8)
        width = (rel["bin_hi"] - rel["bin_lo"]).to_numpy()
        ax.bar(rel["bin_lo"], rel["acc"], width=width, align="edge", color="#5BA3D0", edgecolor="white", linewidth=0.6, label="Observed accuracy")
        ax.scatter(rel["conf"], rel["acc"], s=np.clip(rel["n"] / rel["n"].max() * 60, 6, 60), color="#D55E00", zorder=3, label="Bin (size = count)")
        row = cal[(cal.split == split) & (cal.model == mid)]
        txt = "\n".join(f"{'raw' if r.calibration == 'raw' else 'temp. scaled'}: ECE {r.ece}" for r in row.itertuples())
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Confidence"); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.text(-0.14, 1.02, lab, transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Accuracy"); axes[0].legend(loc="upper left", handlelength=1)
    save(fig, f"Fig_reliability_{mid}")


def fig_ita(mid, scheme="stratum4_center"):
    p = RESULTS / "tables" / "ita_accuracy_by_stratum.csv"; q = RESULTS / "tables" / "ita_class_composition.csv"
    if not p.exists():
        return
    acc = pd.read_csv(p); comp = pd.read_csv(q)
    order = ["light", "intermediate", "tan", "dark"] if "4" in scheme else ["very light", "light", "intermediate", "tan", "brown", "dark"]
    splits = [s for s in ("test", "external") if ((acc.split == s) & (acc.model == mid) & (acc.scheme == scheme)).any()]
    fig, axes = plt.subplots(1, len(splits), layout="constrained", figsize=(DOUBLE if len(splits) > 1 else SINGLE, 2.8), sharey=True, squeeze=False)
    for ax, split, lab in zip(axes[0], splits, "ab"):
        a = acc[(acc.split == split) & (acc.model == mid) & (acc.scheme == scheme)].set_index("stratum").reindex(order).dropna(subset=["n"])
        c = comp[(comp.split == split) & (comp.scheme == scheme)].set_index("stratum").reindex(a.index)
        x = np.arange(len(a))
        ax.bar(x, a["accuracy_mean"], color="#0072B2", width=0.6, edgecolor="white")
        ax.errorbar(x, a["accuracy_mean"], yerr=[a["accuracy_mean"] - a["ci_lo_mean"], a["ci_hi_mean"] - a["accuracy_mean"]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.set_xticks(x); ax.set_ylim(0, 100)
        ax.set_xticklabels([f"{s.capitalize()}\n(n = {int(n)})" for s, n in zip(a.index, a["n"])], fontsize=7.5)
        for i, share in enumerate(c["normal_share_pct"]):
            ax.text(i, 3, f"Normal\n{share:.0f}%", ha="center", va="bottom", fontsize=6.5, color="white")
        ax.set_xlabel("ITA stratum"); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        if len(splits) > 1:
            ax.text(-0.12, 1.02, lab, transform=ax.transAxes, fontweight="bold", fontsize=10)
    axes[0, 0].set_ylabel("Accuracy (%)")
    save(fig, f"Fig_ita_{mid}")


def fig_external_recall(mid):
    pi, pe = RESULTS / "tables" / f"per_class_test_{mid}.csv", RESULTS / "tables" / f"per_class_external_{mid}.csv"
    if not (pi.exists() and pe.exists()):
        return
    ti, te = pd.read_csv(pi).set_index("class"), pd.read_csv(pe).set_index("class")
    labels = [CLASS_LABELS[c] for c in CLASSES]; x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(layout="constrained", figsize=(SINGLE, 2.6))
    for off, t, col, lab in ((-w / 2, ti, "#0072B2", "Internal test"), (w / 2, te, "#D55E00", "External")):
        r = t.reindex(labels)
        ax.bar(x + off, r["recall_mean"], width=w, color=col, edgecolor="white", label=lab)
        ax.errorbar(x + off, r["recall_mean"], yerr=[r["recall_mean"] - r["recall_pooled_ci_lo"], r["recall_pooled_ci_hi"] - r["recall_mean"]],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{l} ({int(ni)}/{int(ne)})" for l, ni, ne in zip(labels, ti.reindex(labels)["support"], te.reindex(labels)["support"])], fontsize=7, rotation=30, ha="right"); ax.set_ylim(0, 100); ax.set_ylabel("Recall (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1)
    save(fig, f"Fig_external_recall_{mid}")


def main() -> None:
    fig_class_distribution()
    for mid in (PRIMARY, "M6", "H1"):
        fig_confusion(mid); fig_confusion(mid, "external")
    fig_roc_pr(PRIMARY)
    fig_reliability(PRIMARY)
    fig_ita(PRIMARY)
    fig_external_recall(PRIMARY)
    log("stage 14 done")


if __name__ == "__main__":
    main()
