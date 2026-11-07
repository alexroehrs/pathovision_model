"""Stage 07 - performance tables from the per-image predictions.

Runs on CPU from the CSV files written by stages 04-06.  Produces, under
``results/tables/``:

* ``overall_performance``   - test accuracy, balanced accuracy, macro and
  weighted F1 per model: mean +- SD over seeds, per-seed values, Wilson CI of
  the pooled-seed accuracy, and validation accuracy per seed;
* ``per_class_<split>_<model>`` - recall (Wilson CI), precision, F1, support,
  averaged over seeds, for the primary model on test and external;
* ``confusion_<split>_<model>`` - confusion matrices summed over seeds;
* ``external_results``      - internal vs external accuracy / macro F1 per model
  with the drop, plus the class-matched sensitivity analyses;
* ``external_by_source``    - accuracy per external source dataset;
* ``balancing_effect``      - paired (same seed) comparison of each balanced
  configuration with its unbalanced counterpart;
* ``disease_only``          - metrics after removing the Normal class;
* ``variants``              - brightness-normalised and centre-cropped
  re-evaluations against the unperturbed test/external sets;
* ``efficiency``            - parameters, GFLOPs, latency, training time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, CLASS_LABELS, DATA_DIR, EXTERNAL_DIR, MODEL_BACKBONE, MODEL_BALANCING,  # noqa: E402
                    MODEL_CONFIGS, PRIMARY, RESULTS, SEEDS, available_predictions, global_metrics, log, mean_sd,
                    per_class_table, run_dir, save_table, wilson_ci)

ALL_MODELS = list(MODEL_CONFIGS) + ["M6", "H1"]
PAIRS = [("M1", "M1b"), ("M1", "M5"), ("M2b", "M2"), ("M3b", "M3"), ("M4b", "M4")]


def preds(split):
    """{model: {seed: df}} for a split."""
    out = {}
    for mid, seed, df in available_predictions(split, ALL_MODELS):
        out.setdefault(mid, {})[seed] = df
    return out


def summarise(split_preds, name, val_preds=None):
    rows = []
    for mid in ALL_MODELS:
        if mid not in split_preds:
            continue
        per_seed = {s: global_metrics(d["class"], d["pred"]) for s, d in split_preds[mid].items()}
        pooled = pd.concat(split_preds[mid].values())
        k = int((pooled["class"] == pooled["pred"]).sum()); n = len(pooled)
        lo, hi = wilson_ci(k, n)
        row = {"model": mid, "backbone": MODEL_BACKBONE.get(mid, {"M6": "Ensemble M2+M3+M4", "H1": f"RF on M4 embeddings"}.get(mid)),
               "balancing": MODEL_BALANCING.get(mid, "-"), "n_seeds": len(per_seed), "n_images": per_seed[next(iter(per_seed))]["n"],
               "accuracy_mean_sd": mean_sd([m["accuracy"] for m in per_seed.values()]),
               "accuracy_mean": np.mean([m["accuracy"] for m in per_seed.values()]),
               "accuracy_sd": np.std([m["accuracy"] for m in per_seed.values()], ddof=1) if len(per_seed) > 1 else np.nan,
               "accuracy_pooled": 100 * k / n, "accuracy_pooled_ci_lo": lo, "accuracy_pooled_ci_hi": hi,
               "balanced_accuracy_mean_sd": mean_sd([m["balanced_accuracy"] for m in per_seed.values()]),
               "f1_macro_mean_sd": mean_sd([m["f1_macro"] for m in per_seed.values()]),
               "f1_macro_mean": np.mean([m["f1_macro"] for m in per_seed.values()]),
               "f1_weighted_mean_sd": mean_sd([m["f1_weighted"] for m in per_seed.values()]),
               "f1_weighted_mean": np.mean([m["f1_weighted"] for m in per_seed.values()])}
        for s in SEEDS:
            row[f"accuracy_seed{s}"] = per_seed[s]["accuracy"] if s in per_seed else np.nan
            row[f"f1_weighted_seed{s}"] = per_seed[s]["f1_weighted"] if s in per_seed else np.nan
            if val_preds and mid in val_preds and s in val_preds[mid]:
                v = val_preds[mid][s]
                row[f"val_accuracy_seed{s}"] = 100 * (v["class"] == v["pred"]).mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    save_table(df, name)
    return df


def per_class_and_confusion(split_preds, split, mid):
    if mid not in split_preds:
        return
    tabs = [per_class_table(d["class"], d["pred"]) for d in split_preds[mid].values()]
    pooled = pd.concat(split_preds[mid].values())
    pc = per_class_table(pooled["class"], pooled["pred"])        # Wilson CI on pooled seeds
    pc["support"] = tabs[0]["support"]
    for col in ("recall", "precision", "f1"):
        pc[f"{col}_mean"] = np.mean([t[col] for t in tabs], axis=0)
        pc[f"{col}_sd"] = np.std([t[col] for t in tabs], axis=0, ddof=1) if len(tabs) > 1 else np.nan
        pc[f"{col}_mean_sd"] = [mean_sd([t[col][i] for t in tabs]) for i in range(len(pc))]
    pc = pc.rename(columns={"recall": "recall_pooled", "recall_ci_lo": "recall_pooled_ci_lo",
                            "recall_ci_hi": "recall_pooled_ci_hi", "precision": "precision_pooled", "f1": "f1_pooled"})
    save_table(pc, f"per_class_{split}_{mid}")
    cm = sum(confusion_matrix(d["class"], d["pred"], labels=CLASSES) for d in split_preds[mid].values())
    cm_df = pd.DataFrame(cm, index=[CLASS_LABELS[c] for c in CLASSES], columns=[CLASS_LABELS[c] for c in CLASSES])
    cm_df.index.name = "true\\pred"
    save_table(cm_df.reset_index(), f"confusion_{split}_{mid}", float_format="%d")
    (cm_df / cm_df.sum(axis=1).to_numpy()[:, None] * 100).round(2).reset_index().to_csv(
        RESULTS / "tables" / f"confusion_{split}_{mid}_rownorm.csv", index=False)


def external_analysis(test_p, ext_p):
    if not ext_p:
        log("no external predictions; skipping external analysis")
        return
    rows = []
    manifest = pd.read_csv(EXTERNAL_DIR / "external_manifest.csv") if (EXTERNAL_DIR / "external_manifest.csv").exists() else None
    ext_support = ext_p[PRIMARY][next(iter(ext_p[PRIMARY]))]["class"].value_counts()
    matched_classes = matched(ext_support)
    for mid in ALL_MODELS:
        if mid not in ext_p or mid not in test_p:
            continue
        seeds = sorted(set(ext_p[mid]) & set(test_p[mid]))
        acc_t = [global_metrics(test_p[mid][s]["class"], test_p[mid][s]["pred"]) for s in seeds]
        acc_e = [global_metrics(ext_p[mid][s]["class"], ext_p[mid][s]["pred"]) for s in seeds]
        # class-matched: same classes (external support >= 30) on both sides
        def restrict(df):
            return df[df["class"].isin(matched_classes)]
        acc_tm = [global_metrics(restrict(test_p[mid][s])["class"], restrict(test_p[mid][s])["pred"], matched_classes) for s in seeds]
        acc_em = [global_metrics(restrict(ext_p[mid][s])["class"], restrict(ext_p[mid][s])["pred"], matched_classes) for s in seeds]
        rows.append({"model": mid, "n_seeds": len(seeds),
                     "internal_accuracy": mean_sd([m["accuracy"] for m in acc_t]),
                     "external_accuracy": mean_sd([m["accuracy"] for m in acc_e]),
                     "drop_accuracy_pp": mean_sd([t["accuracy"] - e["accuracy"] for t, e in zip(acc_t, acc_e)]),
                     "internal_f1_macro": mean_sd([m["f1_macro"] for m in acc_t]),
                     "external_f1_macro": mean_sd([m["f1_macro"] for m in acc_e]),
                     "internal_balanced_acc": mean_sd([m["balanced_accuracy"] for m in acc_t]),
                     "external_balanced_acc": mean_sd([m["balanced_accuracy"] for m in acc_e]),
                     "drop_balanced_acc_pp": mean_sd([t["balanced_accuracy"] - e["balanced_accuracy"] for t, e in zip(acc_t, acc_e)]),
                     "matched_classes": ";".join(matched_classes),
                     "internal_accuracy_matched": mean_sd([m["accuracy"] for m in acc_tm]),
                     "external_accuracy_matched": mean_sd([m["accuracy"] for m in acc_em]),
                     "drop_matched_pp": mean_sd([t["accuracy"] - e["accuracy"] for t, e in zip(acc_tm, acc_em)]),
                     "internal_balanced_acc_matched": mean_sd([m["balanced_accuracy"] for m in acc_tm]),
                     "external_balanced_acc_matched": mean_sd([m["balanced_accuracy"] for m in acc_em])})
    save_table(pd.DataFrame(rows), "external_results")

    # per-source accuracy for the primary model
    if manifest is not None and PRIMARY in ext_p:
        # the manifest can list a path twice when two source labels map to the same file name
        src = manifest.drop_duplicates("path").set_index("path")["source"]
        rows = []
        for s, d in ext_p[PRIMARY].items():
            d = d.assign(source=d["relative_path"].map(src))
            for (source, cls), g in d.groupby(["source", "class"]):
                rows.append({"seed": s, "source": source, "class": CLASS_LABELS[cls], "n": len(g),
                             "accuracy": 100 * (g["class"] == g["pred"]).mean()})
        df = pd.DataFrame(rows).groupby(["source", "class"]).agg(n=("n", "first"), accuracy_mean=("accuracy", "mean"),
                                                                   accuracy_sd=("accuracy", "std")).reset_index()
        save_table(df, f"external_by_source_{PRIMARY}")


def matched(ext_support, min_support: int = 30) -> list[str]:
    """Classes with adequate external support (fallback: every class present)."""
    m = [c for c in CLASSES if ext_support.get(c, 0) >= min_support]
    return m or [c for c in CLASSES if ext_support.get(c, 0) > 0]


def balancing_effect(test_p):
    minority = minority_classes()
    rows = []
    for a, b in PAIRS:
        if a not in test_p or b not in test_p:
            continue
        seeds = sorted(set(test_p[a]) & set(test_p[b]))
        rec = {}
        for mid in (a, b):
            rec[mid] = {"acc": [], "f1m": [], "min_rec": [], "bal": []}
            for s in seeds:
                d = test_p[mid][s]; m = global_metrics(d["class"], d["pred"])
                pc = per_class_table(d["class"], d["pred"]).set_index("class")
                rec[mid]["acc"].append(m["accuracy"]); rec[mid]["f1m"].append(m["f1_macro"]); rec[mid]["bal"].append(m["balanced_accuracy"])
                rec[mid]["min_rec"].append(np.mean([pc.loc[CLASS_LABELS[c], "recall"] for c in minority]))
        rows.append({"unbalanced": a, "balanced": b, "n_seeds": len(seeds), "minority_classes": ";".join(minority),
                     "acc_unbalanced": mean_sd(rec[a]["acc"]), "acc_balanced": mean_sd(rec[b]["acc"]),
                     "delta_acc_pp": mean_sd(np.subtract(rec[b]["acc"], rec[a]["acc"])),
                     "f1_macro_unbalanced": mean_sd(rec[a]["f1m"]), "f1_macro_balanced": mean_sd(rec[b]["f1m"]),
                     "delta_f1_macro_pp": mean_sd(np.subtract(rec[b]["f1m"], rec[a]["f1m"])),
                     "balanced_acc_unbalanced": mean_sd(rec[a]["bal"]), "balanced_acc_balanced": mean_sd(rec[b]["bal"]),
                     "minority_recall_unbalanced": mean_sd(rec[a]["min_rec"]), "minority_recall_balanced": mean_sd(rec[b]["min_rec"]),
                     "delta_minority_recall_pp": mean_sd(np.subtract(rec[b]["min_rec"], rec[a]["min_rec"]))})
    save_table(pd.DataFrame(rows), "balancing_effect")


def minority_classes(k: int = 2) -> list[str]:
    """The k classes with the fewest training images (from provenance.csv or the folder tree)."""
    prov = DATA_DIR / "provenance.csv"
    if prov.exists():
        c = pd.read_csv(prov).query("split == 'train'")["class"].value_counts()
    else:
        c = pd.Series({cls: len(list((DATA_DIR / "train" / cls).glob("*"))) for cls in CLASSES if (DATA_DIR / "train" / cls).exists()})
    return list(c.sort_values().index[:k]) if len(c) else ["Candidiasis", "Tinea"]


def disease_only(test_p):
    rows = []
    dis = [c for c in CLASSES if c != "Normal"]
    for mid in ALL_MODELS:
        if mid not in test_p:
            continue
        ms = []
        for s, d in test_p[mid].items():
            dd = d[d["class"] != "Normal"]
            ms.append(global_metrics(dd["class"], dd["pred"], dis))
        rows.append({"model": mid, "n_images": ms[0]["n"], "accuracy": mean_sd([m["accuracy"] for m in ms]),
                     "f1_macro": mean_sd([m["f1_macro"] for m in ms]), "f1_weighted": mean_sd([m["f1_weighted"] for m in ms])})
    save_table(pd.DataFrame(rows), "disease_only")


def variants(base_p: dict, var_p: dict, name: str):
    rows = []
    for mid in ALL_MODELS:
        if mid not in base_p or mid not in var_p:
            continue
        seeds = sorted(set(base_p[mid]) & set(var_p[mid]))
        acc_b = [100 * (base_p[mid][s]["class"] == base_p[mid][s]["pred"]).mean() for s in seeds]
        acc_v = [100 * (var_p[mid][s]["class"] == var_p[mid][s]["pred"]).mean() for s in seeds]
        # Normal-class recall is the quantity most exposed to the brightness cue
        rn_b = [100 * ((d := base_p[mid][s])[d["class"] == "Normal"]["pred"] == "Normal").mean() for s in seeds]
        rn_v = [100 * ((d := var_p[mid][s])[d["class"] == "Normal"]["pred"] == "Normal").mean() for s in seeds]
        rows.append({"model": mid, "variant": name, "n_seeds": len(seeds), "accuracy_base": mean_sd(acc_b),
                     "accuracy_variant": mean_sd(acc_v), "delta_pp": mean_sd(np.subtract(acc_v, acc_b)),
                     "normal_recall_base": mean_sd(rn_b), "normal_recall_variant": mean_sd(rn_v)})
    return rows


def efficiency():
    rows = []
    for mid in MODEL_CONFIGS:
        for s in SEEDS:
            rd = run_dir(mid, s)
            e, d = rd / "efficiency.json", rd / "done.json"
            if not e.exists():
                continue
            ej = json.load(open(e)); dj = json.load(open(d)) if d.exists() else {}
            rows.append({"model": mid, "seed": s, "backbone": MODEL_BACKBONE[mid], "params_M": ej["params_M"],
                         "gflops": ej.get("gflops"), "latency_bs1_ms": ej["latency_bs1_ms"], "input": "x".join(map(str, ej["input_size"])),
                         "train_time_min": dj.get("train_time_s", np.nan) / 60, "best_epoch": dj.get("best_epoch"),
                         "device": ej.get("device")})
    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby("model", sort=False).agg(backbone=("backbone", "first"), input=("input", "first"), params_M=("params_M", "first"),
                                                  gflops=("gflops", "first"), latency_bs1_ms=("latency_bs1_ms", "mean"),
                                                  train_time_min=("train_time_min", "mean"), best_epoch_mean=("best_epoch", "mean"),
                                                  device=("device", "first")).reset_index()
        save_table(df, "efficiency_per_run"); save_table(agg, "efficiency")


def main() -> None:
    test_p, val_p, ext_p = preds("test"), preds("val"), preds("external")
    if not test_p:
        sys.exit("no test predictions found under reanalysis/runs/")
    summarise(test_p, "overall_performance", val_p)
    summarise(val_p, "validation_performance")
    if ext_p:
        summarise(ext_p, "external_performance")
    for mid in {PRIMARY, "M1", "M6", "H1"} & set(test_p):
        per_class_and_confusion(test_p, "test", mid)
        per_class_and_confusion(ext_p, "external", mid)
    external_analysis(test_p, ext_p)
    balancing_effect(test_p)
    disease_only(test_p)
    rows = variants(test_p, preds("test_brightnorm"), "test_brightnorm") + variants(test_p, preds("test_centercrop"), "test_centercrop")
    rows += variants(ext_p, preds("external_centercrop"), "external_centercrop")
    if rows:
        save_table(pd.DataFrame(rows), "variants")
    efficiency()
    log("stage 07 done")


if __name__ == "__main__":
    main()
