"""Stage 10 - source-dataset confounding analysis (CPU).

The internal corpus merges two Kaggle collections whose class coverage
differs, so "source" and "class" are partly confounded.  This stage
quantifies the issue:

* ``source_by_class_split``   - image counts per source x class x split;
* ``source_shortcut``         - accuracy a classifier would reach on the test
  split by predicting the majority class of each source (how much of the
  task the source identity alone explains);
* ``accuracy_by_source``      - test accuracy of every model per source
  dataset, and within each class per source (paired by seed);
* ``source_probe``            - linear probe (logistic regression, 5-fold CV)
  predicting the source dataset from the primary model's penultimate
  embeddings, within each class that has both sources, against the chance
  level given by the class prior; a probe far above chance means the network
  encodes acquisition/source characteristics.

Inputs: ``datasets/data/provenance.csv`` (stage 01), ``runs/M4/seed*/emb_*.npz``
and ``pred_test.csv`` (stage 04).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, CLASS_LABELS, DATA_DIR, MODEL_CONFIGS, PRIMARY, SEEDS, available_predictions, log, mean_sd, run_dir, save_table  # noqa: E402

MODELS = list(MODEL_CONFIGS) + ["M6", "H1"]


def main() -> None:
    prov_path = DATA_DIR / "provenance.csv"
    if not prov_path.exists():
        sys.exit("provenance.csv not found; run stage 01")
    prov = pd.read_csv(prov_path)
    src = prov.set_index("relative_path")["source_dataset"]

    counts = prov.groupby(["split", "class", "source_dataset"]).size().reset_index(name="n")
    save_table(counts, "source_by_class_split", float_format="%d")

    # Source-only shortcut on the test split: predict the majority class of the source (from train)
    maj = prov.query("split == 'train'").groupby("source_dataset")["class"].agg(lambda s: s.value_counts().index[0])
    test = prov.query("split == 'test'").copy()
    test["shortcut_pred"] = test["source_dataset"].map(maj)
    rows = [{"rule": "majority class of source (train)", "n_test": len(test), "accuracy": 100 * (test["shortcut_pred"] == test["class"]).mean()}]
    # prior-based shortcut: sample class from the source's train class distribution -> expected accuracy
    dist = prov.query("split == 'train'").groupby(["source_dataset", "class"]).size().unstack(fill_value=0)
    dist = dist.div(dist.sum(1), axis=0)
    exp_acc = np.mean([dist.loc[r.source_dataset].get(r["class"], 0) for _, r in test.iterrows()])
    rows.append({"rule": "class drawn from source's train class prior (expected)", "n_test": len(test), "accuracy": 100 * exp_acc})
    save_table(pd.DataFrame(rows), "source_shortcut")

    # Accuracy by source per model
    rows = []
    for mid, seed, df in available_predictions("test", MODELS):
        df = df.assign(source=df["relative_path"].map(src))
        for source, g in df.groupby("source"):
            rows.append({"model": mid, "seed": seed, "source": source, "class": "all", "n": len(g), "accuracy": 100 * (g["class"] == g["pred"]).mean()})
        for (source, cls), g in df.groupby(["source", "class"]):
            rows.append({"model": mid, "seed": seed, "source": source, "class": CLASS_LABELS[cls], "n": len(g), "accuracy": 100 * (g["class"] == g["pred"]).mean()})
    if rows:
        d = pd.DataFrame(rows)
        agg = d.groupby(["model", "source", "class"], sort=False).agg(n=("n", "first"), accuracy=("accuracy", mean_sd)).reset_index()
        save_table(agg, "accuracy_by_source")

    # Linear probe for source identity on embeddings of the primary model
    rows = []
    for seed in SEEDS:
        npz = run_dir(PRIMARY, seed) / "emb_train.npz"
        if not npz.exists():
            continue
        e = np.load(npz, allow_pickle=True)
        emb = pd.DataFrame({"relative_path": e["path"], "class": e["label"]})
        emb["source"] = emb["relative_path"].map(src)
        X = e["X"]
        for cls in ["all"] + CLASSES:
            m = np.ones(len(emb), bool) if cls == "all" else (emb["class"] == cls).to_numpy()
            y = emb.loc[m, "source"].to_numpy()
            vc = pd.Series(y).value_counts()
            if len(vc) < 2 or vc.min() < 20:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1))
            yp = cross_val_predict(clf, X[m], y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
            bal = np.mean([np.mean(yp[y == s] == s) for s in vc.index])
            rows.append({"seed": seed, "class": cls if cls == "all" else CLASS_LABELS[cls], "n": int(m.sum()),
                         "sources": ";".join(f"{k}={v}" for k, v in vc.items()), "probe_accuracy": 100 * np.mean(yp == y),
                         "probe_balanced_accuracy": 100 * bal, "chance_balanced": 100 / len(vc), "majority_rate": 100 * vc.max() / vc.sum()})
            log(f"source probe seed {seed} {cls}: balanced acc {100*bal:.1f}% (n={m.sum()})")
    if rows:
        d = pd.DataFrame(rows)
        agg = d.groupby("class", sort=False).agg(n=("n", "first"), sources=("sources", "first"), probe_accuracy=("probe_accuracy", mean_sd),
                                                 probe_balanced_accuracy=("probe_balanced_accuracy", mean_sd),
                                                 chance_balanced=("chance_balanced", "first"), majority_rate=("majority_rate", "first")).reset_index()
        save_table(agg, f"source_probe_{PRIMARY}")
    log("stage 10 done")


if __name__ == "__main__":
    main()
