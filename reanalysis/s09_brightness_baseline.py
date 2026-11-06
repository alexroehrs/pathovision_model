"""Stage 09 - how much of the task is solved by global colour statistics?

Computes, for every image of the internal split and of the external set, a
handful of global image statistics (mean and SD of CIELAB L*, a*, b*, mean
HSV saturation, image size and aspect ratio) and trains two shallow
classifiers on the training split:

* ``normal_vs_disease`` - logistic regression, Normal class against the rest;
* ``seven_class``       - multinomial logistic regression and a random forest.

Their test/external performance is the "brightness/colour baseline" against
which the deep models are compared.  Per-class mean L* is reported as well
(the Normal-class brightness artefact described in the paper).

Outputs ``results/tables/image_stats_by_class.csv``,
``brightness_baseline.csv`` and ``results/image_stats/<split>.csv``
(per-image statistics, reused by stage 11).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, CLASS_LABELS, DATA_DIR, EXTERNAL_DIR, RESULTS, ensure_dir, global_metrics, log, save_table  # noqa: E402
from infer import VALID_EXTENSIONS  # noqa: E402

FEATURES = ["mean_L", "sd_L", "mean_a", "sd_a", "mean_b", "sd_b", "mean_sat", "height", "width", "aspect"]


def image_stats(path: Path) -> dict | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    small = cv2.resize(img, (256, int(256 * h / w)) if w >= h else (int(256 * w / h), 256))
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100 / 255); a = lab[:, :, 1] - 128; b = lab[:, :, 2] - 128
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return {"mean_L": L.mean(), "sd_L": L.std(), "mean_a": a.mean(), "sd_a": a.std(), "mean_b": b.mean(), "sd_b": b.std(),
            "mean_sat": hsv[:, :, 1].mean() / 255, "height": h, "width": w, "aspect": w / h}


def collect(root: Path, split_name: str, out_dir: Path) -> pd.DataFrame:
    cache = out_dir / f"{split_name}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for cls in CLASSES:
        d = root / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in VALID_EXTENSIONS:
                st = image_stats(f)
                if st:
                    st.update(relative_path=str(f.relative_to(root.parent if split_name in ("train", "val", "test") else root)),
                              split=split_name, **{"class": cls})
                    rows.append(st)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    log(f"{split_name}: statistics for {len(df)} images")
    return df


def main() -> None:
    out_dir = ensure_dir(RESULTS / "image_stats")
    tr, va, te = (collect(DATA_DIR / s, s, out_dir) for s in ("train", "val", "test"))
    ext = collect(EXTERNAL_DIR, "external", out_dir) if (EXTERNAL_DIR / "Acne").exists() else None

    allint = pd.concat([tr, va, te])
    stats = allint.groupby("class")[["mean_L", "sd_L", "mean_sat"]].agg(["mean", "std"])
    stats.columns = ["_".join(c) for c in stats.columns]
    stats = stats.reindex(CLASSES).reset_index()
    stats["class"] = stats["class"].map(CLASS_LABELS)
    stats.insert(1, "n", allint["class"].value_counts().reindex(CLASSES).to_numpy())
    if ext is not None:
        e = ext.groupby("class")["mean_L"].agg(["mean", "std"]).reindex(CLASSES)
        stats["external_mean_L_mean"], stats["external_mean_L_std"] = e["mean"].to_numpy(), e["std"].to_numpy()
    save_table(stats, "image_stats_by_class")

    rows = []
    Xtr = tr[FEATURES].to_numpy(); ytr = tr["class"].to_numpy()
    evals = [("test", te)] + ([("external", ext)] if ext is not None else [])
    for task in ("normal_vs_disease", "seven_class"):
        if task == "normal_vs_disease":
            f = lambda y: np.where(y == "Normal", "Normal", "Disease")
            models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))}
            labels = ["Disease", "Normal"]
        else:
            f = lambda y: y
            models = {"logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
                      "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)}
            labels = CLASSES
        for name, clf in models.items():
            clf.fit(Xtr, f(ytr))
            for split, d in evals:
                yp = clf.predict(d[FEATURES].to_numpy()); yt = f(d["class"].to_numpy())
                m = global_metrics(yt, yp, labels)
                extra = {}
                if task == "normal_vs_disease":
                    extra = {"normal_recall": 100 * np.mean(yp[yt == "Normal"] == "Normal") if (yt == "Normal").any() else np.nan,
                             "normal_precision": 100 * np.mean(yt[yp == "Normal"] == "Normal") if (yp == "Normal").any() else np.nan}
                rows.append({"task": task, "classifier": name, "split": split, "n": m["n"], "features": "+".join(FEATURES),
                             "accuracy": m["accuracy"], "acc_ci_lo": m["acc_ci_lo"], "acc_ci_hi": m["acc_ci_hi"],
                             "balanced_accuracy": m["balanced_accuracy"], "f1_macro": m["f1_macro"], **extra})
                log(f"{task}/{name}/{split}: acc={m['accuracy']:.2f} bal={m['balanced_accuracy']:.2f}")
    save_table(pd.DataFrame(rows), "brightness_baseline")
    log("stage 09 done")


if __name__ == "__main__":
    main()
