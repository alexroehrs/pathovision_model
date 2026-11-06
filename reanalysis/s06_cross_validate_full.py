"""Stage 06 - 5-fold cross-validation with the full training recipe.

The original ``scripts/cross_validate.py`` used a simplified recipe.  Here the
pooled train+val images (the test split stays untouched) are divided into five
stratified folds; for each fold, ``scripts/train.py`` is run with the model's
own configuration file (same recipe as stage 03) on the other four folds, and
the held-out fold is predicted.  Fold directories are built from symbolic
links, so no image is copied.

Outputs: ``runs/cv/<model>/fold<k>/`` (checkpoint, config, pred_val.csv) and
``results/tables/cv_folds.csv`` (accuracy, macro/weighted F1 per fold).

Usage:
    python reanalysis/s06_cross_validate_full.py [--models M1 M4] [--folds 5] [--override ...]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, CV_MODELS, DATA_DIR, MODEL_CONFIGS, PROJECT, REANALYSIS, RUNS,  # noqa: E402
                    ensure_dir, global_metrics, log, save_table, sh, write_json)
from infer import VALID_EXTENSIONS, get_device, load_checkpoint, predict_folder, relative_paths  # noqa: E402
from s03_train_all import parse_override, set_nested  # noqa: E402

CV_DIR = REANALYSIS / "cv_folds"


def build_folds(k: int, seed: int) -> None:
    if (CV_DIR / f"fold{k-1}" / "val").exists():
        return
    items = []
    for split in ("train", "val"):
        for cls in CLASSES:
            for f in sorted((DATA_DIR / split / cls).iterdir()):
                if f.suffix.lower() in VALID_EXTENSIONS:
                    items.append((f.resolve(), cls))
    df = pd.DataFrame(items, columns=["path", "class"])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df["path"], df["class"])):
        root = CV_DIR / f"fold{fold}"
        for part, idx in (("train", tr_idx), ("val", va_idx)):
            for cls in CLASSES:
                ensure_dir(root / part / cls)
            for _, r in df.iloc[idx].iterrows():
                dst = root / part / r["class"] / Path(r["path"]).name
                if not dst.exists():
                    dst.symlink_to(r["path"])
        if not (root / "test").exists():
            (root / "test").symlink_to((DATA_DIR / "test").resolve(), target_is_directory=True)
        df.iloc[va_idx].assign(fold=fold).to_csv(root / "heldout.csv", index=False)
    log(f"built {k} folds from {len(df)} train+val images")


def run_fold(model_id: str, fold: int, overrides: list[str]) -> dict | None:
    rd = RUNS / "cv" / model_id / f"fold{fold}"
    done = rd / "done.json"
    if not done.exists():
        if rd.exists():
            shutil.rmtree(rd)
        ensure_dir(rd)
        with open(PROJECT / MODEL_CONFIGS[model_id]) as f:
            cfg = yaml.safe_load(f)
        cfg["experiment"].update(seed=42, output_dir=str(rd), name=f"{model_id}_cv{fold}")
        cfg["data"]["dataset_dir"] = str(CV_DIR / f"fold{fold}")
        for o in overrides:
            kk, v = parse_override(o)
            set_nested(cfg, kk, v)
        with open(rd / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        t0 = time.time()
        sh([sys.executable, "scripts/train.py", "--config", str(rd / "config.yaml")])
        write_json({"model_id": model_id, "fold": fold, "train_time_s": time.time() - t0}, done)
    pred = rd / "pred_val.csv"
    if not pred.exists():
        model, config = load_checkpoint(rd / "checkpoints" / "best.pt", get_device())
        df = predict_folder(model, config, CV_DIR / f"fold{fold}" / "val")
        df.insert(1, "relative_path", df["path"].map(lambda p: str(Path(p).resolve().relative_to(DATA_DIR.resolve()))))
        df.to_csv(pred, index=False)
    df = pd.read_csv(pred)
    m = global_metrics(df["class"], df["pred"])
    m.update(model=model_id, fold=fold)
    log(f"{model_id} fold {fold}: acc={m['accuracy']:.2f} F1w={m['f1_weighted']:.2f}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CV_MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    build_folds(args.folds, seed=42)
    rows = []
    for mid in args.models:
        for k in range(args.folds):
            rows.append(run_fold(mid, k, args.override))
    df = pd.DataFrame(rows)[["model", "fold", "n", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]]
    save_table(df, "cv_folds")
    log("stage 06 done")


if __name__ == "__main__":
    main()
