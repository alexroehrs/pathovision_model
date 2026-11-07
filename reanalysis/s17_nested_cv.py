"""Stage 17 - nested 5-fold cross-validation (GPU).

In stage 06 the held-out fold also served as the validation set for early
stopping and checkpoint selection, which biases the fold estimates upwards.
Here the outer folds are the same as in stage 06 (stratified, seed 42, pooled
train+val), but within each outer training portion a stratified 10 % inner
validation set is held out for checkpoint selection; the outer fold is used
only once, for the final prediction.

Outputs: ``runs/ncv/<model>/fold<k>/`` with ``pred_heldout.csv`` (outer fold)
and ``pred_test.csv`` (the untouched test split).  Tables are written by
stage 23.

    python reanalysis/s17_nested_cv.py [--models M1 M4] [--folds 5] [--override ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, CV_MODELS, DATA_DIR, MODEL_CONFIGS, REANALYSIS, RUNS, log  # noqa: E402
from extra import link_tree, predict_run, train_run  # noqa: E402
from infer import VALID_EXTENSIONS  # noqa: E402

NCV_DIR = REANALYSIS / "ncv_folds"


def pooled() -> pd.DataFrame:
    items = []
    for split in ("train", "val"):
        for cls in CLASSES:
            for f in sorted((DATA_DIR / split / cls).iterdir()):
                if f.suffix.lower() in VALID_EXTENSIONS:
                    items.append((str(f.resolve()), cls))
    return pd.DataFrame(items, columns=["src", "class"])


def build(k: int) -> list[Path]:
    df = pooled()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)          # identical to stage 06
    roots = []
    for fold, (tr_idx, ho_idx) in enumerate(skf.split(df["src"], df["class"])):
        tr = df.iloc[tr_idx]
        inner_tr, inner_va = train_test_split(tr, test_size=0.10, stratify=tr["class"], random_state=100 + fold)
        items = pd.concat([inner_tr.assign(split="train"), inner_va.assign(split="val"), df.iloc[ho_idx].assign(split="test")])
        roots.append(link_tree(items, NCV_DIR / f"fold{fold}"))
    log(f"nested CV: {k} outer folds from {len(df)} images, 10 % inner validation")
    return roots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CV_MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    roots = build(args.folds)
    for fold, root in enumerate(roots):          # fold-major: complete folds of both models first
        for mid in args.models:
            rd = RUNS / "ncv" / mid / f"fold{fold}"
            train_run(rd, MODEL_CONFIGS[mid], 42, root, args.override, name=f"{mid}_ncv{fold}", extra_meta={"model_id": mid, "fold": fold})
            predict_run(rd, [("heldout", root / "test", DATA_DIR), ("test", DATA_DIR / "test", DATA_DIR)],
                        batch_size=args.batch_size, num_workers=args.num_workers)
    log("stage 17 done")


if __name__ == "__main__":
    main()
