"""Stage 20 - effect of the inherited, non-deduplicated split (GPU).

Quantifies how much cross-split duplication inflates the internal estimate.
The two Kaggle collections are re-organised with the original
``datasets/organize_dataset.py`` into their *inherited* train/test partitions
**without** deduplication (images of unknown partition go to train, as in the
original pipeline); because neither collection ships a validation partition,
a stratified 10 % of the inherited training images (seed 42) is held out for
checkpoint selection.  Every image is hashed (MD5 and the 256-bit dHash of
the original pipeline) and each inherited test image is flagged when an
identical image exists in the inherited training or validation partition.

M4 and M1 are trained on this split with their unchanged recipes (three
seeds) and evaluated on the inherited test partition and the external set.

Needs the raw Kaggle downloads (``datasets/raw``); if they are missing they
are downloaded again with the user's own Kaggle credentials (stage 01).

Outputs: ``results/dataset_qc/leaky_manifest.csv`` and
``runs/leaky/<model>/seed<s>/pred_{test,external}.csv``.  Tables: stage 23.

    python reanalysis/s20_leaky_split.py [--models M4 M1] [--seeds ...] [--override ...]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, DATASETS, EXTERNAL_DIR, MODEL_CONFIGS, RAW_DIR, RESULTS, RUNS, SEEDS, ensure_dir, log, sh  # noqa: E402
from extra import predict_run, train_run  # noqa: E402
from s01_build_internal import compute_dhash, download_sources, iter_images, md5  # noqa: E402

LEAKY_DIR = DATASETS / "leaky_inherited"
DONE = LEAKY_DIR / ".built"


def build() -> pd.DataFrame:
    man_path = RESULTS / "dataset_qc" / "leaky_manifest.csv"
    if DONE.exists() and man_path.exists():
        return pd.read_csv(man_path)
    download_sources()
    if LEAKY_DIR.exists():
        shutil.rmtree(LEAKY_DIR)
    sh([sys.executable, "datasets/organize_dataset.py", "--raw-dir", str(RAW_DIR), "--output", str(LEAKY_DIR)])
    # validation: stratified 10 % of the inherited training partition
    tr = [(p, p.parent.name) for c in CLASSES for p in iter_images(LEAKY_DIR / "train" / c)]
    df = pd.DataFrame(tr, columns=["p", "class"])
    _, va = train_test_split(df, test_size=0.10, stratify=df["class"], random_state=42)
    for p, c in zip(va["p"], va["class"]):
        dst = ensure_dir(LEAKY_DIR / "val" / c) / p.name
        shutil.move(str(p), dst)
    rows = []
    for p in iter_images(LEAKY_DIR):
        rel = p.relative_to(LEAKY_DIR)
        rows.append({"relative_path": str(rel), "split": rel.parts[0], "class": rel.parts[1], "md5": md5(p), "dhash": compute_dhash(p)})
    man = pd.DataFrame(rows)
    fit = man[man.split.isin(["train", "val"])]
    md5_fit, dh_fit = set(fit.md5), set(fit.dhash)
    md5_tr, dh_tr = set(man.loc[man.split == "train", "md5"]), set(man.loc[man.split == "train", "dhash"])
    man["dup_in_fit_md5"] = man.md5.isin(md5_fit) & (man.split == "test")
    man["dup_in_fit"] = (man.md5.isin(md5_fit) | man.dhash.isin(dh_fit)) & (man.split == "test")
    man["val_dup_in_train"] = (man.md5.isin(md5_tr) | man.dhash.isin(dh_tr)) & (man.split == "val")
    ensure_dir(man_path.parent)
    man.to_csv(man_path, index=False)
    DONE.write_text("ok")
    t = man[man.split == "test"]
    log(f"inherited split: {man.split.value_counts().to_dict()}; test images with an identical image in train/val: "
        f"{int(t.dup_in_fit.sum())}/{len(t)} (MD5 only {int(t.dup_in_fit_md5.sum())})")
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M4", "M1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    build()
    if args.build_only:
        return
    ext = (EXTERNAL_DIR / "Acne").exists()
    for seed in args.seeds:
        for mid in args.models:
            rd = RUNS / "leaky" / mid / f"seed{seed}"
            train_run(rd, MODEL_CONFIGS[mid], seed, LEAKY_DIR, args.override, name=f"leaky_{mid}_seed{seed}",
                      extra_meta={"model_id": mid, "split": "inherited, not deduplicated"})
            jobs = [("test", LEAKY_DIR / "test", LEAKY_DIR)]
            if ext:
                jobs.append(("external", EXTERNAL_DIR, EXTERNAL_DIR))
            predict_run(rd, jobs, batch_size=args.batch_size, num_workers=args.num_workers)
    log("stage 20 done")


if __name__ == "__main__":
    main()
