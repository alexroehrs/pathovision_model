"""Stage 01 - build the internal dataset with full provenance.

Reproduces the data pipeline of the original repository (download -> organise
-> perceptual-hash deduplication -> stratified 80/10/10 split with seed 42)
and adds what the review asked for:

* each Kaggle source is downloaded into its own folder so that every image can
  be traced back to its source dataset and source folder (``provenance.csv``);
* a near-duplicate audit (dHash Hamming distance <= ``--near-threshold``) within
  and across splits, reported but not deleted;
* source x class and split x class count tables.

Requires the Kaggle CLI to be configured by the user (``~/.kaggle/kaggle.json``
or the ``KAGGLE_USERNAME``/``KAGGLE_KEY`` environment variables).  This script
never reads or prints the credentials.

Usage:
    python reanalysis/s01_build_internal.py [--near-threshold 10] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, DATA_DIR, DATASETS, MERGED_DIR, PROJECT, RAW_DIR, RESULTS,  # noqa: E402
                    ensure_dir, log, save_table, sh, write_json)

sys.path.insert(0, str(PROJECT / "datasets"))
from organize_dataset import CLASS_MAPPING, VALID_EXTENSIONS  # noqa: E402
from detect_duplicates2 import compute_dhash                 # noqa: E402

KAGGLE_SOURCES = {
    "skin_diseases_youssef": "youssefmohmmed/human-skin-diseases-image",
    "dermnet_goel": "shubhamgoel27/dermnet",
}
DONE = DATA_DIR / ".reanalysis_done.json"


def kaggle_configured() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return cfg.exists()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
            yield p


def download_sources() -> None:
    for key, kid in KAGGLE_SOURCES.items():
        dest = RAW_DIR / key
        if dest.exists() and any(iter_images(dest)):
            log(f"{key}: already downloaded ({dest})")
            continue
        if not kaggle_configured():
            sys.exit("Kaggle CLI is not configured. In Colab, run the credential cell of the "
                     "notebook first (it asks you to paste your own kaggle.json).")
        ensure_dir(dest)
        sh(["kaggle", "datasets", "download", "-d", kid, "-p", str(dest), "--unzip"])


def scan_provenance() -> pd.DataFrame:
    """One row per raw image with its source dataset/folder and mapped class."""
    rows = []
    for key in KAGGLE_SOURCES:
        for p in iter_images(RAW_DIR / key):
            folder = p.parent.name
            cls = CLASS_MAPPING.get(folder)
            if cls is None:
                continue  # folders that the original pipeline ignores
            parts = [x.lower() for x in p.relative_to(RAW_DIR / key).parts]
            src_split = "test" if "test" in parts else ("train" if "train" in parts else "unknown")
            rows.append({"source_dataset": key, "source_folder": folder, "source_split": src_split,
                         "source_file": p.name, "class": cls, "md5": md5(p)})
    df = pd.DataFrame(rows)
    log(f"raw images mapped to target classes: {len(df)}")
    return df


def near_duplicate_audit(root: Path, threshold: int, out_name: str) -> dict:
    """dHash near-duplicate pairs (Hamming <= threshold) with split/class info."""
    items = []
    for p in iter_images(root):
        rel = p.relative_to(root)
        split = rel.parts[0] if len(rel.parts) == 3 else "all"
        cls = rel.parts[-2]
        try:
            h = compute_dhash(p)
        except ValueError:
            continue
        items.append((str(rel), split, cls, np.frombuffer(h.encode(), dtype=np.uint8) - 48))
    if not items:
        return {}
    H = np.stack([x[3] for x in items]).astype(np.uint8)
    pairs = []
    # Block-wise Hamming distance to keep memory bounded (N ~ 8k -> 64M entries).
    for i in range(len(items)):
        d = np.count_nonzero(H[i + 1:] != H[i], axis=1)
        for j in np.nonzero(d <= threshold)[0]:
            j = int(j) + i + 1
            pairs.append({"path_a": items[i][0], "split_a": items[i][1], "class_a": items[i][2],
                          "path_b": items[j][0], "split_b": items[j][1], "class_b": items[j][2],
                          "hamming": int(d[j - i - 1])})
    pairs_df = pd.DataFrame(pairs)
    out = ensure_dir(RESULTS / "dataset_qc")
    pairs_df.to_csv(out / f"{out_name}.csv", index=False)
    summary = {
        "n_images": len(items), "threshold": threshold, "n_pairs": len(pairs_df),
        "n_exact": int((pairs_df["hamming"] == 0).sum()) if len(pairs_df) else 0,
        "n_cross_split": int((pairs_df["split_a"] != pairs_df["split_b"]).sum()) if len(pairs_df) else 0,
        "n_cross_class": int((pairs_df["class_a"] != pairs_df["class_b"]).sum()) if len(pairs_df) else 0,
    }
    log(f"near-duplicate audit ({out_name}): {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-threshold", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="rebuild even if the split exists")
    args = ap.parse_args()

    if DONE.exists() and not args.force:
        log(f"internal dataset already built ({DONE}); use --force to rebuild")
        return

    download_sources()
    prov = scan_provenance()
    ensure_dir(RESULTS / "dataset_qc")
    prov.to_csv(RESULTS / "dataset_qc" / "raw_provenance.csv", index=False)

    # Source folder x mapped class (documents the DermNet folder contents issue)
    folder_counts = prov.groupby(["source_dataset", "source_folder", "class"]).size().reset_index(name="n_images")
    save_table(folder_counts, "source_folder_by_class", subdir="dataset_qc")

    # Organise -> merged (fresh)
    for d in (MERGED_DIR, DATA_DIR):
        if d.exists():
            shutil.rmtree(d)
    sh([sys.executable, "datasets/organize_dataset.py", "--raw-dir", str(RAW_DIR), "--output", str(MERGED_DIR)])

    # Exact perceptual-hash deduplication (as in the original pipeline)
    sh([sys.executable, "datasets/detect_duplicates2.py", "--dataset-dir", str(MERGED_DIR),
        "--method", "dhash", "--delete", "--output-dir", str(RESULTS / "dataset_qc" / "dedup_merged")])

    # Near-duplicate audit of the deduplicated pool (report only)
    audit_merged = near_duplicate_audit(MERGED_DIR, args.near_threshold, "near_duplicates_merged")

    # Stratified split, seed 42
    sh([sys.executable, "reproduction/datasets/create_split.py", "--input", str(MERGED_DIR),
        "--output", str(DATA_DIR), "--seed", "42"])

    # Cross-split leakage audit on the final split
    audit_split = near_duplicate_audit(DATA_DIR, args.near_threshold, "near_duplicates_split")

    # Provenance of every image in the final split (join by md5)
    by_md5 = prov.drop_duplicates("md5").set_index("md5")
    rows = []
    for p in iter_images(DATA_DIR):
        rel = p.relative_to(DATA_DIR)
        h = md5(p)
        src = by_md5.loc[h] if h in by_md5.index else None
        rows.append({"relative_path": str(rel), "split": rel.parts[0], "class": rel.parts[1], "md5": h,
                     "source_dataset": None if src is None else src["source_dataset"],
                     "source_folder": None if src is None else src["source_folder"],
                     "source_split": None if src is None else src["source_split"],
                     "source_file": None if src is None else src["source_file"]})
    final = pd.DataFrame(rows)
    final.to_csv(DATA_DIR / "provenance.csv", index=False)
    final.to_csv(RESULTS / "dataset_qc" / "provenance.csv", index=False)
    unmatched = int(final["source_dataset"].isna().sum())
    log(f"final images: {len(final)}; unmatched provenance: {unmatched}")

    counts = final.pivot_table(index="class", columns="split", values="md5", aggfunc="count", fill_value=0)
    counts = counts.reindex(CLASSES)[["train", "val", "test"]]
    counts["total"] = counts.sum(axis=1)
    save_table(counts.reset_index(), "split_counts", subdir="dataset_qc", float_format="%d")
    src_cls = final.pivot_table(index="class", columns="source_dataset", values="md5", aggfunc="count", fill_value=0)
    save_table(src_cls.reindex(CLASSES).reset_index(), "source_by_class", subdir="dataset_qc", float_format="%d")
    src_cls_split = final.groupby(["split", "class", "source_dataset"]).size().reset_index(name="n_images")
    save_table(src_cls_split, "source_by_class_split", subdir="dataset_qc", float_format="%d")

    write_json({"n_raw_mapped": len(prov), "n_final": len(final), "unmatched_provenance": unmatched,
                "near_dup_merged": audit_merged, "near_dup_split": audit_split,
                "split_counts": counts.to_dict()}, DONE)
    shutil.copy2(DONE, RESULTS / "dataset_qc" / "build_summary.json")
    log("stage 01 done")


if __name__ == "__main__":
    main()
