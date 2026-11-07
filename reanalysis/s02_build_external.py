"""Stage 02 - build the unified external validation set.

Thin wrapper around ``datasets/build_external_unified.py`` (unchanged
pipeline: four Kaggle sources + SD-198 from HuggingFace, label mapping,
quality filter, removal of near-duplicates of internal images at dHash
Hamming <= 10).  Overlap removal is run against BOTH the deduplicated merged
pool and the final split, as recommended in that script's help text.

Adds the source x class count table that the review asked for, and a
near-duplicate audit between the external set and the internal test split.

Usage:
    python reanalysis/s02_build_external.py [--force]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, DATA_DIR, EXTERNAL_DIR, EXTERNAL_RAW_DIR, MERGED_DIR, RESULTS,  # noqa: E402
                    ensure_dir, log, save_table, sh)

DONE = EXTERNAL_DIR / ".reanalysis_done"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if DONE.exists() and not args.force:
        log(f"external set already built ({EXTERNAL_DIR}); use --force to rebuild")
        return
    if not (DATA_DIR / "train").exists():
        sys.exit("internal split not found; run s01_build_internal.py first")

    if EXTERNAL_DIR.exists():
        # keep the committed manifests of the original build for comparison
        keep = ensure_dir(RESULTS / "dataset_qc" / "external_original_manifests")
        for f in EXTERNAL_DIR.glob("*.csv"):
            shutil.copy2(f, keep / f.name)
        shutil.rmtree(EXTERNAL_DIR)

    internal = [str(MERGED_DIR), str(DATA_DIR)] if MERGED_DIR.exists() else [str(DATA_DIR)]
    sh([sys.executable, "datasets/build_external_unified.py", "--internal-dir", *internal,
        "--raw-dir", str(EXTERNAL_RAW_DIR), "--out-dir", str(EXTERNAL_DIR), "--threshold", "10"])

    manifest_all = pd.read_csv(EXTERNAL_DIR / "external_manifest.csv")
    manifest = manifest_all.drop_duplicates("path")   # a path can be listed twice (same file name under two source labels)
    n_dup = len(manifest_all) - len(manifest)
    log(f"external manifest: {len(manifest_all)} rows, {len(manifest)} unique files ({n_dup} duplicated path entries)")
    counts = manifest.pivot_table(index="class", columns="source", values="path", aggfunc="count", fill_value=0)
    counts = counts.reindex(CLASSES).fillna(0).astype(int)
    counts["total"] = counts.sum(axis=1)
    save_table(counts.reset_index(), "external_source_by_class", subdir="dataset_qc", float_format="%d")
    for f in ("external_label_mapping.csv", "external_discarded_labels.csv"):
        shutil.copy2(EXTERNAL_DIR / f, RESULTS / "dataset_qc" / f)

    # Near-duplicate audit external vs internal test split (should be empty)
    from s01_build_internal import compute_dhash, iter_images  # noqa: E402
    import numpy as np
    def hashes(root, cls_from_parts):
        out = []
        for p in iter_images(root):
            try:
                h = compute_dhash(p)
            except ValueError:
                continue
            out.append((str(p.relative_to(root)), cls_from_parts(p.relative_to(root)), np.frombuffer(h.encode(), dtype=np.uint8) - 48))
        return out
    ext = hashes(EXTERNAL_DIR, lambda r: r.parts[0])
    tst = hashes(DATA_DIR / "test", lambda r: r.parts[0])
    He = np.stack([x[2] for x in ext]).astype(np.uint8); Ht = np.stack([x[2] for x in tst]).astype(np.uint8)
    pairs = []
    for i in range(len(tst)):
        d = np.count_nonzero(He != Ht[i], axis=1)
        for j in np.nonzero(d <= 10)[0]:
            pairs.append({"test_path": tst[i][0], "test_class": tst[i][1],
                          "external_path": ext[j][0], "external_class": ext[j][1], "hamming": int(d[j])})
    pd.DataFrame(pairs).to_csv(RESULTS / "dataset_qc" / "near_duplicates_external_vs_test.csv", index=False)
    log(f"external images: {len(manifest)}; near-duplicate pairs vs test split: {len(pairs)}")
    DONE.write_text(f"n_images={len(manifest)}\nduplicated_manifest_rows={n_dup}\nnear_dup_pairs_vs_test={len(pairs)}\n")
    log("stage 02 done")


if __name__ == "__main__":
    main()
