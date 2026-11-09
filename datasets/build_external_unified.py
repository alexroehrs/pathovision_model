"""Build the unified external validation set (4 Kaggle sources + SD-198).

Rebuilds datasets/external_unified/ (about 4,600 images, 7 classes) and writes the
label-level curation metadata:

  - external_label_mapping.csv    (source, source_label, mapped_class, counts)
  - external_discarded_labels.csv (unmapped source labels, with counts)
  - external_manifest.csv         (one row per retained image)

Steps: download sources (kaggle CLI + HuggingFace) -> map source labels to
the 7 classes (one class max per label; unmapped labels discarded and
recorded) -> content-independent QC -> dHash dedup against the internal
corpus and within the external pool (same 256-bit dHash and threshold as
detect_duplicates.py) -> copy to out-dir and write manifests.

Per-class counts of the build used in the article (November 2026; the sources
may change upstream): Acne 1,396 | Candidiasis 55 | Eczema 1,273 |
NailFungus 727 | Normal 298 | Psoriasis 519 | Tinea 330 | total 4,598 unique
files.  Images whose file names coincide are written once, so the manifest can
list a few more rows than there are files; analyses de-duplicate it by path.

Usage:
    python datasets/build_external_unified.py \
        --internal-dir datasets/merged datasets/final \
        --raw-dir datasets/external_raw \
        --out-dir datasets/external_unified

Run with --dry-run first to review the SD-198 label selection (the script
prints kept and discarded labels; adjust the CONFIG section if needed).
"versicolor" labels (tinea versicolor, Malassezia) are INCLUDED by default
(INCLUDE_VERSICOLOR = True), matching the external set used in the paper;
set INCLUDE_VERSICOLOR = False to exclude them.

Requires: kaggle CLI configured; `pip install datasets pillow` for SD-198.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
PATHOVISION_CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]

# --------------------------------------------------------------------------
# CONFIG — sources and label mappings (Online Resource 1, Table S4 of the article)
# --------------------------------------------------------------------------

KAGGLE_SOURCES = {
    # source key            kaggle dataset id
    "acne_dataset":        "nayanchaure/acne-dataset",
    "nail_disease":        "josephrasanjana/nail-disease-image-classification-dataset",
    "skin_normal":         "lysaapriani/skin-disease-and-normal-skin-dataset",
    "skin_10classes":      "ismailpromus/skin-diseases-image-dataset",
}

SD198_HF_ID = "resyhgerwshshgdfghsdfgh/SD-198"  # 6,584 images, 198 labels

# Kaggle sources: REGEX matching on the normalized folder name (lowercase,
# spaces/underscores/hyphens collapsed). Folder names on Kaggle carry
# prefixes/counts (e.g. "1. Eczema 1677"), so exact matching is not enough.
# First matching rule wins; an unmatched folder is DISCARDED (and recorded).
# Mapping follows Online Resource 1, Table S4: each source label -> at most one class.
KAGGLE_LABEL_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "acne_dataset": [
        (r"acne", "Acne"),                 # single-condition repository
    ],
    "nail_disease": [
        (r"onychomycosis|fung", "NailFungus"),  # "healthy"/"psoriasis" folders discarded
    ],
    "skin_normal": [
        (r"^normal", "Normal"),            # dermatitis folders discarded
    ],
    "skin_10classes": [
        (r"eczema", "Eczema"),             # "1. Eczema 1677"
        (r"atopic dermatitis", "Eczema"),  # "3. Atopic Dermatitis - 1.25k"
        (r"psoriasis", "Psoriasis"),       # "7. Psoriasis pictures Lichen Planus and related diseases - 2k"
        # "9. Tinea Ringworm Candidiasis and other Fungal Infections" is a
        # mixed folder and is deliberately NOT mapped (discarded).
    ],
}

# SD-198: REGEX selection over the 198 fine-grained labels (normalized to
# lowercase with spaces). First matching rule wins; a label matching no rule
# is DISCARDED and recorded. Review the printed selection before building.
# The original external set (Tinea = 330 = five tinea_* labels + tinea
# versicolor) included versicolor, so it is kept by default for fidelity to
# the evaluated set; set False to apply a strict dermatophyte-only criterion.
INCLUDE_VERSICOLOR = True

# Labels matching these substrings are never mapped (distinct clinical
# entities that would otherwise match a broad pattern). Reviewable via
# --dry-run, which prints them under DROP.
SD198_EXCLUDE_SUBSTRINGS = [
    "keloidalis",  # acne keloidalis nuchae != acne vulgaris
    "rosacea",     # (acne) rosacea is a distinct entity, not acne vulgaris
]

SD198_PATTERNS: list[tuple[str, str]] = [
    (r"\bacne\b",                          "Acne"),
    (r"candidiasis|candida",               "Candidiasis"),
    (r"eczema|atopic dermatitis",          "Eczema"),
    (r"onychomycosis",                     "NailFungus"),
    (r"psoriasis",                         "Psoriasis"),
    (r"tinea|dermatophyt|ringworm",        "Tinea"),
]


def normalize_label(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", name.strip().lower())


def map_sd198_label(label: str) -> str | None:
    norm = normalize_label(label)
    if not INCLUDE_VERSICOLOR and "versicolor" in norm:
        return None
    if any(s in norm for s in SD198_EXCLUDE_SUBSTRINGS):
        return None
    for pattern, cls in SD198_PATTERNS:
        if re.search(pattern, norm):
            return cls
    return None


# --------------------------------------------------------------------------
# dHash — identical convention to datasets/detect_duplicates.py
# --------------------------------------------------------------------------

def compute_dhash(image_path: str | Path, hash_size: int = 16) -> str | None:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten())


def hamming(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def passes_qc(image_path: Path, min_side: int = 64, min_std: float = 5.0) -> bool:
    """Content-independent QC: decodable, minimum size, not near-uniform."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        return False
    if float(np.std(img)) < min_std:
        return False
    return True


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------

def download_kaggle_sources(raw_dir: Path) -> None:
    for key, kaggle_id in KAGGLE_SOURCES.items():
        dest = raw_dir / key
        if dest.exists() and any(dest.iterdir()):
            logger.info("Kaggle source '%s' already present, skipping", key)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", kaggle_id, dest)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(dest), "--unzip"],
            check=True,
        )


def export_sd198(raw_dir: Path) -> Path:
    """Export SD-198 from HuggingFace to raw_dir/sd198/<label>/*.jpg."""
    dest = raw_dir / "sd198"
    if dest.exists() and any(dest.iterdir()):
        logger.info("SD-198 already exported, skipping")
        return dest
    from datasets import load_dataset  # lazy import

    # Note: a logged "HEAD .../dataset_infos.json 404" is harmless (optional
    # metadata file this repo does not have); the download proceeds anyway.
    logger.info("Loading SD-198 from HuggingFace (%s)...", SD198_HF_ID)
    try:
        ds = load_dataset(SD198_HF_ID)
    except Exception as first_err:  # fall back to the auto-converted parquet branch
        logger.warning("Default load failed (%s); retrying parquet branch...", first_err)
        try:
            ds = load_dataset(SD198_HF_ID, revision="refs/convert/parquet")
        except Exception:
            raise SystemExit(
                "Could not load SD-198 from HuggingFace. Try `pip install -U "
                "datasets huggingface_hub` and rerun; or download/export the "
                "dataset manually to a folder-per-label tree and pass it via "
                "--sd198-dir."
            ) from first_err
    split = ds[list(ds.keys())[0]]
    label_feature = split.features["label"]
    dest.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(split):
        label_name = label_feature.int2str(sample["label"])
        label_dir = dest / normalize_label(label_name).replace(" ", "_")
        label_dir.mkdir(exist_ok=True)
        sample["image"].convert("RGB").save(label_dir / f"sd198_{i:05d}.jpg", quality=95)
    logger.info("SD-198 exported: %d images", len(split))
    return dest


# --------------------------------------------------------------------------
# Candidate collection
# --------------------------------------------------------------------------

def iter_label_dirs(source_dir: Path):
    """Yield (label_name, [image paths]) for every leaf folder with images."""
    for d in sorted(p for p in source_dir.rglob("*") if p.is_dir()):
        images = [f for f in d.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        if images:
            yield d.name, images


def collect_candidates(raw_dir: Path, sd198_dir: Path):
    """Returns candidates[(source, source_label, mapped_class)] = [paths],
    and discarded[(source, source_label)] = n_images."""
    candidates: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    discarded: Counter = Counter()

    missing = [str(raw_dir / key) for key in KAGGLE_SOURCES if not (raw_dir / key).exists()]
    if missing:
        raise SystemExit(
            "Kaggle source folder(s) not found:\n  " + "\n  ".join(missing) +
            "\nExpected one sub-folder per source under --raw-dir "
            f"({raw_dir.resolve()}): {', '.join(KAGGLE_SOURCES)}. "
            "Run without --skip-download to fetch them, or point --raw-dir to the "
            "folder where they were downloaded."
        )

    for key in KAGGLE_SOURCES:
        src_dir = raw_dir / key
        patterns = KAGGLE_LABEL_PATTERNS[key]
        for label, images in iter_label_dirs(src_dir):
            norm = normalize_label(label)
            cls = next((c for pat, c in patterns if re.search(pat, norm)), None)
            if cls is None:
                discarded[(key, label)] += len(images)
            else:
                candidates[(key, label, cls)].extend(images)

    for label, images in iter_label_dirs(sd198_dir):
        cls = map_sd198_label(label)
        if cls is None:
            discarded[("sd198", label)] += len(images)
        else:
            candidates[("sd198", label, cls)].extend(images)

    return candidates, discarded


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified external validation set")
    parser.add_argument("--internal-dir", type=Path, nargs="+", required=True,
                        help="One or more internal corpus roots to remove overlap against. "
                             "Pass BOTH the raw merged corpus and the final split "
                             "(e.g. datasets/merged datasets/final): the original build "
                             "removed overlap against the raw sources, so comparing only "
                             "against the deduplicated final corpus under-removes.")
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/external_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/external_unified"))
    parser.add_argument("--sd198-dir", type=Path, default=None,
                        help="Pre-exported SD-198 folder (skips HuggingFace export)")
    parser.add_argument("--threshold", type=int, default=10,
                        help="dHash Hamming threshold (same default as detect_duplicates.py)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report label selection and counts without copying files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.skip_download:
        download_kaggle_sources(args.raw_dir)
    sd198_dir = args.sd198_dir or export_sd198(args.raw_dir)

    candidates, discarded = collect_candidates(args.raw_dir, sd198_dir)

    print("\n=== SD-198 label selection (REVIEW) ===")
    for (source, label, cls), imgs in sorted(candidates.items()):
        if source == "sd198":
            print(f"  KEEP  {label:55s} -> {cls:12s} ({len(imgs)} imgs)")
    for (source, label), n in sorted(discarded.items()):
        if source == "sd198":
            print(f"  DROP  {label:55s} (unmapped, {n} imgs)")

    if args.dry_run:
        total = sum(len(v) for v in candidates.values())
        print(f"\nDry run: {total} candidate images before QC/dedup; "
              f"{sum(discarded.values())} images under {len(discarded)} discarded labels.")
        return

    # ---- QC ----
    logger.info("Running quality control...")
    qc_fail: Counter = Counter()
    for key in list(candidates):
        kept = [p for p in candidates[key] if passes_qc(p)]
        qc_fail[key] = len(candidates[key]) - len(kept)
        candidates[key] = kept

    # ---- Hash internal corpus ----
    logger.info("Hashing internal corpus for overlap removal: %s", [str(d) for d in args.internal_dir])
    internal_hashes: list[str] = []
    for root in args.internal_dir:
        for img in root.rglob("*"):
            if img.suffix.lower() in VALID_EXTENSIONS:
                h = compute_dhash(img)
                if h:
                    internal_hashes.append(h)
    internal_hashes = list(set(internal_hashes))
    logger.info("Internal images hashed (unique): %d", len(internal_hashes))

    # Bucket internal hashes by 16-bit prefix for a cheap pre-filter
    def near_internal(h: str) -> bool:
        return any(hamming(h, ih) <= args.threshold for ih in internal_hashes)

    # ---- Dedup: vs internal corpus, then within external pool ----
    logger.info("Deduplicating external candidates (threshold=%d)...", args.threshold)
    overlap_removed: Counter = Counter()
    seen_external: list[str] = []
    retained: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for key, imgs in candidates.items():
        for p in imgs:
            h = compute_dhash(p)
            if h is None:
                qc_fail[key] += 1
                continue
            if near_internal(h):
                overlap_removed[key] += 1
                continue
            if any(hamming(h, eh) <= args.threshold for eh in seen_external):
                overlap_removed[key] += 1
                continue
            seen_external.append(h)
            retained[key].append(p)

    # ---- Copy + manifests ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for (source, label, cls), imgs in sorted(retained.items()):
        class_dir = args.out_dir / cls
        class_dir.mkdir(exist_ok=True)
        for p in imgs:
            dest = class_dir / f"{source}_{p.stem}{p.suffix.lower()}"
            shutil.copy2(p, dest)
            manifest_rows.append([str(dest.relative_to(args.out_dir)), source, label, cls])

    with open(args.out_dir / "external_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "source_label", "class"])
        w.writerows(manifest_rows)

    with open(args.out_dir / "external_label_mapping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "source_label", "mapped_class",
                    "n_raw", "n_qc_fail", "n_overlap_removed", "n_retained"])
        for key in sorted(set(candidates) | set(retained)):
            source, label, cls = key
            n_ret = len(retained.get(key, []))
            n_raw = n_ret + qc_fail[key] + overlap_removed[key]
            w.writerow([source, label, cls, n_raw, qc_fail[key], overlap_removed[key], n_ret])

    with open(args.out_dir / "external_discarded_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "source_label", "n_images", "reason"])
        for (source, label), n in sorted(discarded.items()):
            w.writerow([source, label, n, "no unambiguous correspondence to target classes"])

    # ---- Summary vs the evaluated set ----
    expected = {"Acne": 1399, "Candidiasis": 55, "Eczema": 1419, "NailFungus": 776,
                "Normal": 298, "Psoriasis": 527, "Tinea": 330}
    per_class: Counter = Counter()
    for (_, _, cls), imgs in retained.items():
        per_class[cls] += len(imgs)
    per_source: Counter = Counter()
    for (source, _, _), imgs in retained.items():
        per_source[source] += len(imgs)
    print("\n=== Retained images per source ===")
    for source in list(KAGGLE_SOURCES) + ["sd198"]:
        print(f"  {source:16s} {per_source[source]:6,d}")

    print("\n=== Final composition (vs the evaluated set) ===")
    for cls in PATHOVISION_CLASSES:
        got, exp = per_class[cls], expected[cls]
        flag = "" if got == exp else f"   <-- expected {exp:,}"
        print(f"  {cls:12s} {got:6,d}{flag}")
    total, exp_total = sum(per_class.values()), sum(expected.values())
    print(f"  {'TOTAL':12s} {total:6,d}" + ("" if total == exp_total else f"   <-- expected {exp_total:,}"))
    if total != exp_total:
        print("\nNOTE: deviations usually mean the Kaggle/HF sources changed since the "
              "original build, or the SD-198 label selection differs — review "
              "external_label_mapping.csv and adjust the CONFIG section.")


if __name__ == "__main__":
    main()
