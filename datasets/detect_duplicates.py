"""Detect duplicate images across the dataset using perceptual hashing.

Checks:
- Overlapping images between the two Kaggle sources
- Data leakage (same image in train and test)
- Quality control

Usage:
    python datasets/detect_duplicates.py --dataset-dir datasets/merged
    python datasets/detect_duplicates.py --dataset-dir datasets/merged --delete
    python datasets/detect_duplicates.py --dataset-dir datasets/merged --delete --dry-run
"""

import argparse
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Priority order: images in higher-priority splits are KEPT when deduplicating.
# train > val > test (we prefer to keep training data)
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def compute_dhash(image_path: str | Path, hash_size: int = 16) -> str:
    """Compute difference hash (dHash) for an image.

    dHash is robust to minor variations in size, aspect ratio, and brightness.

    Args:
        image_path: Path to image file.
        hash_size: Hash grid size (default 16 = 256-bit hash).

    Returns:
        Hex string hash.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize to (hash_size+1, hash_size) for horizontal gradient
    resized = cv2.resize(img, (hash_size + 1, hash_size))

    # Compute differences between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert to hash string
    return "".join(str(int(b)) for b in diff.flatten())


def compute_md5(image_path: str | Path) -> str:
    """Compute MD5 hash of file bytes (exact duplicate detection)."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def choose_file_to_delete(info1: dict, info2: dict) -> dict:
    """Given two duplicate image infos, return the one that should be deleted.

    Strategy:
    - Keep the image from the higher-priority split (train > val > test).
    - If same split, keep the one whose class folder sorts first (arbitrary but
      deterministic), i.e. delete the second one.

    Args:
        info1: Dict with keys 'path', 'split', 'class'.
        info2: Dict with keys 'path', 'split', 'class'.

    Returns:
        The info dict of the file to delete.
    """
    p1 = SPLIT_PRIORITY.get(info1["split"], 99)
    p2 = SPLIT_PRIORITY.get(info2["split"], 99)

    if p1 < p2:
        # info1 is more important → delete info2
        return info2
    elif p2 < p1:
        return info1
    else:
        # Same split: keep the one that comes first alphabetically
        if str(info1["path"]) <= str(info2["path"]):
            return info2
        return info1


def find_duplicates(
    dataset_dir: str | Path,
    method: str = "dhash",
    threshold: int = 10,
) -> dict[str, list[tuple[str, str]]]:
    """Find duplicate image pairs across the dataset.

    Args:
        dataset_dir: Root directory of the merged dataset.
        method: 'md5' for exact duplicates, 'dhash' for perceptual.
        threshold: Reserved for near-duplicate matching; the current implementation
            groups images by identical hash only (Hamming distance 0), as used in the paper.

    Returns:
        Dictionary with duplicate analysis results.
    """
    dataset_dir = Path(dataset_dir)

    # Collect all images with their split info
    images = []
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in VALID_EXTENSIONS:
                    images.append({
                        "path": img_path,
                        "split": split,
                        "class": cls_dir.name,
                    })

    print(f"Computing hashes for {len(images)} images...")

    # Compute hashes
    hashes = {}
    for i, img_info in enumerate(images):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(images)} images...")
        try:
            if method == "md5":
                h = compute_md5(img_info["path"])
            else:
                h = compute_dhash(img_info["path"])
            hashes[str(img_info["path"])] = {
                "hash": h,
                **img_info,
            }
        except Exception as e:
            logger.warning("Failed to hash %s: %s", img_info["path"], e)

    # Group by hash → find exact duplicates
    hash_to_paths: dict[str, list[dict]] = defaultdict(list)
    for path, info in hashes.items():
        hash_to_paths[info["hash"]].append(info)

    exact_duplicates = []
    for h, infos in hash_to_paths.items():
        if len(infos) > 1:
            for i in range(len(infos)):
                for j in range(i + 1, len(infos)):
                    exact_duplicates.append((
                        str(infos[i]["path"]),
                        str(infos[j]["path"]),
                        infos[i]["split"],
                        infos[j]["split"],
                        infos[i],
                        infos[j],
                    ))

    # Cross-split leakage
    cross_split_leaks = [
        entry for entry in exact_duplicates if entry[2] != entry[3]
    ]

    results = {
        "total_images": len(images),
        "exact_duplicates": exact_duplicates,
        "cross_split_leaks": cross_split_leaks,
        "num_exact_duplicates": len(exact_duplicates),
        "num_cross_split_leaks": len(cross_split_leaks),
    }

    return results


def delete_duplicates(
    exact_duplicates: list,
    dry_run: bool = False,
) -> dict:
    """Delete the lower-priority file from each duplicate pair.

    Args:
        exact_duplicates: List of tuples (path1, path2, split1, split2, info1, info2)
                          as returned by find_duplicates().
        dry_run: If True, only print what would be deleted without actually deleting.

    Returns:
        Summary dict with counts of deleted / skipped files.
    """
    to_delete: set[str] = set()

    for p1, p2, s1, s2, info1, info2 in exact_duplicates:
        victim = choose_file_to_delete(info1, info2)
        to_delete.add(str(victim["path"]))

    deleted = 0
    skipped = 0
    errors = 0

    action_label = "[DRY-RUN] Would delete" if dry_run else "Deleting"

    for path_str in sorted(to_delete):
        path = Path(path_str)
        if not path.exists():
            logger.warning("File already gone, skipping: %s", path_str)
            skipped += 1
            continue
        print(f"  {action_label}: {path_str}")
        if not dry_run:
            try:
                path.unlink()
                deleted += 1
            except OSError as e:
                logger.error("Failed to delete %s: %s", path_str, e)
                errors += 1
        else:
            deleted += 1  # count as "would delete" in dry-run

    return {
        "files_targeted": len(to_delete),
        "deleted": deleted,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run,
    }


def analyze_normal_class(dataset_dir: str | Path) -> dict:
    """Investigate the Normal class for distribution artifacts.

    Compares image statistics (resolution, brightness, color distribution)
    between Normal and disease classes.

    Args:
        dataset_dir: Root of dataset split (e.g., datasets/merged/test).

    Returns:
        Dictionary with analysis results.
    """
    dataset_dir = Path(dataset_dir)
    classes = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]
    stats = {}

    for cls in classes:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue

        images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])[:100]
        resolutions = []
        mean_brightness = []
        mean_colors = []

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            resolutions.append((h, w))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness.append(float(np.mean(gray)))
            mean_colors.append(img.mean(axis=(0, 1)).tolist())

        if resolutions:
            stats[cls] = {
                "count": len(images),
                "avg_resolution": f"{np.mean([r[0] for r in resolutions]):.0f}x{np.mean([r[1] for r in resolutions]):.0f}",
                "avg_brightness": float(np.mean(mean_brightness)),
                "std_brightness": float(np.std(mean_brightness)),
                "avg_color_bgr": [float(np.mean([c[i] for c in mean_colors])) for i in range(3)],
            }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect (and optionally delete) duplicates")
    parser.add_argument("--dataset-dir", type=str, default="datasets/merged")
    parser.add_argument("--method", choices=["md5", "dhash"], default="dhash")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Reserved; identical-hash matching is always used.")
    parser.add_argument("--analyze-normal", action="store_true", help="Analyze Normal class artifacts")
    parser.add_argument("--output-dir", type=str, default="results/01_dataset_qc")
    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Delete the lower-priority duplicate from each pair. "
            "Priority: train > val > test. Use --dry-run to preview first."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Combined with --delete: print what would be deleted without actually deleting.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset_dir = Path(args.dataset_dir)

    # ── Duplicate detection ────────────────────────────────────────────────────
    print("\n=== Duplicate Detection ===")
    results = find_duplicates(dataset_dir, method=args.method, threshold=args.threshold)

    print(f"\nTotal images: {results['total_images']}")
    print(f"Exact duplicates found: {results['num_exact_duplicates']}")
    print(f"Cross-split leaks (data leakage!): {results['num_cross_split_leaks']}")

    if results["cross_split_leaks"]:
        print("\nWARNING: Data leakage detected!")
        for p1, p2, s1, s2, *_ in results["cross_split_leaks"][:20]:
            print(f"  {s1}: {Path(p1).name}  <->  {s2}: {Path(p2).name}")

    if results["exact_duplicates"]:
        print(f"\nDuplicate pairs (first 20):")
        for p1, p2, s1, s2, *_ in results["exact_duplicates"][:20]:
            print(f"  [{s1}] {Path(p1).name}  ==  [{s2}] {Path(p2).name}")

    # ── Deletion ───────────────────────────────────────────────────────────────
    deletion_summary = None
    if args.delete and results["exact_duplicates"]:
        mode = "DRY-RUN" if args.dry_run else "LIVE"
        print(f"\n=== Deleting Duplicates ({mode}) ===")
        print("Strategy: keep train > val > test; within same split keep first alphabetically.\n")
        deletion_summary = delete_duplicates(
            results["exact_duplicates"],
            dry_run=args.dry_run,
        )
        print(
            f"\nDeletion summary: "
            f"{deletion_summary['deleted']} {'would be ' if args.dry_run else ''}deleted, "
            f"{deletion_summary['skipped']} skipped, "
            f"{deletion_summary['errors']} errors."
        )
    elif args.delete and not results["exact_duplicates"]:
        print("\nNo duplicates found — nothing to delete.")

    # ── Normal class analysis ──────────────────────────────────────────────────
    all_stats = {}
    if args.analyze_normal:
        print("\n=== Normal Class Analysis ===")
        for split in ["train", "test"]:
            print(f"\n  Split: {split}")
            stats = analyze_normal_class(dataset_dir / split)
            for cls, s in stats.items():
                print(f"    {cls:15s}: brightness={s['avg_brightness']:.1f}±{s['std_brightness']:.1f}, "
                      f"res={s['avg_resolution']}")
            all_stats[split] = stats

    # ── Save results ───────────────────────────────────────────────────────────
    import json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "total_images": results["total_images"],
        "num_exact_duplicates": results["num_exact_duplicates"],
        "num_cross_split_leaks": results["num_cross_split_leaks"],
        "method": args.method,
    }
    if deletion_summary:
        report["deletion"] = deletion_summary
    if args.analyze_normal and all_stats:
        report["normal_class_analysis"] = all_stats

    with open(output_dir / "dataset_qc_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    if results["exact_duplicates"]:
        with open(output_dir / "duplicate_pairs.txt", "w") as f:
            for p1, p2, s1, s2, *_ in results["exact_duplicates"]:
                f.write(f"[{s1}] {p1}  ==  [{s2}] {p2}\n")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()