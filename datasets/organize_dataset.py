"""Organize and merge the two Kaggle datasets into a unified structure.

Creates the following directory layout:
    datasets/merged/
        train/
            Acne/
            Candidiasis/
            Eczema/
            NailFungus/
            Normal/
            Psoriasis/
            Tinea/
        val/
            (same classes)
        test/
            (same classes)

Usage:
    python datasets/organize_dataset.py
    python datasets/organize_dataset.py --raw-dir datasets/raw --output datasets/merged
"""

import argparse
import json
import shutil
from pathlib import Path

# Target classes matching the project
TARGET_CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]

# Mapping from source directory names to target class names
# Adjust these based on the actual directory names in the downloaded datasets
CLASS_MAPPING = {
    # Human Skin Diseases dataset (Youssef Mohamed, Kaggle)
    "Acne": "Acne",
    "acne": "Acne",
    "Candidiasis": "Candidiasis",
    "candidiasis": "Candidiasis",
    "Eczema": "Eczema",
    "eczema": "Eczema",
    "Nail Fungus": "NailFungus",
    "NailFungus": "NailFungus",
    "nail fungus": "NailFungus",
    "Nail_Fungus": "NailFungus",
    "Normal": "Normal",
    "normal": "Normal",
    "Normal Skin": "Normal",
    "Psoriasis": "Psoriasis",
    "psoriasis": "Psoriasis",
    "Tinea": "Tinea",
    "tinea": "Tinea",
    "Ringworm": "Tinea",
    "ringworm": "Tinea",
    # Dermnet dataset (Shubham Goel, Kaggle) - longer directory names
    "Acne and Rosacea Photos": "Acne",
    "Eczema Photos": "Eczema",
    "Nail Fungus and other Nail Disease": "NailFungus",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Tinea",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def find_image_dirs(raw_dir: Path) -> list[tuple[Path, str, str]]:
    """Recursively find directories containing images and map to target classes.

    Returns:
        List of (directory_path, source_split, target_class) tuples.
    """
    results = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_dir():
            continue

        # Check if directory name maps to a target class
        dir_name = path.name
        target_class = CLASS_MAPPING.get(dir_name)
        if target_class is None:
            continue

        # Check if it contains images
        images = [f for f in path.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        if not images:
            continue

        # Detect split from parent path
        source_split = "unknown"
        parent_parts = [p.lower() for p in path.parts]
        if "valid" in parent_parts or "validation" in parent_parts:
            source_split = "val"
        elif "train" in parent_parts:
            source_split = "train"
        elif "test" in parent_parts:
            source_split = "test"
        elif "val" in parent_parts or "validation" in parent_parts:
            source_split = "val"

        results.append((path, source_split, target_class))
        print(f"  Found: {path.relative_to(raw_dir)} -> {target_class} ({source_split}, {len(images)} images)")

    return results


def organize_dataset(raw_dir: Path, output_dir: Path) -> dict:
    """Organize images from raw downloads into the target structure.

    Args:
        raw_dir: Directory containing raw downloaded datasets.
        output_dir: Output directory for organized dataset.

    Returns:
        Manifest dictionary with counts.
    """
    print(f"Scanning raw directory: {raw_dir}")
    image_dirs = find_image_dirs(raw_dir)

    if not image_dirs:
        print("\nERROR: No matching image directories found!")
        print("Expected class names:", list(CLASS_MAPPING.keys()))
        print(f"Check that datasets are extracted in: {raw_dir}")
        return {}

    # Create target structure
    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    manifest = {"total": 0, "per_split": {}, "per_class": {}}
    copy_count = 0

    for src_dir, source_split, target_class in image_dirs:
        if source_split == "unknown":
            # Default unknown splits to train
            source_split = "train"

        dest_dir = output_dir / source_split / target_class

        images = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])

        for img_path in images:
            # Avoid filename collisions with prefix
            dest_name = f"{src_dir.parent.name}_{img_path.name}"
            dest_path = dest_dir / dest_name

            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                copy_count += 1

    # Count final distribution
    total = 0
    for split in ["train", "val", "test"]:
        split_count = 0
        for cls in TARGET_CLASSES:
            cls_dir = output_dir / split / cls
            count = len([f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])
            split_count += count

            key = f"{split}/{cls}"
            manifest[key] = count

        manifest["per_split"][split] = split_count
        total += split_count

    manifest["total"] = total

    print(f"\nOrganized {copy_count} images -> {output_dir}")
    print(f"Total: {total}")

    # Print distribution table
    print(f"\n{'Class':<15} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("-" * 50)
    for cls in TARGET_CLASSES:
        train_n = manifest.get(f"train/{cls}", 0)
        val_n = manifest.get(f"val/{cls}", 0)
        test_n = manifest.get(f"test/{cls}", 0)
        cls_total = train_n + val_n + test_n
        print(f"{cls:<15} {train_n:>7} {val_n:>7} {test_n:>7} {cls_total:>7}")

    for split in ["train", "val", "test"]:
        print(f"{split}: {manifest['per_split'][split]}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize PathoVision datasets")
    parser.add_argument("--raw-dir", type=str, default="datasets/raw", help="Raw downloads directory")
    parser.add_argument("--output", type=str, default="datasets/merged", help="Output directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output)

    if not raw_dir.exists():
        print(f"Raw directory not found: {raw_dir}")
        print("Run 'python datasets/download_kaggle.py' first.")
        return

    organize_dataset(raw_dir, output_dir)


if __name__ == "__main__":
    main()
