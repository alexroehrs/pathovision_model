"""Create stratified 80/10/10 train/val/test split from merged dataset.

Reads all images from datasets/merged/ (organized by class folders),
creates a stratified 80/10/10 split into datasets/data/.

The split is stratified per class to maintain class distribution
across splits. Uses sklearn's train_test_split with fixed seed for
reproducibility.

Usage:
    python datasets/create_split.py
    python datasets/create_split.py --input datasets/merged --output datasets/data
"""

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

TARGET_CLASSES = [
    "Acne", "Candidiasis", "Eczema", "NailFungus",
    "Normal", "Psoriasis", "Tinea",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(input_dir: Path) -> dict[str, list[Path]]:
    """Collect all image paths per class from the input directory.

    Args:
        input_dir: Root of merged dataset (contains class subdirs).

    Returns:
        Dict mapping class name to list of image Paths.
    """
    class_images: dict[str, list[Path]] = {}

    for cls in TARGET_CLASSES:
        cls_dir = input_dir / cls
        if not cls_dir.exists():
            print(f"  WARNING: Class directory not found: {cls_dir}")
            class_images[cls] = []
            continue

        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ])
        class_images[cls] = images
        print(f"  {cls}: {len(images)} images")

    return class_images


def create_split(
    class_images: dict[str, list[Path]],
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Create stratified 80/10/10 train/val/test split.

    Strategy:
    - For each class, split images into train (80%), temp (20%).
    - Split temp into val (50% of temp = 10%) and test (50% of temp = 10%).
    - Copy files into output_dir/train/<class>/, val/<class>/, test/<class>/.

    Args:
        class_images: Images per class.
        output_dir: Output root directory.
        seed: Random seed for reproducibility.

    Returns:
        Manifest dict with counts per split per class.
    """
    # Create output structure
    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    manifest = {}
    total = 0

    for cls in TARGET_CLASSES:
        images = class_images[cls]
        if not images:
            manifest[f"train/{cls}"] = 0
            manifest[f"val/{cls}"] = 0
            manifest[f"test/{cls}"] = 0
            continue

        # First split: 80% train, 20% temp
        train_imgs, temp_imgs = train_test_split(
            images, test_size=0.20, random_state=seed, shuffle=True,
        )

        # Second split: temp into 50% val, 50% test (10% each of total)
        # Handle edge case: 1 image in temp -> assign to val
        if len(temp_imgs) <= 1:
            val_imgs = temp_imgs
            test_imgs = []
        else:
            val_imgs, test_imgs = train_test_split(
                temp_imgs, test_size=0.50, random_state=seed, shuffle=True,
            )

        # Copy files
        for split, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dest_dir = output_dir / split / cls
            for img_path in img_list:
                dest_path = dest_dir / img_path.name
                # Avoid overwriting (e.g. if same filename in different source dirs)
                counter = 1
                while dest_path.exists():
                    stem = img_path.stem
                    dest_path = dest_dir / f"{stem}_{counter}{img_path.suffix}"
                    counter += 1
                shutil.copy2(img_path, dest_path)

            manifest[f"{split}/{cls}"] = len(img_list)

        total += len(images)

    manifest["total"] = total

    # Print distribution
    print(f"\n{'Class':<15} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("-" * 55)
    grand = {"train": 0, "val": 0, "test": 0}
    for cls in TARGET_CLASSES:
        t = manifest.get(f"train/{cls}", 0)
        v = manifest.get(f"val/{cls}", 0)
        te = manifest.get(f"test/{cls}", 0)
        grand["train"] += t
        grand["val"] += v
        grand["test"] += te
        print(f"{cls:<15} {t:>7} {v:>7} {te:>7} {t+v+te:>7}")

    print("-" * 55)
    print(f"{'TOTAL':<15} {grand['train']:>7} {grand['val']:>7} {grand['test']:>7} {total:>7}")
    for s in ["train", "val", "test"]:
        pct = grand[s] / total * 100 if total > 0 else 0
        print(f"  {s}: {grand[s]} ({pct:.1f}%)")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 80/10/10 stratified split")
    parser.add_argument("--input", type=str, default="datasets/merged", help="Input merged dir")
    parser.add_argument("--output", type=str, default="datasets/data", help="Output data dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Run 'python datasets/organize_dataset.py' first.")
        return

    # Check if input has per-class dirs (merged format) or per-split dirs
    # If input is merged, images are directly in <class>/ dirs
    has_class_dirs = any(
        (input_dir / cls).exists() for cls in TARGET_CLASSES
    )

    if has_class_dirs:
        # Input is merged flat format: <class>/image.jpg
        print(f"Detected flat class structure in {input_dir}")
        print("Collecting images per class...")
        class_images = collect_images(input_dir)
        create_split(class_images, output_dir, seed=args.seed)
    else:
        # Input might already have train/val/test structure
        # We need to read from all splits and re-split
        print(f"No flat class dirs found. Checking for train/val/test structure...")
        all_images: dict[str, list[Path]] = {cls: [] for cls in TARGET_CLASSES}
        for split in ["train", "val", "test"]:
            split_dir = input_dir / split
            if not split_dir.exists():
                continue
            for cls in TARGET_CLASSES:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    images = sorted([
                        f for f in cls_dir.iterdir()
                        if f.suffix.lower() in VALID_EXTENSIONS
                    ])
                    all_images[cls].extend(images)
                    print(f"  {split}/{cls}: {len(images)} images")

        total_found = sum(len(v) for v in all_images.values())
        if total_found == 0:
            print("ERROR: No images found in any format.")
            print(f"Expected: {input_dir}/{{class}}/ or {input_dir}/{{split}}/{{class}}/")
            return

        print(f"\nTotal images found: {total_found}")
        create_split(all_images, output_dir, seed=args.seed)

    print(f"\nSplit created at: {output_dir}")


if __name__ == "__main__":
    main()
