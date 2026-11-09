"""Generate Grad-CAM visualizations for model interpretability.

Modes:
  --mode random      : Random samples (original behavior)
  --mode systematic  : Top-3 correct + top-3 errors PER CLASS (7×6 = 42 images)
  --mode confusions  : Focus on confusion pairs (Candidiasis↔Psoriasis, Tinea↔Psoriasis/Eczema)

Usage:
    python scripts/gradcam_demo.py --checkpoint outputs/checkpoints/best.pt
    python scripts/gradcam_demo.py --checkpoint outputs/checkpoints/best.pt --mode systematic
    python scripts/gradcam_demo.py --checkpoint outputs/checkpoints/best.pt --mode confusions
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import denormalize, get_val_transforms
from src.evaluation.gradcam import (
    generate_gradcam,
    generate_gradcam_batch,
    get_target_layer,
    visualize_gradcam,
)
from src.models.factory import build_model
from src.utils.device import get_device


# Known confusion pairs from the confusion matrix
CONFUSION_PAIRS = [
    ("Candidiasis", "Psoriasis"),   # 18.52% misclassification
    ("Tinea", "Psoriasis"),         # 20.59% misclassification
    ("Tinea", "Eczema"),            # 16.67% misclassification
    ("Psoriasis", "Eczema"),        # 8.86% misclassification
]


def systematic_gradcam(
    model: torch.nn.Module,
    dataset: SkinDiseaseDataset,
    class_names: list[str],
    device: torch.device,
    save_dir: Path,
    samples_per_class: int = 3,
) -> None:
    """Generate Grad-CAM for top correct and error samples per class.

    For each of the 7 classes: top-N most confident correct + top-N worst errors.

    Args:
        model: Trained model.
        dataset: Test dataset.
        class_names: Ordered class names.
        device: Compute device.
        save_dir: Output directory.
        samples_per_class: Number of correct/error samples per class.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    target_layer = get_target_layer(model)

    # Collect all predictions with confidence
    results_by_class: dict[str, dict[str, list]] = {
        cls: {"correct": [], "wrong": []} for cls in class_names
    }

    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    sample_idx = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        with torch.no_grad():
            logits = model(images)
            probas = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu()
            confs = probas.max(dim=1).values.cpu()

        for i in range(images.shape[0]):
            true_cls = class_names[labels[i].item()]
            pred_cls = class_names[preds[i].item()]
            conf = confs[i].item()

            entry = {
                "idx": sample_idx,
                "image": images[i].cpu(),
                "true_label": labels[i].item(),
                "pred_label": preds[i].item(),
                "confidence": conf,
                "true_class": true_cls,
                "pred_class": pred_cls,
            }

            if labels[i].item() == preds[i].item():
                results_by_class[true_cls]["correct"].append(entry)
            else:
                results_by_class[true_cls]["wrong"].append(entry)

            sample_idx += 1

    # Generate Grad-CAM for top samples per class
    for cls in class_names:
        cls_dir = save_dir / cls.replace(" ", "_")
        cls_dir.mkdir(exist_ok=True)

        # Top-N most confident correct
        correct = sorted(results_by_class[cls]["correct"], key=lambda x: -x["confidence"])
        for i, entry in enumerate(correct[:samples_per_class]):
            img_tensor = entry["image"].unsqueeze(0).to(device)
            heatmaps = generate_gradcam(model, img_tensor, target_layer=target_layer)
            img_np = denormalize(entry["image"].numpy())
            title = f"CORRECT | {cls} ({entry['confidence']:.1%})"
            visualize_gradcam(
                img_np, heatmaps[0], title=title,
                save_path=cls_dir / f"correct_{i+1}.png",
            )

        # Top-N worst errors (lowest confidence for correct class)
        wrong = sorted(results_by_class[cls]["wrong"], key=lambda x: -x["confidence"])
        for i, entry in enumerate(wrong[:samples_per_class]):
            img_tensor = entry["image"].unsqueeze(0).to(device)
            heatmaps = generate_gradcam(model, img_tensor, target_layer=target_layer)
            img_np = denormalize(entry["image"].numpy())
            title = f"ERROR | True: {cls} → Pred: {entry['pred_class']} ({entry['confidence']:.1%})"
            visualize_gradcam(
                img_np, heatmaps[0], title=title,
                save_path=cls_dir / f"error_{i+1}_as_{entry['pred_class']}.png",
            )

    plt.close("all")
    print(f"Systematic Grad-CAM: {len(class_names)} classes × {samples_per_class*2} samples")


def confusion_pair_gradcam(
    model: torch.nn.Module,
    dataset: SkinDiseaseDataset,
    class_names: list[str],
    device: torch.device,
    save_dir: Path,
    samples_per_pair: int = 4,
) -> None:
    """Generate Grad-CAM for known confusion pairs.

    Args:
        model: Trained model.
        dataset: Test dataset.
        class_names: Ordered class names.
        device: Compute device.
        save_dir: Output directory.
        samples_per_pair: Samples per confusion pair.
    """
    save_dir = save_dir / "confusion_pairs"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    target_layer = get_target_layer(model)

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    # Collect misclassifications for each confusion pair
    confusions: dict[tuple, list] = {pair: [] for pair in CONFUSION_PAIRS}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            confs = F.softmax(logits, dim=1).max(dim=1).values.cpu()

        for i in range(images.shape[0]):
            true_cls = class_names[labels[i].item()]
            pred_cls = class_names[preds[i].item()]

            for pair in CONFUSION_PAIRS:
                if (true_cls == pair[0] and pred_cls == pair[1]) or \
                   (true_cls == pair[1] and pred_cls == pair[0]):
                    confusions[pair].append({
                        "image": images[i].cpu(),
                        "true_class": true_cls,
                        "pred_class": pred_cls,
                        "confidence": confs[i].item(),
                    })

    # Generate Grad-CAM for each pair
    for pair, entries in confusions.items():
        if not entries:
            print(f"  No confusions found for {pair[0]} ↔ {pair[1]}")
            continue

        pair_dir = save_dir / f"{pair[0]}_vs_{pair[1]}"
        pair_dir.mkdir(exist_ok=True)

        for i, entry in enumerate(entries[:samples_per_pair]):
            img_tensor = entry["image"].unsqueeze(0).to(device)
            heatmaps = generate_gradcam(model, img_tensor, target_layer=target_layer)
            img_np = denormalize(entry["image"].numpy())
            title = f"True: {entry['true_class']} → Pred: {entry['pred_class']} ({entry['confidence']:.1%})"
            visualize_gradcam(
                img_np, heatmaps[0], title=title,
                save_path=pair_dir / f"confusion_{i+1}.png",
            )

        print(f"  {pair[0]} ↔ {pair[1]}: {len(entries)} confusions, saved {min(len(entries), samples_per_pair)}")

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["random", "systematic", "confusions", "all"],
                        default="all", help="Analysis mode")
    parser.add_argument("--num-samples", type=int, default=3, help="Samples per class/pair")
    parser.add_argument("--output-dir", type=str, default="results/06_gradcam")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    class_names = config["data"]["classes"]
    transform = get_val_transforms(config)
    dataset_dir = Path(config["data"]["dataset_dir"])
    output_dir = Path(args.output_dir)

    test_dataset = SkinDiseaseDataset(
        root_dir=dataset_dir / "test", transform=transform, classes=class_names,
    )

    if args.mode in ("random", "all"):
        print("\n=== Random Grad-CAM Samples ===")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=2)
        generate_gradcam_batch(
            model=model, dataloader=test_loader, class_names=class_names,
            device=device, save_dir=output_dir / "random", num_samples=20,
        )

    if args.mode in ("systematic", "all"):
        print("\n=== Systematic Grad-CAM (per class) ===")
        systematic_gradcam(
            model, test_dataset, class_names, device,
            save_dir=output_dir / "systematic", samples_per_class=args.num_samples,
        )

    if args.mode in ("confusions", "all"):
        print("\n=== Confusion Pair Grad-CAM ===")
        confusion_pair_gradcam(
            model, test_dataset, class_names, device,
            save_dir=output_dir, samples_per_pair=args.num_samples,
        )

    print(f"\nAll Grad-CAM visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
