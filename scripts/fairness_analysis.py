"""Fairness analysis — skin tone stratification.

Usage:
    python scripts/fairness_analysis.py --checkpoint outputs/checkpoints/best.pt
    python scripts/fairness_analysis.py --dataset-dir datasets/merged/test
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_val_transforms
from src.evaluation.fairness import (
    analyze_dataset_skin_tones,
    compute_fairness_metrics,
    plot_fairness_comparison,
    plot_skin_tone_distribution,
)
from src.evaluation.report import run_inference
from src.models.factory import build_model
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairness analysis")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--dataset-dir", type=str, default="datasets/merged/test")
    parser.add_argument("--output-dir", type=str, default="results/07_fairness")
    parser.add_argument("--max-per-class", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]

    # Step 1: Analyze skin tone distribution
    print("\n=== Skin Tone Distribution Analysis ===")
    tone_df = analyze_dataset_skin_tones(
        args.dataset_dir, classes, max_per_class=args.max_per_class,
    )

    if tone_df.empty:
        print("No images found. Check --dataset-dir.")
        return

    print(f"\nAnalyzed {len(tone_df)} images")
    print(f"\nDistribution by tone:")
    print(tone_df["skin_tone_group"].value_counts().sort_index().to_string())

    print(f"\nDistribution by class:")
    ct = tone_df.groupby(["class", "skin_tone_group"]).size().unstack(fill_value=0)
    print(ct.to_string())

    # Save distribution plot
    plot_skin_tone_distribution(tone_df, save_path=output_dir / "skin_tone_distribution.png")
    tone_df.to_csv(output_dir / "skin_tone_annotations.csv", index=False)

    # Step 2: Stratified model evaluation (if checkpoint provided)
    if args.checkpoint:
        print("\n=== Stratified Model Evaluation ===")
        device = get_device()
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint["config"]

        model = build_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        transform = get_val_transforms(config)
        test_dataset = SkinDiseaseDataset(
            root_dir=Path(args.dataset_dir), transform=transform, classes=classes,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
        )

        y_true, y_pred, _ = run_inference(model, test_loader, device)

        # Map predictions to skin tones
        paths = [s[0] for s in test_dataset.samples]
        path_to_tone = {row["path"]: row["skin_tone_group"] for _, row in tone_df.iterrows()}
        skin_tones = [path_to_tone.get(str(p), "Unknown") for p in paths]

        # Compute fairness metrics
        fairness_df = compute_fairness_metrics(y_true, y_pred, skin_tones, classes)
        print("\nFairness Metrics by Skin Tone:")
        print(fairness_df.to_string(index=False, float_format="%.2f"))

        fairness_df.to_csv(output_dir / "fairness_metrics.csv", index=False)
        plot_fairness_comparison(fairness_df, save_path=output_dir / "fairness_comparison.png")

        # Equity gap
        if len(fairness_df) >= 2:
            max_acc = fairness_df["Accuracy (%)"].max()
            min_acc = fairness_df["Accuracy (%)"].min()
            print(f"\nEquity gap (Accuracy): {max_acc - min_acc:.2f}pp")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
