"""External validation — evaluate a trained model on the unified external dataset.

The unified external validation dataset combines four Kaggle sources and SD-198,
as documented in Online Resource 1 (Table S4) of the article:
- 4 Kaggle sources: acne_dataset, nail_disease, skin_normal, skin_10classes
- 1 HuggingFace source: SD-198 (with REGEX label mapping)

The external dataset must be preprocessed via:
    python datasets/build_external_unified.py --internal-dir datasets/merged ...

Usage:
    python scripts/external_validation_unified.py \
        --checkpoint <path/to/best.pt> \
        --external-dir datasets/external_unified \
        --output-dir reproduction/evidence/external_validation_unified_m4
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from collections import defaultdict, Counter

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import compute_full_report
from src.evaluation.report import run_inference
from src.models.factory import build_model
from src.utils.device import get_device


PATHOVISION_CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_external_unified(checkpoint_path: str, external_dir: Path, output_dir: Path) -> None:
    """
    Validate a trained model on the unified external validation dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        external_dir: Path to external_unified directory
        output_dir: Where to save results
    """
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading model from checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info("Loading external unified dataset from: %s", external_dir)
    transform = get_val_transforms(config)
    
    # Load dataset (should have class subdirectories)
    external_dataset = SkinDiseaseDataset(
        root_dir=external_dir,
        transform=transform,
        classes=PATHOVISION_CLASSES,
    )
    
    if len(external_dataset) == 0:
        logger.error("No images found in external directory. Check structure: %s", external_dir)
        return
    
    external_loader = torch.utils.data.DataLoader(
        external_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Run inference
    logger.info("Running inference on %d external images...", len(external_dataset))
    y_true, y_pred, y_proba = run_inference(model, external_loader, device)
    
    # Compute metrics
    logger.info("Computing metrics...")
    report = compute_full_report(y_true, y_pred, PATHOVISION_CLASSES)
    
    # Print summary
    print(f"\n{'='*80}")
    print("External Validation Results")
    print("Dataset: unified external set (4 Kaggle sources + SD-198)")
    print(f"{'='*80}")
    print(f"  Global Accuracy:        {report['global']['global_accuracy']:>7.2f}%")
    print(f"  Weighted F1-Score:      {report['global']['f1_weighted']:>7.2f}%")
    print(f"  Macro F1-Score:         {report['global']['f1_macro']:>7.2f}%")
    print(f"  Images Evaluated:       {len(external_dataset):>7d}")
    print(f"\n{report['classification_report']}")
    print(f"{'='*80}\n")
    
    # Save per-class metrics
    logger.info("Saving per-class metrics...")
    report["per_class"].to_csv(output_dir / "external_validation_m4_per_class.csv", index=False)
    
    # Save predictions CSV
    logger.info("Saving prediction CSV...")
    predictions_csv = output_dir / "external_validation_m4_predictions.csv"
    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["y_true", "y_pred", "y_true_idx", "y_pred_idx"] +
                       [f"proba_{cls}" for cls in PATHOVISION_CLASSES]
        )
        writer.writeheader()
        
        for i in range(len(y_true)):
            row = {
                "y_true": y_true[i],
                "y_pred": y_pred[i],
                "y_true_idx": PATHOVISION_CLASSES.index(y_true[i]),
                "y_pred_idx": PATHOVISION_CLASSES.index(y_pred[i]),
            }
            for j, cls in enumerate(PATHOVISION_CLASSES):
                row[f"proba_{cls}"] = y_proba[i, j]
            writer.writerow(row)
    
    logger.info("Predictions saved to: %s", predictions_csv)
    
    # Generate manifest
    logger.info("Creating dataset manifest...")
    manifest = {
        "dataset_name": "external_unified (4 Kaggle + SD-198)",
        "total_images": len(external_dataset),
        "classes": PATHOVISION_CLASSES,
        "source_manifest": str(external_dir / "external_manifest.csv"),
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        import json
        json.dump(manifest, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Per-class metrics: external_validation_m4_per_class.csv")
    print(f"  - Predictions: external_validation_m4_predictions.csv")
    print(f"  - Manifest: manifest.json")
    print(f"\nFor article metrics, run:")
    print(f"  python reproduction/scripts/generate_external_validation_metrics.py \\")
    print(f"    --predictions-csv {predictions_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a trained model on the unified external dataset"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (e.g., outputs/checkpoints/M4_improved_best.pt)")
    parser.add_argument("--external-dir", type=Path, default=Path("datasets/external_unified"),
                       help="Path to external_unified directory (must be preprocessed)")
    parser.add_argument("--output-dir", type=Path, 
                       default=Path("reproduction/evidence/external_validation_unified_m4"),
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)
    
    # Verify external dir exists
    if not args.external_dir.exists():
        logger.error("External directory not found: %s", args.external_dir)
        logger.error("First run: python datasets/build_external_unified.py")
        sys.exit(1)
    
    validate_external_unified(args.checkpoint, args.external_dir, args.output_dir)


if __name__ == "__main__":
    main()
