"""Run stratified K-fold cross-validation.

Usage:
    python scripts/cross_validate.py --config configs/base.yaml --folds 5
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.evaluation.metrics import compute_global_metrics, format_cv_results
from src.evaluation.report import run_inference
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.utils.config import apply_cli_overrides, load_config
from src.utils.device import get_device
from src.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate PathoVision")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.overrides:
        config = apply_cli_overrides(config, args.overrides)

    device = get_device()
    class_names = config["data"]["classes"]

    # Load full dataset (train + val combined)
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    dataset_dir = Path(config["data"]["dataset_dir"])
    full_dataset = SkinDiseaseDataset(
        root_dir=dataset_dir / "train",
        transform=val_transform,  # placeholder, will be overridden
        classes=class_names,
    )

    labels = np.array(full_dataset.get_labels())
    samples = full_dataset.samples

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{args.folds}")
        print(f"{'='*60}")

        set_seed(42 + fold)

        # Create fold datasets
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_ds = _SamplesDataset(train_samples, full_dataset.classes, train_transform)
        val_ds = _SamplesDataset(val_samples, full_dataset.classes, val_transform)

        batch_size = config["data"].get("batch_size", 16)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        )

        # Build fresh model
        model = build_model(config)
        criterion = build_loss(config)

        opt_cfg = config["training"]["optimizer"]
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_cfg["lr"],
        )

        # Train
        fold_config = config.copy()
        fold_config["experiment"]["name"] = f"fold_{fold+1}"
        fold_config["experiment"]["output_dir"] = f"outputs/cv_fold_{fold+1}"

        trainer = Trainer(
            model=model, config=fold_config,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer,
        )
        trainer.train()

        # Evaluate on fold validation
        y_true, y_pred, _ = run_inference(model, val_loader, device)
        metrics = compute_global_metrics(y_true, y_pred)
        fold_results.append(metrics)

        print(f"Fold {fold+1}: Accuracy={metrics['global_accuracy']:.2f}%, F1={metrics['f1_weighted']:.2f}%")

    # Summary with 95% CI
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY ({args.folds} folds)")
    print(f"{'='*60}")

    for metric_name in fold_results[0]:
        values = [r[metric_name] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ci_lower = mean - 1.96 * std / np.sqrt(len(values))
        ci_upper = mean + 1.96 * std / np.sqrt(len(values))
        print(f"  {metric_name}: {mean:.2f}% ± {std:.2f}% (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")

    # Formatted table for the article
    cv_table = format_cv_results(fold_results)
    print(f"\nFormatted for the article:")
    print(cv_table[["Metric", "Formatted"]].to_string(index=False))

    # Save results
    import pandas as pd
    output_dir = Path("results/04_cross_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_table.to_csv(output_dir / "cross_validation_results.csv", index=False)

    # LaTeX table
    latex = cv_table[["Metric", "Mean", "Std", "95% CI Lower", "95% CI Upper"]].to_latex(
        index=False, float_format="%.2f",
        caption=f"Cross-validation results ({args.folds}-fold stratified).",
        label="tab:cv_results",
    )
    with open(output_dir / "cross_validation_table.tex", "w") as f:
        f.write(latex)

    print(f"\nResults saved to: {output_dir}/")


class _SamplesDataset(torch.utils.data.Dataset):
    """Thin dataset wrapper for cross-validation folds."""

    def __init__(self, samples, classes, transform):
        self.samples = samples
        self.classes = classes
        self.idx_to_class = {i: c for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return {"image": image, "label": label, "class_name": self.idx_to_class[label], "path": str(path)}

    def get_labels(self):
        return [label for _, label in self.samples]


if __name__ == "__main__":
    main()
