"""Generate all evidence: prediction CSVs, ensemble, McNemar test, summary spreadsheet.

Run AFTER train_all.py completes (all 3 checkpoints must exist).

Output: evidence/
  csvs/
    M2_swin_predictions.csv
    M3_convnext_predictions.csv
    M4_efficientnetb2_predictions.csv
    M6_ensemble_predictions.csv
  reports/
    mcnemar_test.txt
    summary_results.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.device import get_device

EVIDENCE = PROJECT / "evidence"
CSVS = EVIDENCE / "csvs"
REPORTS = EVIDENCE / "reports"

MODELS = {
    "M2_swin":       ("outputs/swin/checkpoints/best.pt",     "configs/swin_tiny.yaml"),
    "M3_convnext":   ("outputs/convnext/checkpoints/best.pt", "configs/convnext_tiny.yaml"),
    "M4_improved":   ("outputs/improved/checkpoints/best.pt", "configs/improved.yaml"),
}

CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]


def load_model(ckpt_path: str, config_path: str) -> tuple:
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if config is None:
        config = load_config(config_path)
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config, device


def run_inference_with_paths(model, dataloader, device) -> pd.DataFrame:
    """Run inference and return DataFrame with path, y_true, y_pred, y_proba."""
    rows = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            paths = batch["path"]

            logits = model(images).float()
            probas = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            for i in range(len(labels)):
                rows.append({
                    "path": paths[i],
                    "y_true": CLASSES[labels[i].item()],
                    "y_true_idx": labels[i].item(),
                    "y_pred": CLASSES[preds[i]],
                    "y_pred_idx": int(preds[i]),
                    **{f"proba_{cls}": float(probas[i][j]) for j, cls in enumerate(CLASSES)},
                })

    return pd.DataFrame(rows)


def mcnemar_test(y_true_a: np.ndarray, y_pred_a: np.ndarray,
                 y_true_b: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """McNemar's test for paired nominal data.

    Compares two classifiers on the same test samples.
    H0: both classifiers have the same error rate.

    Returns dict with b, c, statistic, p_value, significant_05.
    """
    assert len(y_true_a) == len(y_true_b) == len(y_pred_a) == len(y_pred_b)
    assert np.array_equal(y_true_a, y_true_b), "y_true must be identical"

    correct_a = y_pred_a == y_true_a
    correct_b = y_pred_b == y_true_b

    # b: M4 correct, M6 wrong
    b = int(np.sum(correct_a & ~correct_b))
    # c: M4 wrong, M6 correct
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        stat = 0.0
        p_value = 1.0
    else:
        # McNemar with continuity correction
        stat = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(1 - chi2.cdf(stat, 1))

    return {
        "n_samples": len(y_true_a),
        "M4_correct_M6_wrong (b)": b,
        "M4_wrong_M6_correct (c)": c,
        "chi2_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "significant_0.05": p_value < 0.05,
    }


def main():
    print("=" * 60)
    print("EVIDENCE GENERATION - PathoVision")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # --- Step 1: Per-model inference CSVs ---
    predictions = {}

    for name, (ckpt_path, config_path) in MODELS.items():
        ckpt_full = PROJECT / ckpt_path
        config_full = PROJECT / config_path

        if not ckpt_full.exists():
            print(f"\nWARNING: SKIP {name}: checkpoint not found at {ckpt_full}")
            continue

        print(f"\n{'='*40}")
        print(f"INFERENCE: {name}")
        print(f"Checkpoint: {ckpt_full}")
        print(f"{'='*40}")

        model, config, dev = load_model(str(ckpt_full), str(config_full))
        transform = get_val_transforms(config)
        dataset_dir = Path(config["data"]["dataset_dir"])

        test_ds = SkinDiseaseDataset(
            root_dir=dataset_dir / "test",
            transform=transform,
            classes=CLASSES,
        )
        test_loader = DataLoader(
            test_ds, batch_size=config["data"].get("batch_size", 16),
            shuffle=False, num_workers=0, pin_memory=True,
        )

        df = run_inference_with_paths(model, test_loader, dev)
        acc = (df["y_pred_idx"] == df["y_true_idx"]).mean() * 100
        print(f"  Samples: {len(df)} | Accuracy: {acc:.2f}%")

        csv_path = CSVS / f"{name}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        predictions[name] = df

    # --- Step 2: Ensemble (soft voting M2+M3+M4) ---
    ensemble_members = ["M2_swin", "M3_convnext", "M4_improved"]
    available = [m for m in ensemble_members if m in predictions]

    if len(available) >= 2:
        print(f"\n{'='*40}")
        print(f"ENSEMBLE: soft voting ({', '.join(available)})")
        print(f"{'='*40}")

        # Average probabilities
        proba_cols = [f"proba_{cls}" for cls in CLASSES]
        avg_proba = np.mean([predictions[m][proba_cols].values for m in available], axis=0)
        ensemble_preds = np.argmax(avg_proba, axis=1)

        df_ens = predictions[available[0]][["path", "y_true", "y_true_idx"]].copy()
        df_ens["y_pred_idx"] = ensemble_preds
        df_ens["y_pred"] = [CLASSES[i] for i in ensemble_preds]
        for j, cls in enumerate(CLASSES):
            df_ens[f"proba_{cls}"] = avg_proba[:, j]

        acc = (df_ens["y_pred_idx"] == df_ens["y_true_idx"]).mean() * 100
        print(f"  Samples: {len(df_ens)} | Accuracy: {acc:.2f}%")

        csv_path = CSVS / "M6_ensemble_predictions.csv"
        df_ens.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        predictions["M6_ensemble"] = df_ens

    # --- Step 3: McNemar test M6 vs M4 ---
    if "M6_ensemble" in predictions and "M4_improved" in predictions:
        print(f"\n{'='*40}")
        print("MCNEMAR TEST: M6 (ensemble) vs M4 (EfficientNetV2-S)")
        print(f"{'='*40}")

        df_m4 = predictions["M4_improved"]
        df_m6 = predictions["M6_ensemble"]

        result = mcnemar_test(
            df_m4["y_true_idx"].values,
            df_m4["y_pred_idx"].values,
            df_m6["y_true_idx"].values,
            df_m6["y_pred_idx"].values,
        )

        acc_m4 = (df_m4["y_pred_idx"] == df_m4["y_true_idx"]).mean() * 100
        acc_m6 = (df_m6["y_pred_idx"] == df_m6["y_true_idx"]).mean() * 100
        delta = acc_m6 - acc_m4

        print(f"  M4 accuracy:  {acc_m4:.2f}%")
        print(f"  M6 accuracy:  {acc_m6:.2f}%")
        print(f"  Delta (M6-M4): {delta:+.2f} pp")
        print(f"  McNemar χ²:   {result['chi2_statistic']:.4f}")
        print(f"  p-value:      {result['p_value']:.4f}")
        print(f"  Significant (α=0.05): {result['significant_0.05']}")

        # Save McNemar report
        report_path = REPORTS / "mcnemar_test.txt"
        with open(report_path, "w") as f:
            f.write("McNemar Test: M6 (Ensemble) vs M4 (EfficientNetV2-S)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"M4 accuracy:  {acc_m4:.2f}%\n")
            f.write(f"M6 accuracy:  {acc_m6:.2f}%\n")
            f.write(f"Delta:        {delta:+.2f} pp\n\n")
            f.write(f"n samples:              {result['n_samples']}\n")
            f.write(f"M4 correct, M6 wrong:   {result['M4_correct_M6_wrong (b)']}\n")
            f.write(f"M4 wrong, M6 correct:   {result['M4_wrong_M6_correct (c)']}\n")
            f.write(f"McNemar χ² (corrected): {result['chi2_statistic']:.4f}\n")
            f.write(f"p-value:                {result['p_value']:.4f}\n")
            f.write(f"Significant at α=0.05:  {result['significant_0.05']}\n")
        print(f"  Report: {report_path}")

    # --- Step 4: Summary spreadsheet ---
    print(f"\n{'='*40}")
    print("SUMMARY SPREADSHEET")
    print(f"{'='*40}")

    summary = []
    for name, df in predictions.items():
        acc = (df["y_pred_idx"] == df["y_true_idx"]).mean() * 100
        summary.append({"Model": name, "Accuracy": round(acc, 2), "Samples": len(df)})

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    summary_path = REPORTS / "summary_results.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # --- Step 5: Evidence manifest ---
    manifest = []
    for p in sorted(EVIDENCE.rglob("*")):
        if p.is_file():
            manifest.append({"file": str(p.relative_to(EVIDENCE)), "size_bytes": p.stat().st_size})
    pd.DataFrame(manifest).to_csv(EVIDENCE / "manifest.csv", index=False)

    print(f"\n{'='*40}")
    print("ALL EVIDENCE GENERATED")
    print(f"Folder: {EVIDENCE}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
