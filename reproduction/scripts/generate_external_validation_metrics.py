"""Generate external validation metrics for M4 model with Wilson Confidence Intervals.

This script computes per-class metrics of the external validation set from a
predictions file or from per-class counts supplied by the user.

Usage:
    python reproduction/scripts/generate_external_validation_metrics.py \
        --predictions-csv path/to/external_predictions.csv \
        --output-dir reproduction/evidence
        
Or generate from raw counts:
    python reproduction/scripts/generate_external_validation_metrics.py \
        --counts-json path/to/counts.json \
        --output-dir reproduction/evidence
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PATHOVISION_CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    Args:
        successes: Number of correct predictions
        total: Total number of samples
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) as percentages
    """
    if total == 0:
        return 0.0, 0.0
    
    from scipy import stats
    
    # Wilson score interval (more accurate for small samples)
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / total
    
    denominator = 1 + z**2 / total
    centre_adjusted = (p + z**2 / (2*total)) / denominator
    adjustment = z * np.sqrt(p*(1-p)/total + z**2/(4*total**2)) / denominator
    
    lower = max(0.0, centre_adjusted - adjustment) * 100
    upper = min(1.0, centre_adjusted + adjustment) * 100
    
    return lower, upper


def load_predictions_csv(csv_path: Path) -> Dict[str, Dict]:
    """Load predictions from CSV and group by class."""
    class_predictions = {cls: {"correct": 0, "total": 0} for cls in PATHOVISION_CLASSES}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true = row.get("y_true", "").strip()
            y_pred = row.get("y_pred", "").strip()
            
            if y_true not in class_predictions:
                continue
                
            class_predictions[y_true]["total"] += 1
            if y_true == y_pred:
                class_predictions[y_true]["correct"] += 1
    
    return class_predictions


def generate_metrics_from_counts(class_predictions: Dict[str, Dict]) -> list:
    """Generate per-class metrics with Wilson CI."""
    rows = []
    total_correct = 0
    total_samples = 0
    
    for cls in PATHOVISION_CLASSES:
        data = class_predictions[cls]
        correct = data["correct"]
        total = data["total"]
        
        if total == 0:
            accuracy = 0.0
            ci_lower, ci_upper = 0.0, 0.0
        else:
            accuracy = (correct / total) * 100
            ci_lower, ci_upper = wilson_score_interval(correct, total)
        
        total_correct += correct
        total_samples += total
        
        rows.append({
            "class": cls,
            "n_total": total,
            "n_correct": correct,
            "accuracy_percent": accuracy,
            "ci_lower_percent": ci_lower,
            "ci_upper_percent": ci_upper,
            "ci_range": f"{ci_lower:.1f}–{ci_upper:.1f}",
        })
    
    # Add overall metrics
    if total_samples > 0:
        overall_accuracy = (total_correct / total_samples) * 100
        ci_lower, ci_upper = wilson_score_interval(total_correct, total_samples)
        
        rows.append({
            "class": "OVERALL",
            "n_total": total_samples,
            "n_correct": total_correct,
            "accuracy_percent": overall_accuracy,
            "ci_lower_percent": ci_lower,
            "ci_upper_percent": ci_upper,
            "ci_range": f"{ci_lower:.1f}–{ci_upper:.1f}",
        })
    
    return rows


def write_metrics_csv(rows: list, output_path: Path) -> None:
    """Write metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["class", "n_total", "n_correct", "accuracy_percent", 
                  "ci_lower_percent", "ci_upper_percent", "ci_range"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info("Metrics written to: %s", output_path)


def print_summary(rows: list) -> None:
    """Print formatted summary table."""
    print("\n=== External Validation M4 - Per-Class Metrics with Wilson 95% CI ===")
    print(f"{'Class':<15} {'N':>5} {'Correct':>7} {'Accuracy':>10} {'95% CI (%)':>20}")
    print("-" * 70)
    
    for row in rows:
        cls = row["class"]
        n = row["n_total"]
        correct = row["n_correct"]
        acc = row["accuracy_percent"]
        ci_range = row["ci_range"]
        
        print(f"{cls:<15} {n:>5d} {correct:>7d} {acc:>9.2f}% {ci_range:>20s}")
    
    print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate external validation metrics with Wilson CI")
    parser.add_argument("--predictions-csv", type=Path, 
                       help="CSV with y_true/y_pred columns")
    parser.add_argument("--counts-json", type=Path,
                       help="JSON with per-class counts {class: {correct, total}}")
    parser.add_argument("--output-dir", type=Path, default=Path("reproduction/evidence"),
                       help="Output directory")
    parser.add_argument("--interactive", action="store_true",
                       help="Prompt for counts interactively")
    args = parser.parse_args()
    
    # Load predictions or counts
    if args.predictions_csv:
        logger.info("Loading predictions from CSV: %s", args.predictions_csv)
        class_predictions = load_predictions_csv(args.predictions_csv)
    elif args.counts_json:
        logger.info("Loading counts from JSON: %s", args.counts_json)
        with open(args.counts_json, "r") as f:
            class_predictions = json.load(f)
    elif args.interactive:
        logger.info("Interactive mode: entering class counts")
        class_predictions = {}
        for cls in PATHOVISION_CLASSES:
            total = int(input(f"{cls} - Total images: "))
            correct = int(input(f"{cls} - Correct predictions: "))
            class_predictions[cls] = {"correct": correct, "total": total}
    else:
        parser.error("provide --predictions-csv, --counts-json or --interactive")

    # Generate metrics
    metrics = generate_metrics_from_counts(class_predictions)
    
    # Write output
    output_file = args.output_dir / "external_validation_m4_per_class.csv"
    write_metrics_csv(metrics, output_file)
    
    # Print summary
    print_summary(metrics)
    
    # Report total
    total_from_rows = sum(r["n_total"] for r in metrics if r["class"] != "OVERALL")
    print(f"\nTotal images validated: {total_from_rows}")


if __name__ == "__main__":
    main()
