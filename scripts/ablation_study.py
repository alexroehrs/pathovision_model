"""Run ablation study — train with each ablation config and compare.

Usage:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --configs configs/ablation/no_augmentation.yaml configs/ablation/no_dropout.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ablation import (
    ABLATION_CONFIGS,
    collect_ablation_results,
    generate_ablation_latex,
    run_ablation_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config paths (default: all ablation configs)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only collect results")
    parser.add_argument("--output-dir", type=str, default="results/05_ablation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.skip_training:
        configs = {}
        if args.configs:
            for c in args.configs:
                configs[Path(c).stem] = c
        else:
            configs = ABLATION_CONFIGS

        print(f"\n=== Ablation Study: {len(configs)} experiments ===\n")

        for name, config_path in configs.items():
            if not Path(config_path).exists():
                print(f"  SKIP: {name} ({config_path} not found)")
                continue
            print(f"\n--- {name} ---")
            run_ablation_experiment(config_path, name)

    # Collect results
    print("\n=== Collecting Results ===")
    results_df = collect_ablation_results()

    if results_df.empty:
        print("No results found. Run training first.")
        return

    print("\nAblation Results:")
    print(results_df.to_string(index=False, float_format="%.2f"))

    # Generate LaTeX
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_ablation_latex(results_df, save_path=output_dir / "ablation_table.tex")
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
