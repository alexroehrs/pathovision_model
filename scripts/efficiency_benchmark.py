"""Benchmark computational efficiency of all model architectures.

Usage:
    python scripts/efficiency_benchmark.py
    python scripts/efficiency_benchmark.py --configs configs/base.yaml configs/swin_tiny.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

from src.evaluation.efficiency import benchmark_model
from src.models.factory import build_model
from src.utils.config import load_config
from src.utils.device import get_device


DEFAULT_CONFIGS = [
    ("EfficientNetB2 (Baseline)", "configs/base.yaml"),
    ("EfficientNetV2-S", "configs/improved.yaml"),
    ("Swin-Tiny", "configs/swin_tiny.yaml"),
    ("ConvNeXt-Tiny", "configs/convnext_tiny.yaml"),
    ("ViT-Small (DINOv2)", "configs/dinov2_vit.yaml"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model efficiency")
    parser.add_argument("--configs", nargs="+", default=None, help="Config paths to benchmark")
    parser.add_argument("--names", nargs="+", default=None, help="Model names")
    parser.add_argument("--output", type=str, default="results/09_efficiency/efficiency.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = get_device()
    results = []

    if args.configs:
        configs = list(zip(args.names or args.configs, args.configs))
    else:
        configs = DEFAULT_CONFIGS

    for name, config_path in configs:
        if not Path(config_path).exists():
            print(f"  SKIP: {name} ({config_path} not found)")
            continue

        print(f"\n--- Benchmarking: {name} ---")
        config = load_config(config_path)
        input_size = tuple(config["data"]["input_size"])

        model = build_model(config)
        model = model.to(device)
        model.eval()

        metrics = benchmark_model(model, input_size, device)
        metrics["Model"] = name
        metrics["Input Size"] = f"{input_size[0]}×{input_size[1]}"
        results.append(metrics)

        del model
        torch.cuda.empty_cache()

    # Results table
    df = pd.DataFrame(results)
    cols = ["Model", "Input Size", "params_total_M", "model_size_MB",
            "gflops", "inference_mean_ms", "fps"]
    cols = [c for c in cols if c in df.columns]

    print(f"\n{'='*80}")
    print("EFFICIENCY BENCHMARK")
    print(f"{'='*80}")
    print(df[cols].to_string(index=False, float_format="%.2f"))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # LaTeX
    latex = df[cols].to_latex(
        index=False, float_format="%.2f",
        caption="Computational efficiency comparison of evaluated architectures.",
        label="tab:efficiency",
    )
    with open(output_path.with_suffix(".tex"), "w") as f:
        f.write(latex)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
