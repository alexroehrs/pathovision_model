"""Create radar chart figures for the PathoVision project.

Generates one radar chart per model configuration showing per-class
performance metrics (Accuracy, Recall, Specificity, Precision, F1-Score).

Must be executed from the repository root:
    python scripts/generate_radarchart_models.py

Outputs are saved to results/<model_dir>/figures/
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# DATA
# ============================================================================

datasets = {
    'M1': {
        'output_dir': Path("results/02_baseline/figures"),
        'filename': 'pathovision_baseline_radarchart.png',
        'title': 'PATHOVISION - EfficientNet-B2 (M1)',
        'data': {
            'Acne':        [96.45, 85.34, 98.03, 86.09, 85.71],
            'Candidiasis': [97.95, 61.54, 99.00, 64.00, 62.75],
            'Eczema':      [90.96, 77.98, 93.82, 73.60, 75.72],
            'NailFungus':  [96.66, 89.68, 97.76, 86.26, 87.94],
            'Normal':      [99.78,   100, 99.73, 98.92, 99.46],
            'Psoriasis':   [90.10, 68.05, 95.00, 75.16, 71.43],
            'Tinea':       [90.42, 68.57, 94.30, 68.09, 68.33],
        }
    },
    'M1b': {
        'output_dir': Path("results/02b_baseline_balanced/figures"),
        'filename': 'efficientnetb2_balanced_ce_radarchart.png',
        'title': 'PATHOVISION - EfficientNet-B2 (M1b)',
        'data': {
            'Acne':        [97.20, 87.93, 98.52, 89.47, 88.70],
            'Candidiasis': [99.03, 76.92, 99.67, 86.96, 81.63],
            'Eczema':      [92.79, 80.36, 95.53, 79.88, 80.12],
            'NailFungus':  [97.42, 92.86, 98.13, 88.64, 90.70],
            'Normal':      [99.57, 99.46, 99.60, 98.39, 98.92],
            'Psoriasis':   [90.74, 73.96, 94.47, 74.85, 74.40],
            'Tinea':       [93.97, 79.29, 96.58, 80.43, 79.86],
        }
    },
    'M2': {
        'output_dir': Path("results/03_m2_v2s_balanced/figures"),
        'filename': 'pathovision_improved_radarchart.png',
        'title': 'PATHOVISION - EfficientNet-V2-S (M2)',
        'data': {
            'Acne':        [96.45, 90.52, 97.29, 82.68, 86.42],
            'Candidiasis': [98.49, 80.77, 99.00, 70.00, 75.00],
            'Eczema':      [91.28, 73.81, 95.14, 77.02, 75.38],
            'NailFungus':  [98.06, 92.86, 98.88, 92.86, 92.86],
            'Normal':      [99.89,   100, 99.87, 99.46, 99.73],
            'Psoriasis':   [89.45, 71.01, 93.55, 71.01, 71.01],
            'Tinea':       [94.08, 77.14, 97.08, 82.44, 79.70],
        }
    },
    'M2b': {
        'output_dir': Path("results/03_m2b_v2s_nobal/figures"),
        'filename': 'efficientnetv2s_nobal_radarchart.png',
        'title': 'PATHOVISION - EfficientNet-V2-S (M2b)',
        'data': {
            'Acne':        [96.02, 83.62, 97.79, 84.35, 83.98],
            'Candidiasis': [98.39, 61.54, 99.45, 76.19, 68.09],
            'Eczema':      [93.00, 79.17, 96.06, 81.60, 80.36],
            'NailFungus':  [97.52, 92.86, 98.26, 89.31, 91.05],
            'Normal':      [99.78,   100, 99.73, 98.92, 99.46],
            'Psoriasis':   [90.53, 72.19, 94.61, 74.85, 73.49],
            'Tinea':       [93.54, 82.14, 95.56, 76.67, 79.31],
        }
    },
    'M3': {
        'output_dir': Path("results/03_m3_swin_balanced/figures"),
        'filename': 'swin_tiny_radarchart.png',
        'title': 'PATHOVISION - Swin-Tiny (M3)',
        'data': {
            'Acne':        [98.06, 93.10, 98.77, 91.53, 92.31],
            'Candidiasis': [99.03, 80.77, 99.56, 84.00, 82.35],
            'Eczema':      [92.25, 75.00, 96.06, 80.77, 77.78],
            'NailFungus':  [97.85, 92.86, 98.63, 91.41, 92.13],
            'Normal':      [99.78,   100, 99.73, 98.92, 99.46],
            'Psoriasis':   [90.31, 73.96, 93.95, 73.10, 73.53],
            'Tinea':       [93.43, 80.00, 95.82, 77.24, 78.60],
        }
    },
    'M3b': {
        'output_dir': Path("results/03_m3b_swin_nobal/figures"),
        'filename': 'swin_tiny_nobal_radarchart.png',
        'title': 'PATHOVISION - Swin-Tiny (M3b)',
        'data': {
            'Acne':        [97.95, 94.83, 98.40, 89.43, 92.05],
            'Candidiasis': [98.92, 73.08, 99.67, 86.36, 79.17],
            'Eczema':      [93.00, 77.98, 96.32, 82.39, 80.12],
            'NailFungus':  [97.95, 92.86, 98.75, 92.13, 92.49],
            'Normal':      [99.78,   100, 99.73, 98.92, 99.46],
            'Psoriasis':   [90.53, 76.92, 93.55, 72.63, 74.71],
            'Tinea':       [94.29, 78.57, 97.08, 82.71, 80.59],
        }
    },
    'M4': {
        'output_dir': Path("results/03_m4_convnext_balanced/figures"),
        'filename': 'convnext_tiny_radarchart.png',
        'title': 'PATHOVISION - ConvNeXt-T (M4)',
        'data': {
            'Acne':        [97.63, 93.10, 98.28, 88.52, 90.76],
            'Candidiasis': [98.82, 65.38, 99.78, 89.47, 75.56],
            'Eczema':      [93.76, 84.52, 95.80, 81.61, 83.04],
            'NailFungus':  [98.28, 94.44, 98.88, 92.97, 93.70],
            'Normal':      [99.78,   100, 99.73, 98.92, 99.46],
            'Psoriasis':   [92.79, 75.74, 96.58, 83.12, 79.26],
            'Tinea':       [94.62, 84.29, 96.45, 80.82, 82.52],
        }
    },
    'M4b': {
        'output_dir': Path("results/03_m4b_convnext_nobal/figures"),
        'filename': 'convnext_tiny_nobal_radarchart.png',
        'title': 'PATHOVISION - ConvNeXt-T (M4b)',
        'data': {
            'Acne':        [98.17, 92.24, 99.02, 93.04, 92.64],
            'Candidiasis': [98.49, 57.69, 99.67, 83.33, 68.18],
            'Eczema':      [94.08, 85.71, 95.93, 82.29, 83.97],
            'NailFungus':  [97.95, 91.27, 99.00, 93.50, 92.37],
            'Normal':      [99.89,   100, 99.87, 99.46, 99.73],
            'Psoriasis':   [92.57, 79.29, 95.53, 79.76, 79.53],
            'Tinea':       [94.29, 82.86, 96.32, 80.00, 81.40],
        }
    },
    'M4b': {
        'output_dir': Path("results/05_focal_balanced/figures"),
        'filename': 'efficientnetb2_balanced_radarchart.png',
        'title': 'PATHOVISION - EfficientNet-B2 Focal Loss (M5)',
        'data': {
            'Acne':        [96.45, 88.79, 97.54, 83.74, 86.19],
            'Candidiasis': [98.60, 80.77, 99.11, 72.41, 76.36],
            'Eczema':      [90.64, 77.98, 93.43, 72.38, 75.07],
            'NailFungus':  [96.66, 91.27, 97.51, 85.19, 88.12],
            'Normal':      [99.89,   100, 99.87, 99.46, 99.73],
            'Psoriasis':   [90.10, 66.86, 95.26, 75.84, 71.07],
            'Tinea':       [92.57, 70.71, 96.45, 77.95, 74.16],
        }
    },
}

# ============================================================================
# CONSTANTS
# ============================================================================

METRICS = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'F1-Score']

COLORS = {
    'Accuracy':    '#1f77b4',
    'Recall':      '#ff7f0e',
    'Specificity': '#2ca02c',
    'Precision':   '#17becf',
    'F1-Score':    '#9467bd',
}

# ============================================================================
# PLOTTING
# ============================================================================

def plot_radar(model_id: str, config: dict) -> None:
    """Generate and save a radar chart for a single model configuration."""
    df = pd.DataFrame(config['data'], index=METRICS)
    classes = list(config['data'].keys())

    num_vars = len(classes)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for metric in METRICS:
        values = df.loc[metric].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=metric, color=COLORS[metric])
        ax.fill(angles, values, alpha=0.1, color=COLORS[metric])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, size=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=10)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
    ax.set_title(config['title'], size=16, fontweight='bold', pad=30)

    plt.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.15), fontsize=15, ncol=5, frameon=True)
    plt.tight_layout()

    output_path = config['output_dir'] / config['filename']
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    for config in datasets.values():
        config['output_dir'].mkdir(parents=True, exist_ok=True)

    for model_id, config in datasets.items():
        plot_radar(model_id, config)

    print("\nDone!")