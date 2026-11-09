"""Create composite Grad-CAM figures for the PathoVision article.

Figure 1: Correctly classified examples (one per class)
Figure 2: Misclassified examples (confusion pairs)

Uses real Grad-CAM outputs from results/06_gradcam/systematic/
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = Path("results/06_gradcam/systematic")
OUTPUT_DIR = Path("article/figures")

CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]
CLASS_LABELS = ["Acne", "Candidiasis", "Eczema", "Nail Fungus", "Normal", "Psoriasis", "Tinea"]


def create_figure_correct() -> None:
    """Figure 1: Grad-CAM for correctly classified examples."""
    fig, axes = plt.subplots(7, 3, figsize=(12, 28))
    fig.suptitle(
        "Grad-CAM Visualizations for Correctly Classified Examples",
        fontsize=16, fontweight="bold", y=0.995,
    )

    for i, (cls, label) in enumerate(zip(CLASSES, CLASS_LABELS)):
        img_path = RESULTS_DIR / cls / "correct_1.png"
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        w, h = img.size
        third = w // 3

        original = img.crop((0, 0, third, h))
        heatmap = img.crop((third, 0, 2 * third, h))
        overlay = img.crop((2 * third, 0, w, h))

        for j, (subimg, title) in enumerate(
            [(original, "Original"), (heatmap, "Grad-CAM"), (overlay, "Overlay")]
        ):
            axes[i, j].imshow(subimg)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(label, fontsize=12, fontweight="bold", rotation=0,
                                       labelpad=80, va="center")
            if i == 0:
                axes[i, j].set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0.08, 0, 1, 0.99])
    output_path = OUTPUT_DIR / "gradcam_correct.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_figure_errors() -> None:
    """Figure 2: Grad-CAM for misclassified examples (confusion pairs)."""
    confusion_pairs = [
        ("Candidiasis", "error_1_as_Tinea.png", "Candidiasis → Tinea"),
        ("Candidiasis", "error_1_as_Psoriasis.png", "Candidiasis → Psoriasis"),
        ("Eczema", "error_1_as_Psoriasis.png", "Eczema → Psoriasis"),
        ("Eczema", "error_1_as_Acne.png", "Eczema → Acne"),
        ("Tinea", "error_1_as_Eczema.png", "Tinea → Eczema"),
        ("Tinea", "error_2_as_Candidiasis.png", "Tinea → Candidiasis"),
        ("Psoriasis", "error_2_as_Eczema.png", "Psoriasis → Eczema"),
    ]

    valid_pairs = [
        (cls, fname, label)
        for cls, fname, label in confusion_pairs
        if (RESULTS_DIR / cls / fname).exists()
    ]

    n_rows = len(valid_pairs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    fig.suptitle(
        "Grad-CAM Visualizations for Misclassified Examples (Confusion Pairs)",
        fontsize=16, fontweight="bold", y=0.995,
    )

    for i, (cls, fname, label) in enumerate(valid_pairs):
        img_path = RESULTS_DIR / cls / fname
        img = Image.open(img_path)
        w, h = img.size
        third = w // 3

        original = img.crop((0, 0, third, h))
        heatmap = img.crop((third, 0, 2 * third, h))
        overlay = img.crop((2 * third, 0, w, h))

        for j, (subimg, title) in enumerate(
            [(original, "Original"), (heatmap, "Grad-CAM"), (overlay, "Overlay")]
        ):
            axes[i, j].imshow(subimg)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(label, fontsize=11, fontweight="bold", rotation=0,
                                       labelpad=110, va="center")
            if i == 0:
                axes[i, j].set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0.1, 0, 1, 0.99])
    output_path = OUTPUT_DIR / "gradcam_errors.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_figure_correct()
    create_figure_errors()
    print("Done!")
