"""Generate all project figures for all trained models.

Usage: python scripts/generate_all_figures.py
Output: evidence/figures/
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

FIGURES = PROJECT / "evidence" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Run per-model evaluation + figures for all trained checkpoints
import subprocess

CHECKPOINTS = [
    ("M2_swin", "outputs/swin/checkpoints/best.pt", "configs/swin_tiny.yaml"),
    ("M3_convnext", "outputs/convnext/checkpoints/best.pt", "configs/convnext_tiny.yaml"),
    ("M4_improved", "outputs/improved/checkpoints/best.pt", "configs/improved.yaml"),
    ("M4_balanced", "outputs/checkpoints/best.pt", "configs/balanced.yaml"),
]

for name, ckpt, cfg in CHECKPOINTS:
    ckpt_path = PROJECT / ckpt
    if not ckpt_path.exists():
        print(f"SKIP {name}: checkpoint not found")
        continue

    out_dir = FIGURES / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}\n{name}\n{'='*50}")

    # 1. Full evaluation (ROC, PR, confusion matrix)
    rc = subprocess.run([
        sys.executable, str(PROJECT / "scripts" / "evaluate.py"),
        "--checkpoint", str(ckpt_path),
        "--config", str(PROJECT / cfg),
        "--output-dir", str(out_dir),
    ], cwd=PROJECT)
    print(f"  evaluate.py: exit={rc.returncode}")

    # 2. Grad-CAM figures
    rc = subprocess.run([
        sys.executable, str(PROJECT / "scripts" / "gradcam_demo.py"),
        "--model-path", str(ckpt_path),
    ], cwd=PROJECT)
    print(f"  gradcam_demo.py: exit={rc.returncode}")

print(f"\nAll figures saved to: {FIGURES}")
