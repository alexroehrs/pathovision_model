"""Sequential training: M4 (EfficientNetB2) -> M2 (Swin) -> M3 (ConvNeXt)."""
import subprocess, sys, time
from pathlib import Path

MODELS = [
    ("M4-EfficientNetB2", "configs/balanced.yaml", "outputs"),
    ("M2-Swin", "configs/swin_tiny.yaml", "outputs/swin"),
    ("M3-ConvNeXt", "configs/convnext_tiny.yaml", "outputs/convnext"),
]

PROJECT = Path(__file__).parent.parent
results = {}

for name, config, outdir in MODELS:
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"Config: {config} | Output: {outdir}")
    print(f"{'='*60}")
    start = time.time()
    rc = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", config,
         f"experiment.output_dir={outdir}"],
        cwd=PROJECT,
    ).returncode
    elapsed = time.time() - start
    results[name] = {"exit": rc, "time_h": elapsed / 3600}
    checkpoint = PROJECT / outdir / "checkpoints" / "best.pt"
    print(f"\n{name}: exit={rc} | time={elapsed/3600:.2f}h | best.pt={'OK' if checkpoint.exists() else 'MISSING'}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, r in results.items():
    print(f"  {name}: exit={r['exit']}, {r['time_h']:.2f}h")
