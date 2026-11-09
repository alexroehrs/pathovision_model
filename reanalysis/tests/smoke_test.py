"""End-to-end smoke test of the re-execution pipeline on synthetic images.

Creates a tiny folder-per-class dataset (7 classes, colour-coded noise images)
plus a fake provenance table and a 3-class external set in a temporary root,
then runs stages 03-23 with 1 epoch, 64x64 inputs and randomly initialised
backbones (no download).  It checks only that every stage runs and produces
its output files; the numbers are meaningless.  Takes a few minutes on CPU.

    python reanalysis/tests/smoke_test.py [--keep] [--root DIR]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REANALYSIS = HERE.parent
PROJECT = REANALYSIS.parent
CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]
COLORS = [(200, 120, 110), (230, 200, 170), (190, 90, 80), (240, 220, 200), (235, 200, 180), (220, 130, 120), (200, 150, 120)]


def make_image(path: Path, color, rng, bright=0.0):
    img = np.clip(rng.normal(np.array(color) + bright, 25, size=(96, 128, 3)), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def build(root: Path):
    rng = np.random.default_rng(0)
    data = root / "datasets" / "data"
    prov = []
    for split, n in (("train", 14), ("val", 5), ("test", 6)):
        for k, cls in enumerate(CLASSES):
            d = data / split / cls; d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                name = f"{split}_{cls}_{i}.jpg"
                make_image(d / name, COLORS[k], rng, bright=30 if cls == "Normal" else 0)
                prov.append({"relative_path": f"{split}/{cls}/{name}", "split": split, "class": cls, "md5": f"{split}{cls}{i}",
                             "source_dataset": "skin_diseases_youssef" if (i % 2 == 0 or cls in ("Normal", "Candidiasis")) else "dermnet_goel",
                             "source_folder": cls, "source_split": "train", "source_file": name})
    pd.DataFrame(prov).to_csv(data / "provenance.csv", index=False)
    # raw provenance (stage 15) and a tiny raw tree with inherited train/test partitions (stage 20)
    qc = root / "results" / "dataset_qc"; qc.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    for r in prov:
        raw_rows.append({"source_dataset": r["source_dataset"], "source_folder": r["source_folder"], "source_split": "train",
                         "source_file": ("psoriasis-" if r["class"] == "Psoriasis" else "lichen-planus-" if r["class"] == "Eczema" else "") + r["source_file"],
                         "class": r["class"], "md5": r["md5"]})
    pd.DataFrame(raw_rows).to_csv(qc / "raw_provenance.csv", index=False)
    raw = root / "datasets" / "raw"
    for key in ("skin_diseases_youssef", "dermnet_goel"):
        for split, n in (("train", 10), ("test", 3)):
            for k, cls in enumerate(CLASSES):
                d = raw / key / split / cls; d.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    make_image(d / f"{key}_{split}_{cls}_{i}.jpg", COLORS[k], rng)
                if split == "test":   # one exact cross-split duplicate per class
                    shutil.copy2(d / f"{key}_test_{cls}_0.jpg", raw / key / "train" / cls / f"dup_{cls}.jpg")
    ext = root / "datasets" / "external_unified"
    man = []
    for k, cls in enumerate(["Acne", "Eczema", "Psoriasis", "NailFungus"]):
        d = ext / cls; d.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            name = f"ext_{cls}_{i}.jpg"
            make_image(d / name, COLORS[CLASSES.index(cls)], rng, bright=-20)
            man.append({"path": f"{cls}/{name}", "source": "sd198" if i % 2 else "acne_dataset", "source_label": cls.lower(), "class": cls})
    pd.DataFrame(man).to_csv(ext / "external_manifest.csv", index=False)


def run(stage: str, *extra, env):
    cmd = [sys.executable, str(REANALYSIS / stage), *extra]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=PROJECT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()
    root = (args.root or Path(tempfile.mkdtemp(prefix="pv_smoke_"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    build(root)
    env = dict(os.environ, REANALYSIS_ROOT=str(root), REANALYSIS_DATA_ROOT=str(root / "datasets"), PYTHONPATH=str(PROJECT))
    ov = ["--override", "training.epochs=1", "data.input_size=[64,64]", "data.batch_size=8", "data.num_workers=0",
          "model.pretrained=false", "training.mixed_precision=false"]
    models = ["M1", "M4", "M2", "M3", "M4b"]
    run("s03_train_all.py", "--models", "M1", "M4", "M2", "M4b", "--seeds", "42", "43", *ov, env=env)
    # Swin-Tiny has a fixed 224x224 input, so it keeps its native resolution
    run("s03_train_all.py", "--models", "M3", "--seeds", "42", "43", *[o for o in ov if "input_size" not in o], env=env)
    run("s04_predict_all.py", "--models", *models, "--seeds", "42", "43", "--batch-size", "8", "--num-workers", "0", env=env)
    run("s05_ensemble_hybrid.py", env=env)
    run("s06_cross_validate_full.py", "--models", "M1", "M4", "--folds", "2", *ov, env=env)
    for st in ("s07_metrics.py", "s08_calibration.py", "s09_brightness_baseline.py", "s10_source_probe.py"):
        run(st, env=env)
    run("s11_ita.py", "--min-n", "4", env=env)
    run("s12_gradcam_quant.py", "--models", "M4", "--seeds", "42", "--batch-size", "4", "--max-images", "12", env=env)
    run("s13_stats.py", env=env)
    run("s14_figures.py", env=env)
    # data-integrity and robustness experiments (stages 15-23)
    run("s15_label_audit.py", env=env)
    run("s16_reanalysis.py", env=env)
    small = ["--batch-size", "8", "--num-workers", "0"]
    run("s17_nested_cv.py", "--models", "M1", "M4", "--folds", "2", *small, *ov, env=env)
    run("s18_matched_finetune.py", "--backbones", "efficientnet_b2", "convnext_tiny", "--seeds", "42", *small, *ov, env=env)
    run("s19_strict_retrain.py", "--models", "M4", "--seeds", "42", *small, *ov, env=env)
    run("s20_leaky_split.py", "--models", "M4", "--seeds", "42", *small, *ov, env=env)
    run("s21_foundation_probe.py", "--random-init", "--batch-size", "8", "--num-workers", "0", env=env)
    run("s22_faithfulness.py", "--models", "M4", "--seeds", "42", "--n-images", "8", "--batch-size", "4", "--num-workers", "0", env=env)
    run("s23_integrity_tables.py", env=env)

    expected = ["tables/overall_performance.csv", "tables/external_results.csv", "tables/balancing_effect.csv", "tables/variants.csv",
                "tables/efficiency.csv", "tables/calibration.csv", "tables/selective_prediction.csv", "tables/brightness_baseline.csv",
                "tables/source_probe_M4.csv", "tables/accuracy_by_source.csv", "tables/ita_accuracy_by_stratum.csv", "tables/ita_tests.csv",
                "tables/gradcam_localisation.csv", "tables/stats_pairwise.csv", "tables/stats_cv.csv", "tables/stats_external.csv",
                "tables/cv_folds.csv", "figures/Fig_confusion_test_M4.eps", "figures/Fig_roc_pr_test_M4.tif", "figures/Fig_reliability_M4.pdf",
                "figures/Fig_ita_M4.pdf", "figures/Fig_external_recall_M4.pdf", "figures/Fig_class_distribution.pdf",
                "dataset_qc/strict_manifest.csv", "tables/strict_subset.csv", "tables/external_subsets.csv", "tables/aurc.csv",
                "tables/trainable_parameters.csv", "tables/cv_nested_summary.csv", "tables/matched_finetune.csv",
                "tables/strict_retrain.csv", "tables/leaky_split.csv", "tables/probe_frozen.csv", "tables/dinov2_neighbours.csv",
                "tables/border_mask.csv", "tables/deletion_test.csv"]
    missing = [e for e in expected if not (root / "results" / e).exists()]
    print("\nSMOKE TEST", "FAILED, missing: " + ", ".join(missing) if missing else f"PASSED ({root})")
    if not args.keep and not missing and args.root is None:
        shutil.rmtree(root)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
