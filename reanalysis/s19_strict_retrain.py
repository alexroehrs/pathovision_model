"""Stage 19 - retraining on the strict-label subset (GPU).

Stage 15 flags images whose file name names a different entity than the
folder label (e.g. lichen planus inside Psoriasis, nail dystrophies inside
Nail fungus), histology/laboratory images, and images stored under two target
classes.  This stage keeps only the strict-label images (original split
assignment preserved), retrains M4 and M1 with their unchanged recipes (three
seeds) and predicts the strict test subset, the full test split and the
external set.

Outputs: ``runs/strict/<model>/seed<s>/pred_{strict_test,test,external}.csv``.
Tables are written by stage 23.

    python reanalysis/s19_strict_retrain.py [--models M4 M1] [--seeds ...] [--override ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, EXTERNAL_DIR, MODEL_CONFIGS, REANALYSIS, RESULTS, RUNS, SEEDS, log  # noqa: E402
from extra import link_tree, predict_run, train_run  # noqa: E402

STRICT_DIR = REANALYSIS / "strict_data"


def strict_manifest() -> pd.DataFrame:
    for p in (RESULTS / "dataset_qc" / "strict_manifest.csv", REANALYSIS / "results" / "dataset_qc" / "strict_manifest.csv"):
        if p.exists():
            return pd.read_csv(p)
    sys.exit("strict_manifest.csv not found - run stage 15 first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M4", "M1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    man = strict_manifest()
    man["src"] = man["relative_path"].map(lambda r: str((DATA_DIR / r).resolve()))
    missing = [s for s in man["src"] if not Path(s).exists()]
    if missing:
        sys.exit(f"{len(missing)} strict-manifest images not found under {DATA_DIR} (e.g. {missing[0]})")
    root = link_tree(man, STRICT_DIR)
    log("strict subset: " + ", ".join(f"{k}={v}" for k, v in man["split"].value_counts().items()))
    ext = (EXTERNAL_DIR / "Acne").exists()
    for seed in args.seeds:
        for mid in args.models:
            rd = RUNS / "strict" / mid / f"seed{seed}"
            train_run(rd, MODEL_CONFIGS[mid], seed, root, args.override, name=f"strict_{mid}_seed{seed}",
                      extra_meta={"model_id": mid, "subset": "strict"})
            jobs = [("strict_test", root / "test", DATA_DIR), ("test", DATA_DIR / "test", DATA_DIR)]
            if ext:
                jobs.append(("external", EXTERNAL_DIR, EXTERNAL_DIR))
            predict_run(rd, jobs, batch_size=args.batch_size, num_workers=args.num_workers)
    log("stage 19 done")


if __name__ == "__main__":
    main()
