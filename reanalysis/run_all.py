"""Run the re-execution pipeline stage by stage.

    python reanalysis/run_all.py                 # everything, in order
    python reanalysis/run_all.py --stages 03 04  # selected stages
    python reanalysis/run_all.py --from 07       # CPU-only tail from existing CSVs
    python reanalysis/run_all.py --smoke         # tiny synthetic end-to-end test

Stages 01-04, 06 and 12 need the images (and a GPU for reasonable run times);
05, 07-11, 13 and 14 work from the CSV/NPZ outputs and run on any CPU
(stages 09 and 11 also read the images, once, to compute colour statistics).
Every stage is idempotent: completed work is detected and skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = {
    "01": "s01_build_internal.py", "02": "s02_build_external.py", "03": "s03_train_all.py",
    "04": "s04_predict_all.py", "05": "s05_ensemble_hybrid.py", "06": "s06_cross_validate_full.py",
    "07": "s07_metrics.py", "08": "s08_calibration.py", "09": "s09_brightness_baseline.py",
    "10": "s10_source_probe.py", "11": "s11_ita.py", "12": "s12_gradcam_quant.py",
    "13": "s13_stats.py", "14": "s14_figures.py",
}
IMAGE_STAGES = {"01", "02", "03", "04", "06", "09", "11", "12"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", choices=list(STAGES))
    ap.add_argument("--from", dest="from_", choices=list(STAGES))
    ap.add_argument("--skip", nargs="*", default=[], choices=list(STAGES))
    ap.add_argument("--extra", nargs="*", default=[], help="extra CLI arguments passed to every selected stage")
    ap.add_argument("--smoke", action="store_true", help="run tests/smoke_test.py instead")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(subprocess.call([sys.executable, str(HERE / "tests" / "smoke_test.py")]))
    stages = args.stages or [s for s in STAGES if not args.from_ or s >= args.from_]
    stages = [s for s in stages if s not in args.skip]
    t_all = time.time()
    for s in stages:
        print(f"\n{'=' * 78}\n  STAGE {s}: {STAGES[s]}\n{'=' * 78}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, str(HERE / STAGES[s]), *args.extra])
        print(f"  stage {s} finished with code {rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            sys.exit(f"stage {s} failed; fix the problem and rerun `python reanalysis/run_all.py --from {s}`")
    print(f"\nall stages done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
