"""Stage 18 - backbone comparison under a matched fine-tuning protocol (GPU).

The original configurations differ in more than the backbone: input size
(500, 384 or 224 px), number of trainable parameter tensors
(``freeze_except_last_n`` 10/20/30, i.e. about 1.6-15 M trainable backbone
parameters) and optimiser settings.  Here the four backbones are trained with
one shared protocol: the M4 recipe (``configs/convnext_tiny.yaml``: AdamW,
lr 5e-5, cosine warm-up, focal loss with inverse-frequency weights, balanced
sampling, 50 epochs), **full fine-tuning** (every parameter trainable) and a
**224 x 224** input for all, three seeds each.

Outputs: ``runs/matched/<backbone>/seed<s>/`` with predictions on val, test and
external.  Tables are written by stage 23.

    python reanalysis/s18_matched_finetune.py [--backbones ...] [--seeds 42 43 44] [--override ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, EXTERNAL_DIR, RUNS, SEEDS, log  # noqa: E402
from extra import predict_run, train_run  # noqa: E402

BASE = "configs/convnext_tiny.yaml"
BACKBONES = ["efficientnet_b2", "efficientnetv2_s", "swin_tiny", "convnext_tiny"]
MATCHED = ["model.freeze_strategy=\"none\"", "model.freeze_except_last_n=1000000", "data.input_size=[224,224]"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    ext = (EXTERNAL_DIR / "Acne").exists()
    for seed in args.seeds:
        for bb in args.backbones:
            rd = RUNS / "matched" / bb / f"seed{seed}"
            ov = [f"model.backbone=\"{bb}\"", *MATCHED, *args.override]
            train_run(rd, BASE, seed, DATA_DIR, ov, name=f"matched_{bb}_seed{seed}", extra_meta={"backbone": bb, "protocol": "matched"})
            jobs = [("val", DATA_DIR / "val", DATA_DIR), ("test", DATA_DIR / "test", DATA_DIR)]
            if ext:
                jobs.append(("external", EXTERNAL_DIR, EXTERNAL_DIR))
            predict_run(rd, jobs, batch_size=args.batch_size, num_workers=args.num_workers)
    log("stage 18 done")


if __name__ == "__main__":
    main()
