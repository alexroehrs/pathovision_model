"""Stage 21 - frozen-feature baselines and model-independent duplicate audit (GPU).

Extracts frozen, task-agnostic image embeddings for the train, val, test and
external images with

* ``dinov2_vitb14`` - self-supervised DINOv2 ViT-B/14 (timm
  ``vit_base_patch14_dinov2.lvd142m``), 224 x 224 input;
* ``convnext_tiny_in1k`` - the ImageNet ConvNeXt-Tiny used to initialise M4,
  224 x 224 input, no fine-tuning.

The CPU part (stage 23) fits a multinomial logistic regression on the training
embeddings (regularisation chosen on the validation split) and evaluates it
internally and externally, probes image origin, and uses the DINOv2 cosine
similarities - which do not depend on any model trained here - to audit
near-duplicates between test/external and training images.

Outputs: ``runs/probe/<name>/emb_<split>.npz`` (X, path, label).

    python reanalysis/s21_foundation_probe.py [--encoders dinov2_vitb14 convnext_tiny_in1k] [--batch-size 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, DATA_DIR, EXTERNAL_DIR, RUNS, ensure_dir, log  # noqa: E402
from infer import PerturbedDataset, get_device, relative_paths  # noqa: E402

ENCODERS = {
    "dinov2_vitb14": dict(timm_name="vit_base_patch14_dinov2.lvd142m", kwargs={"img_size": 224}),
    "convnext_tiny_in1k": dict(timm_name="convnext_tiny", kwargs={}),
}


def build_encoder(name: str, pretrained: bool = True):
    import timm
    spec = ENCODERS[name]
    model = timm.create_model(spec["timm_name"], pretrained=pretrained, num_classes=0, **spec["kwargs"])
    return model.eval()


def transform(size: int = 224):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([A.Resize(height=size, width=size), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                      ToTensorV2()])


@torch.no_grad()
def extract(model, root: Path, base: Path, device, batch_size: int, num_workers: int, size: int):
    import pandas as pd
    ds = PerturbedDataset(root, transform(size), CLASSES)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                         pin_memory=device.type == "cuda")
    feats, paths, labels = [], [], []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            f = model(x)
        feats.append(f.float().cpu().numpy()); paths += list(b["path"]); labels += list(b["class_name"])
    rel = relative_paths(pd.DataFrame({"path": paths}), base).to_numpy()
    return np.concatenate(feats), rel, np.asarray(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--random-init", action="store_true", help="smoke test only: no weight download")
    args = ap.parse_args()
    device = get_device()
    splits = [("train", DATA_DIR / "train", DATA_DIR), ("val", DATA_DIR / "val", DATA_DIR), ("test", DATA_DIR / "test", DATA_DIR)]
    if (EXTERNAL_DIR / "Acne").exists():
        splits.append(("external", EXTERNAL_DIR, EXTERNAL_DIR))
    for name in args.encoders:
        out_dir = ensure_dir(RUNS / "probe" / name)
        todo = [s for s in splits if not (out_dir / f"emb_{s[0]}.npz").exists()]
        if not todo:
            continue
        model = build_encoder(name, pretrained=not args.random_init).to(device)
        for split, root, base in todo:
            X, rel, y = extract(model, root, base, device, args.batch_size, args.num_workers, args.size)
            np.savez_compressed(out_dir / f"emb_{split}.npz", X=X.astype(np.float16), path=rel, label=y)
            log(f"{name} {split}: {X.shape}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log("stage 21 done")


if __name__ == "__main__":
    main()
