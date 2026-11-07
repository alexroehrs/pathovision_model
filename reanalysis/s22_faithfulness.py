"""Stage 22 - border masking and Grad-CAM faithfulness (GPU).

Two checks that complement the activation-mass summary of stage 12:

1. **Border masking without rescaling** - a frame of 10 % or 15 % of the image
   side is replaced by the ImageNet mean colour before resizing, so that the
   lesion keeps its scale (the central crop of stage 04 also zooms in).
   Writes ``pred_{test,external}_border{10,15}.csv`` into the run folders of
   the requested models.
2. **Deletion test** - for a fixed random sample of test and external images,
   the pixels ranked highest by the Grad-CAM map of the predicted class are
   progressively replaced by the mean colour (0 in normalised space) and the
   probability of the originally predicted class is recorded; the same is done
   with a random pixel order.  A faithful map makes the probability fall
   faster than random deletion.  Writes ``results/faithfulness/deletion_<split>_<model>_seed<s>.csv``.

    python reanalysis/s22_faithfulness.py [--models M4 M1] [--seeds ...] [--n-images 300]
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, DATA_DIR, EXTERNAL_DIR, PRIMARY, RESULTS, SEEDS, available_runs, checkpoint_path, ensure_dir, log, run_dir  # noqa: E402
from extra import predict_run  # noqa: E402
from infer import PerturbedDataset, get_device, load_checkpoint, relative_paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.transforms import get_val_transforms  # noqa: E402
from src.evaluation.gradcam import get_target_layer  # noqa: E402

MEAN_RGB = np.array([0.485, 0.456, 0.406]) * 255
FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def border_mask(image_rgb: np.ndarray, frac: float) -> np.ndarray:
    img = image_rgb.copy()
    h, w = img.shape[:2]
    dh, dw = int(round(h * frac)), int(round(w * frac))
    fill = MEAN_RGB.astype(img.dtype)
    img[:dh] = fill; img[h - dh:] = fill; img[:, :dw] = fill; img[:, w - dw:] = fill
    return img


@torch.no_grad()
def _prob(model, x, cls_idx):
    return torch.softmax(model(x).float(), 1).gather(1, cls_idx[:, None])[:, 0]


def deletion(mid, seed, split, root, device, n_images, batch_size):
    out = ensure_dir(RESULTS / "faithfulness") / f"deletion_{split}_{mid}_seed{seed}.csv"
    if out.exists():
        return
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    model, config = load_checkpoint(checkpoint_path(mid, seed), device)
    ds = PerturbedDataset(root, get_val_transforms(config), CLASSES)
    rng = np.random.default_rng(1234)                      # same images for every model and seed
    if len(ds) > n_images:
        keep = sorted(rng.choice(len(ds), n_images, replace=False))
        ds.samples = [ds.samples[i] for i in keep]
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    cam = GradCAM(model=model, target_layers=[get_target_layer(model)])
    rows = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for batch in loader:
        x = batch["image"].to(device)
        with torch.no_grad():
            pred = model(x).float().argmax(1)
        maps = torch.as_tensor(cam(input_tensor=x, targets=[ClassifierOutputTarget(int(k)) for k in pred]), device=device)
        b, _, h, w = x.shape
        flat = maps.reshape(b, -1)
        order_cam = flat.argsort(1, descending=True)
        order_rnd = torch.stack([torch.randperm(h * w, generator=g) for _ in range(b)]).to(device)
        for kind, order in (("gradcam", order_cam), ("random", order_rnd)):
            for f in FRACTIONS:
                k = int(round(f * h * w))
                mask = torch.ones(b, h * w, device=device)
                if k:
                    mask.scatter_(1, order[:, :k], 0.0)
                xm = x * mask.reshape(b, 1, h, w)                # 0 = mean colour in normalised space
                p = _prob(model, xm, pred).cpu().numpy()
                for i in range(b):
                    rows.append({"path": batch["path"][i], "class": batch["class_name"][i], "pred": CLASSES[int(pred[i])],
                                 "order": kind, "fraction": f, "prob_pred": float(p[i])})
    df = pd.DataFrame(rows)
    base = DATA_DIR if root.resolve().is_relative_to(DATA_DIR.resolve()) else EXTERNAL_DIR
    df.insert(1, "relative_path", relative_paths(df, base))
    df.to_csv(out, index=False)
    log(f"deletion {split} {mid} seed {seed}: {df.path.nunique()} images")
    del model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[PRIMARY, "M1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--deletion-models", nargs="+", default=[PRIMARY])
    args = ap.parse_args()
    device = get_device()
    ext = (EXTERNAL_DIR / "Acne").exists()
    for mid, seed in available_runs(args.models, args.seeds):
        jobs = []
        for frac in (0.10, 0.15):
            tag = f"border{int(frac * 100)}"
            jobs.append((f"test_{tag}", DATA_DIR / "test", DATA_DIR, partial(border_mask, frac=frac)))
            if ext:
                jobs.append((f"external_{tag}", EXTERNAL_DIR, EXTERNAL_DIR, partial(border_mask, frac=frac)))
        predict_run(run_dir(mid, seed), jobs, device=device, batch_size=32, num_workers=args.num_workers)
    for mid, seed in available_runs(args.deletion_models, args.seeds):
        deletion(mid, seed, "test", DATA_DIR / "test", device, args.n_images, args.batch_size)
        if ext:
            deletion(mid, seed, "external", EXTERNAL_DIR, device, args.n_images, args.batch_size)
    log("stage 22 done")


if __name__ == "__main__":
    main()
