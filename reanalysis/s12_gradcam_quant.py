"""Stage 12 - quantitative Grad-CAM localisation (needs checkpoints; GPU helps).

For the primary model (every seed) on the internal test split and the external
set, computes a Grad-CAM heat-map for the predicted class
(``pytorch-grad-cam``, same target layer selection as the original
``src/evaluation/gradcam.py``) and summarises where the activation mass lies:

* ``central_mass``  - fraction of activation inside the central 50 % box;
* ``border_mass``   - fraction inside a 10 % frame along the image border
  (where watermarks and captions occur);
* ``bottom_mass``   - fraction inside the bottom 15 % band;
* ``entropy``       - normalised spatial entropy (1 = diffuse, 0 = one pixel);
* ``peak_in_border`` - whether the activation maximum lies in the border frame.

These are reported per class and for correct vs incorrect predictions, with a
uniform-map reference (central box = 25 %, border frame = 36 %, bottom band =
15 %).  Outputs ``results/gradcam/<split>_<model>_seed<s>.csv`` and the
``gradcam_localisation`` table.

Usage:
    python reanalysis/s12_gradcam_quant.py [--models M4] [--seeds 42 43 44] [--max-images 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, CLASS_LABELS, DATA_DIR, EXTERNAL_DIR, PRIMARY, RESULTS, SEEDS, available_runs, checkpoint_path, ensure_dir, log, mean_sd, save_table  # noqa: E402
from infer import PerturbedDataset, get_device, load_checkpoint, relative_paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.transforms import get_val_transforms  # noqa: E402
from src.evaluation.gradcam import get_target_layer  # noqa: E402


def summarise_cam(cam: np.ndarray) -> dict:
    cam = np.clip(cam, 0, None).astype(np.float64)
    tot = cam.sum()
    if tot <= 0:
        return {"central_mass": np.nan, "border_mass": np.nan, "bottom_mass": np.nan, "entropy": np.nan, "peak_in_border": np.nan}
    p = cam / tot
    h, w = p.shape
    ch, cw = int(h * .25), int(w * .25)
    central = p[ch:h - ch, cw:w - cw].sum()
    bh, bw = int(h * .10), int(w * .10)
    inner = p[bh:h - bh, bw:w - bw].sum()
    bottom = p[int(h * .85):, :].sum()
    ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(p.size)
    py, px = np.unravel_index(p.argmax(), p.shape)
    peak_border = not (bh <= py < h - bh and bw <= px < w - bw)
    return {"central_mass": central, "border_mass": 1 - inner, "bottom_mass": bottom, "entropy": ent, "peak_in_border": float(peak_border)}


def run(mid, seed, split, root, device, batch_size, max_images):
    out = ensure_dir(RESULTS / "gradcam") / f"{split}_{mid}_seed{seed}.csv"
    if out.exists():
        return pd.read_csv(out)
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    model, config = load_checkpoint(checkpoint_path(mid, seed), device)
    ds = PerturbedDataset(root, get_val_transforms(config), CLASSES)
    if max_images and len(ds) > max_images:
        rng = np.random.default_rng(seed); keep = sorted(rng.choice(len(ds), max_images, replace=False))
        ds.samples = [ds.samples[i] for i in keep]
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    cam = GradCAM(model=model, target_layers=[get_target_layer(model)])
    rows = []
    for batch in loader:
        x = batch["image"].to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x).float(), 1)
        pred = probs.argmax(1).cpu().numpy()
        maps = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(k)) for k in pred])
        for i in range(len(pred)):
            r = summarise_cam(maps[i])
            r.update(path=batch["path"][i], **{"class": batch["class_name"][i]}, pred=CLASSES[int(pred[i])],
                     confidence=float(probs[i].max()), correct=int(CLASSES[int(pred[i])] == batch["class_name"][i]))
            rows.append(r)
    df = pd.DataFrame(rows)
    base = DATA_DIR if root.resolve().is_relative_to(DATA_DIR.resolve()) else EXTERNAL_DIR
    df.insert(1, "relative_path", relative_paths(df, base))
    df.to_csv(out, index=False)
    log(f"Grad-CAM {mid} seed {seed} {split}: {len(df)} images, central mass {df['central_mass'].mean():.3f}, border {df['border_mass'].mean():.3f}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[PRIMARY])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=0, help="subsample per split (0 = all)")
    args = ap.parse_args()
    device = get_device()
    frames = []
    for mid, seed in available_runs(args.models, args.seeds):
        for split, root in (("test", DATA_DIR / "test"), ("external", EXTERNAL_DIR)):
            if root.exists():
                frames.append(run(mid, seed, split, root, device, args.batch_size, args.max_images).assign(model=mid, seed=seed, split=split))
    if not frames:
        sys.exit("no checkpoints found")
    df = pd.concat(frames)
    metrics = ["central_mass", "border_mass", "bottom_mass", "entropy", "peak_in_border"]
    rows = []
    for (split, mid), g in df.groupby(["split", "model"]):
        for label, sub in [("all", g), ("correct", g[g["correct"] == 1]), ("incorrect", g[g["correct"] == 0])] + [(CLASS_LABELS[c], g[g["class"] == c]) for c in CLASSES]:
            if len(sub) == 0:
                continue
            per_seed = sub.groupby("seed")[metrics].mean()
            rows.append({"split": split, "model": mid, "subset": label, "n_per_seed": int(len(sub) / sub["seed"].nunique()),
                         **{m: mean_sd(per_seed[m]) for m in metrics}})
    rows.append({"split": "reference", "model": "uniform map", "subset": "-", "n_per_seed": 0, "central_mass": "0.25", "border_mass": "0.36",
                 "bottom_mass": "0.15", "entropy": "1.00", "peak_in_border": "-"})
    save_table(pd.DataFrame(rows), "gradcam_localisation")
    log("stage 12 done")


if __name__ == "__main__":
    main()
