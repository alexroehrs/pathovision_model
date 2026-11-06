"""Stage 04 - per-image predictions for every trained run.

For every ``(model, seed)`` with a checkpoint, writes into its run folder:

* ``pred_val.csv``, ``pred_test.csv``, ``pred_external.csv`` - path, true
  class, predicted class, confidence and the 7 class probabilities;
* ``pred_test_brightnorm.csv`` - test set re-evaluated after per-image CIELAB
  L* mean normalisation (brightness-cue sensitivity analysis);
* ``pred_test_centercrop.csv`` / ``pred_external_centercrop.csv`` - central
  70 % crop before resizing (border watermark/caption sensitivity analysis);
* ``emb_<split>.npz`` - penultimate-layer embeddings for train/val/test/
  external (used by the hybrid classifier and the source-probe analysis);
  only for models listed in ``--embed-models`` (default: M4, the primary);
* ``efficiency.json`` - parameters, GFLOPs (fvcore) and measured latency.

Usage:
    python reanalysis/s04_predict_all.py [--models ...] [--seeds ...] [--batch-size 32]
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, DATA_DIR, EXTERNAL_DIR, HYBRID_BASE, MODEL_CONFIGS, SEEDS, available_runs,  # noqa: E402
                    checkpoint_path, log, pred_path, run_dir, write_json)
from infer import (brightness_normalise, central_crop, get_device,  # noqa: E402
                   load_checkpoint, predict_folder, relative_paths)


def efficiency(model, config, device) -> dict:
    h, w = config["data"]["input_size"]
    x = torch.randn(1, 3, h, w, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    gflops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False); fca.uncalled_modules_warnings(False)
        gflops = fca.total() / 1e9
    except Exception as exc:  # pragma: no cover
        log(f"FLOP count failed: {exc}")
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 30
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = 1000 * (time.perf_counter() - t0) / n
    return {"params_M": n_params / 1e6, "gflops": gflops, "latency_bs1_ms": lat,
            "input_size": [h, w], "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--embed-models", nargs="+", default=[HYBRID_BASE])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--skip-variants", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = get_device()
    ext_available = (EXTERNAL_DIR / "Acne").exists()
    # target brightness: mean L* of the disease classes of the training split
    target_l = None
    if not args.skip_variants:
        target_l = dataset_mean_l_star_disease(DATA_DIR / "train")
        log(f"brightness-normalisation target mean L* = {target_l:.2f}")

    for mid, seed in available_runs(args.models, args.seeds):
        rd = run_dir(mid, seed)
        model, config = load_checkpoint(checkpoint_path(mid, seed), device)
        kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, device=device)
        want_emb = mid in args.embed_models

        jobs = [("val", DATA_DIR / "val", None), ("test", DATA_DIR / "test", None)]
        if ext_available:
            jobs.append(("external", EXTERNAL_DIR, None))
        if not args.skip_variants:
            jobs += [("test_brightnorm", DATA_DIR / "test", partial(brightness_normalise, target_mean_l=target_l)),
                     ("test_centercrop", DATA_DIR / "test", central_crop)]
            if ext_available:
                jobs.append(("external_centercrop", EXTERNAL_DIR, central_crop))
        if want_emb:
            jobs.append(("train", DATA_DIR / "train", None))

        for split, root, perturb in jobs:
            out = pred_path(mid, seed, split)
            emb_out = rd / f"emb_{split}.npz"
            need_emb = want_emb and perturb is None and not emb_out.exists()
            if out.exists() and not args.force and not need_emb:
                continue
            t0 = time.time()
            res = predict_folder(model, config, root, perturb=perturb, with_features=need_emb, **kw)
            if need_emb:
                df, feats = res
                base = DATA_DIR if root.is_relative_to(DATA_DIR) else EXTERNAL_DIR
                np.savez_compressed(emb_out, X=feats.astype(np.float32),
                                    path=relative_paths(df, base).to_numpy(), label=df["class"].to_numpy())
            else:
                df = res
            base = DATA_DIR if root.is_relative_to(DATA_DIR) else EXTERNAL_DIR
            df.insert(1, "relative_path", relative_paths(df, base))
            df.to_csv(out, index=False)
            acc = 100 * (df["class"] == df["pred"]).mean()
            log(f"{mid} seed {seed} {split}: n={len(df)} acc={acc:.2f}% "
                f"({time.time()-t0:.0f}s, {df.attrs['latency_ms']:.1f} ms/img batched)")

        eff_path = rd / "efficiency.json"
        if not eff_path.exists() or args.force:
            write_json(efficiency(model, config, device), eff_path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log("stage 04 done")


def dataset_mean_l_star_disease(train_root: Path) -> float:
    """Mean L* over the six disease classes of the training split."""
    from infer import VALID_EXTENSIONS, mean_l_star
    import cv2
    vals = []
    for cls in [c for c in CLASSES if c != "Normal"]:
        files = sorted(f for f in (train_root / cls).iterdir() if f.suffix.lower() in VALID_EXTENSIONS)[:300]
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                vals.append(mean_l_star(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return float(np.mean(vals))


if __name__ == "__main__":
    main()
