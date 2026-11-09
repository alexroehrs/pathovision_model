"""Helpers shared by the data-integrity and robustness experiments (stages 17-22).

* :func:`link_tree` - build a ``<root>/<split>/<class>/`` tree of symbolic links
  from a table of (source path, split, class), so that a subset or a new split
  of the images never copies a file;
* :func:`train_run` - train one configuration into an arbitrary run folder
  (same mechanics as stage 03: config copy + overrides + ``scripts/train.py``,
  ``done.json`` marker, restart from scratch if interrupted);
* :func:`predict_run` - write ``pred_<name>.csv`` for a trained run.
"""

from __future__ import annotations

import platform
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, PROJECT, ensure_dir, log, sh, write_json  # noqa: E402
from s03_train_all import gpu_name, parse_override, set_nested  # noqa: E402


def link_tree(items: pd.DataFrame, root: Path, splits=("train", "val", "test")) -> Path:
    """``items`` has columns ``src`` (absolute path), ``split``, ``class``.

    Existing trees are reused when their ``.tree.csv`` matches ``items``."""
    marker = root / ".tree.csv"
    items = items[["src", "split", "class"]].astype(str).sort_values(["split", "class", "src"]).reset_index(drop=True)
    if marker.exists():
        old = pd.read_csv(marker, dtype=str)
        if len(old) == len(items) and (old.values == items.values).all():
            return root
    if root.exists():
        shutil.rmtree(root)
    for sp in splits:
        for c in CLASSES:
            ensure_dir(root / sp / c)
    for src_s, sp, cls in zip(items["src"], items["split"], items["class"]):
        src = Path(src_s)
        dst = root / sp / cls / src.name
        if dst.exists() or dst.is_symlink():
            dst = dst.with_name(f"{src.stem}__{abs(hash(str(src))) % 10**8}{src.suffix}")
        dst.symlink_to(src)
    ensure_dir(root)
    items.to_csv(marker, index=False)
    log(f"linked {len(items)} images into {root}")
    return root


def train_run(rd: Path, base_config: str | Path, seed: int, dataset_dir: Path, overrides=(), name: str | None = None,
              extra_meta: dict | None = None) -> Path:
    done = rd / "done.json"
    ckpt = rd / "checkpoints" / "best.pt"
    if done.exists() and ckpt.exists():
        log(f"{rd}: done, skipping")
        return ckpt
    if rd.exists():
        log(f"{rd}: incomplete run found, restarting from scratch")
        shutil.rmtree(rd)
    ensure_dir(rd)
    with open(PROJECT / base_config) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["output_dir"] = str(rd)
    cfg["experiment"]["name"] = name or rd.name
    cfg["data"]["dataset_dir"] = str(dataset_dir)
    for o in overrides:
        k, v = parse_override(o)
        set_nested(cfg, k, v)
    with open(rd / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    t0 = time.time()
    sh([sys.executable, "scripts/train.py", "--config", str(rd / "config.yaml")])
    elapsed = time.time() - t0
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = {"base_config": str(base_config), "seed": seed, "dataset_dir": str(dataset_dir), "overrides": list(overrides),
            "train_time_s": elapsed, "device": gpu_name(), "best_epoch": ck.get("epoch"), "best_metrics": ck.get("metrics"),
            "torch": torch.__version__, "python": platform.python_version()}
    meta.update(extra_meta or {})
    write_json(meta, done)
    log(f"{rd}: finished in {elapsed / 60:.1f} min, best epoch {ck.get('epoch')}")
    return ckpt


def predict_run(rd: Path, jobs, device=None, batch_size: int = 32, num_workers: int = 2, force: bool = False) -> None:
    """``jobs`` = iterable of (name, image_root, base_for_relative_paths[, perturb])."""
    from infer import get_device, load_checkpoint, predict_folder, relative_paths
    device = device or get_device()
    todo = [j for j in jobs if force or not (rd / f"pred_{j[0]}.csv").exists()]
    if not todo:
        return
    model, config = load_checkpoint(rd / "checkpoints" / "best.pt", device)
    for job in todo:
        name, root, base = job[:3]
        perturb = job[3] if len(job) > 3 else None
        df = predict_folder(model, config, Path(root), perturb=perturb, batch_size=batch_size, num_workers=num_workers, device=device)
        df.insert(1, "relative_path", _rel(df, Path(root), Path(base), relative_paths))
        df.to_csv(rd / f"pred_{name}.csv", index=False)
        log(f"{rd} {name}: n={len(df)} acc={100 * (df['class'] == df['pred']).mean():.2f}%")
    del model
    try:
        import torch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _rel(df, root: Path, base: Path, relative_paths):
    """Relative path w.r.t. ``base``; for symlink trees the link target is used."""
    try:
        return relative_paths(df, base)
    except ValueError:
        return df["path"].map(lambda p: str(Path(p).relative_to(root)))
