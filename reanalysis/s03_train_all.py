"""Stage 03 - train every configuration with several seeds.

For each model id in ``common.MODEL_CONFIGS`` and each seed, the original
training configuration is copied into ``reanalysis/runs/<model>/seed<seed>/
config.yaml`` with only ``experiment.seed``, ``experiment.output_dir`` and
``data.dataset_dir`` changed, and ``scripts/train.py`` is executed on it.
Training recipes are therefore identical to the original repository.

A run is considered complete when ``done.json`` exists in its folder; runs
interrupted by a Colab session limit are restarted from scratch on the next
invocation (the trainer has no mid-training resume), so the stage is safe to
re-run until every ``done.json`` exists.

Usage:
    python reanalysis/s03_train_all.py                     # 9 models x 3 seeds
    python reanalysis/s03_train_all.py --models M4 M1 --seeds 42
    python reanalysis/s03_train_all.py --override training.epochs=1 data.input_size=[64,64]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, MODEL_CONFIGS, PROJECT, SEEDS, checkpoint_path, ensure_dir, log, run_dir, sh, write_json  # noqa: E402


def parse_override(s: str):
    key, val = s.split("=", 1)
    try:
        val = json.loads(val)
    except json.JSONDecodeError:
        pass
    return key, val


def set_nested(cfg: dict, key: str, val) -> None:
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = val


def gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return platform.processor() or "cpu"


def train_one(model_id: str, seed: int, overrides: list[str], dataset_dir: Path) -> None:
    rd = run_dir(model_id, seed)
    done = rd / "done.json"
    if done.exists():
        log(f"{model_id} seed {seed}: done, skipping")
        return
    if rd.exists():
        log(f"{model_id} seed {seed}: incomplete run found, restarting from scratch")
        shutil.rmtree(rd)
    ensure_dir(rd)

    with open(PROJECT / MODEL_CONFIGS[model_id]) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["output_dir"] = str(rd)
    cfg["experiment"]["name"] = f"{model_id}_seed{seed}"
    cfg["data"]["dataset_dir"] = str(dataset_dir)
    for o in overrides:
        k, v = parse_override(o)
        set_nested(cfg, k, v)
    cfg_path = rd / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    t0 = time.time()
    sh([sys.executable, "scripts/train.py", "--config", str(cfg_path)])
    elapsed = time.time() - t0

    import torch
    ckpt = torch.load(checkpoint_path(model_id, seed), map_location="cpu", weights_only=False)
    write_json({
        "model_id": model_id, "seed": seed, "config": str(MODEL_CONFIGS[model_id]),
        "overrides": overrides, "train_time_s": elapsed, "device": gpu_name(),
        "best_epoch": ckpt.get("epoch"), "best_metrics": ckpt.get("metrics"),
        "torch": torch.__version__, "python": platform.python_version(),
    }, done)
    log(f"{model_id} seed {seed}: finished in {elapsed/60:.1f} min, best epoch {ckpt.get('epoch')}, "
        f"val metrics {ckpt.get('metrics')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--override", nargs="*", default=[], help="key.path=value (JSON values allowed)")
    ap.add_argument("--dataset-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    for seed in args.seeds:            # seed-major so that one full seed finishes first
        for mid in args.models:
            train_one(mid, seed, args.override, args.dataset_dir)
    log("stage 03 done")


if __name__ == "__main__":
    main()
