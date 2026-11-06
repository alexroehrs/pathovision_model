"""Stage 05 - soft-voting ensemble (M6) and hybrid classifier (H1).

* **M6**: per seed, the class probabilities of M2, M3 and M4 are averaged
  (soft voting, as in the original ``src/hybrid/ensemble.py``) on every
  split/variant for which all three members have predictions.
* **H1**: per seed, a random forest (500 trees, same hyper-parameters as
  ``src/hybrid/classifiers.py``) is fitted on the standardised penultimate
  embeddings of the primary model (M4) extracted from the training split, and
  applied to the val/test/external embeddings written by stage 04.

Outputs follow the run-folder layout (``runs/M6/seed<s>/pred_<split>.csv``,
``runs/H1/seed<s>/pred_<split>.csv``) so downstream stages treat them like
any other model.  Needs only CSV/NPZ inputs, so it runs on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, ENSEMBLE_MEMBERS, HYBRID_BASE, SEEDS, SPLITS, VARIANTS, ensure_dir, log,  # noqa: E402
                    pred_path, proba_matrix, run_dir, write_json)


def finish(df: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    out = df[["path", "relative_path", "class"]].copy()
    out["pred"] = [CLASSES[i] for i in P.argmax(1)]
    out["confidence"] = P.max(1)
    for j, c in enumerate(CLASSES):
        out[f"proba_{c}"] = P[:, j]
    return out


def ensemble(seed: int) -> None:
    for split in SPLITS + VARIANTS:
        paths = [pred_path(m, seed, split) for m in ENSEMBLE_MEMBERS]
        if not all(p.exists() for p in paths):
            continue
        dfs = [pd.read_csv(p).sort_values("relative_path").reset_index(drop=True) for p in paths]
        assert all((d["relative_path"] == dfs[0]["relative_path"]).all() for d in dfs), "member row mismatch"
        P = np.mean([proba_matrix(d) for d in dfs], axis=0)
        out = finish(dfs[0], P)
        ensure_dir(run_dir("M6", seed))
        out.to_csv(pred_path("M6", seed, split), index=False)
        log(f"M6 seed {seed} {split}: acc={100*(out['class']==out['pred']).mean():.2f}%")


def hybrid(seed: int) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    rd = run_dir(HYBRID_BASE, seed)
    train_npz = rd / "emb_train.npz"
    if not train_npz.exists():
        log(f"H1 seed {seed}: no training embeddings for {HYBRID_BASE}, skipping")
        return
    tr = np.load(train_npz, allow_pickle=True)
    scaler = StandardScaler().fit(tr["X"])
    rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2,
                                class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(scaler.transform(tr["X"]), tr["label"])
    ensure_dir(run_dir("H1", seed))
    for split in SPLITS:
        npz = rd / f"emb_{split}.npz"
        base_pred = pred_path(HYBRID_BASE, seed, split)
        if not (npz.exists() and base_pred.exists()):
            continue
        e = np.load(npz, allow_pickle=True)
        P_raw = rf.predict_proba(scaler.transform(e["X"]))
        P = np.zeros((len(P_raw), len(CLASSES)))
        for j, c in enumerate(rf.classes_):
            P[:, CLASSES.index(c)] = P_raw[:, j]
        base = pd.read_csv(base_pred).set_index("relative_path").loc[e["path"]].reset_index()
        out = finish(base, P)
        out.to_csv(pred_path("H1", seed, split), index=False)
        log(f"H1 seed {seed} {split}: acc={100*(out['class']==out['pred']).mean():.2f}%")
    write_json({"base_model": HYBRID_BASE, "classifier": "RandomForest(n_estimators=500, class_weight=balanced)",
                "n_train": int(len(tr["label"])), "embedding_dim": int(tr["X"].shape[1])},
               run_dir("H1", seed) / "done.json")


def main() -> None:
    for seed in SEEDS:
        ensemble(seed)
        hybrid(seed)
    log("stage 05 done")


if __name__ == "__main__":
    main()
