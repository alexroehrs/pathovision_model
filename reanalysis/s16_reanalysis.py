"""Stage 16 - provenance-, label- and statistics-aware re-analysis (CPU).

Uses the outputs of stages 04-15 only (no images, no GPU):

* ``origin_accuracy``        - test accuracy by image origin (DermNet-derived vs
  HSD-only), by duplication status and by label-conflict status, per class;
* ``strict_subset``          - every model on the strict-label test subset of
  stage 15 (and on the test set without true label conflicts);
* ``external_subsets``       - external accuracy on the full set, without the
  near-duplicates of training images found with frozen DINOv2 features (stage
  21, cosine >= 0.95; skipped if stage 21 has not run), SD-198 only, the
  independent sources (SD-198 and skin_normal; the Acne, Skin Diseases and Nail
  Disease sources re-distribute DermNet images) and a protocol-matched
  independent subset that removes sub-diagnoses whose mapping contradicts the
  training folders (nail psoriasis, tinea versicolor, mucous-membrane
  psoriasis); each with the internal-to-external drop and an image-level
  bootstrap CI;
* ``stats_pairwise_holm``    - McNemar tests of stage 13 with Holm correction
  within each seed;
* ``aurc``                   - area under the risk-coverage curve (selective
  prediction on the softmax confidence) for test and external;
* ``color_baseline_ablation`` - colour-statistics random forest with and
  without image-size features, per-class recall;
* ``embedding_neighbours``   - nearest training neighbour (cosine, M4
  penultimate embeddings) of every test and external image and accuracy by
  similarity band;
* ``origin_probe``           - linear probe of image origin on M4 embeddings,
  within class, against the class-only baseline;
* ``trainable_parameters``   - total and trainable parameters per configuration
  (models built with random weights, no download).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_LABELS, CLASSES, EXTERNAL_DIR, MODEL_CONFIGS, PRIMARY, PROJECT, RESULTS, RUNS,  # noqa: E402
                    SEEDS, available_predictions, global_metrics, log, mean_sd, save_table)

MODELS = list(MODEL_CONFIGS) + ["M6", "H1"]
KEY_MODELS = ["M1", "M4", "M6", "H1"]
PAIRS = [(PRIMARY, "M1"), (PRIMARY, "M4b"), ("M6", PRIMARY), ("H1", PRIMARY), ("M1b", "M1"), ("M5", "M1"), ("M2", "M2b"), ("M3", "M3b")]
EXT_SOURCES = ["acne_dataset", "nail_disease", "sd198", "skin_10classes", "skin_normal"]
DERMNET_OVERLAP_SOURCES = {"acne_dataset", "skin_10classes", "nail_disease"}   # documented re-distribution of DermNet images
NEARDUP_COS = 0.95
PROTOCOL_MISMATCH = {"nail_psoriasis", "tinea_versicolor", "mucous_membrane_psoriasis"}
N_BOOT = 2000


# --------------------------------------------------------------------------
def preds(split, models=None):
    out = {}
    for mid, s, df in available_predictions(split, models or MODELS):
        out[(mid, s)] = df
    return out


def acc(df):
    return float((df["class"] == df["pred"]).mean() * 100) if len(df) else float("nan")


def summarise(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def external_meta(ext_df: pd.DataFrame) -> pd.DataFrame:
    """source (from the file-name prefix written by the external builder) and SD-198 sub-label."""
    meta = pd.DataFrame({"relative_path": ext_df["relative_path"].unique()})
    fname = meta["relative_path"].str.split("/").str[-1]
    meta["source"] = fname.str.extract(r"^(" + "|".join(EXT_SOURCES) + r")_")[0].fillna("unknown")
    man_p = EXTERNAL_DIR / "external_manifest.csv"
    if man_p.exists():
        man = pd.read_csv(man_p).drop_duplicates("path")[["path", "source_label"]]
        meta = meta.merge(man, left_on="relative_path", right_on="path", how="left").drop(columns=["path"])
    else:
        meta["source_label"] = np.nan
    meta["source_label"] = meta["source_label"].fillna("unknown")
    return meta


def neardup_paths(split: str = "external") -> set | None:
    """Relative paths whose nearest training/validation image (frozen DINOv2, stage 21/23) has cosine >= NEARDUP_COS."""
    sims_p = RESULTS / "dataset_qc" / f"dinov2_maxcos_{split}.npy"
    emb_p = RUNS / "probe" / "dinov2_vitb14" / f"emb_{split}.npz"
    if not (sims_p.exists() and emb_p.exists()):
        return None
    sims = np.load(sims_p)
    paths = np.asarray(np.load(emb_p, allow_pickle=True)["path"]).astype(str)
    return set(paths[sims >= NEARDUP_COS])


def label_audit() -> pd.DataFrame:
    la = pd.read_csv(RESULTS / "dataset_qc" / "label_audit.csv")
    la["key"] = la["relative_path"]            # same "<split>/<class>/<file>" key as the prediction files
    return la


def with_audit(df: pd.DataFrame, la: pd.DataFrame) -> pd.DataFrame:
    cols = ["key", "origin", "in_both_packages", "label_conflict", "true_conflict", "strict", "target_status"]
    d = df.copy()
    d["key"] = d["relative_path"]
    return d.merge(la[cols], on="key", how="left")


# --------------------------------------------------------------------------
def origin_accuracy(test_p, la):
    rows = []
    for mid in KEY_MODELS:
        runs = [with_audit(df, la) for (m, s), df in test_p.items() if m == mid]
        if not runs:
            continue
        groups = {"all": lambda d: d, "origin=DermNet": lambda d: d[d.origin == "dermnet"],
                  "origin=HSD-only": lambda d: d[d.origin == "hsd_only"],
                  "in both packages": lambda d: d[d.in_both_packages == True],  # noqa: E712
                  "label conflict": lambda d: d[d.label_conflict == True],  # noqa: E712
                  "no label conflict": lambda d: d[d.label_conflict == False],  # noqa: E712
                  "file name off-target": lambda d: d[d.target_status.isin(["off", "nonclinical"])],
                  "strict subset": lambda d: d[d.strict == True]}  # noqa: E712
        for cls in [None] + CLASSES:
            for g, f in groups.items():
                subs = [f(d if cls is None else d[d["class"] == cls]) for d in runs]
                n = len(subs[0])
                if n == 0:
                    continue
                m, sd = summarise([acc(x) for x in subs])
                rows.append({"model": mid, "class": "All" if cls is None else CLASS_LABELS[cls], "group": g, "n": n,
                             "accuracy_mean": m, "accuracy_sd": sd})
    out = pd.DataFrame(rows)
    save_table(out, "origin_accuracy")
    return out


def strict_subset(test_p, la):
    rows = []
    for mid in MODELS:
        runs = [with_audit(df, la) for (m, s), df in test_p.items() if m == mid]
        if not runs:
            continue
        for name, f in (("full test", lambda d: d), ("without true conflicts", lambda d: d[d.true_conflict == False]),  # noqa: E712
                        ("strict-label subset", lambda d: d[d.strict == True])):  # noqa: E712
            gm = [global_metrics(f(d)["class"].to_numpy(), f(d)["pred"].to_numpy()) for d in runs]
            r = {"model": mid, "subset": name, "n": gm[0]["n"]}
            for k in ("accuracy", "balanced_accuracy", "f1_macro"):
                r[k] = mean_sd([g[k] for g in gm])
                r[k + "_mean"] = float(np.mean([g[k] for g in gm]))
            rows.append(r)
    out = pd.DataFrame(rows)
    save_table(out, "strict_subset")
    return out


def _boot_drop(test_runs, ext_runs, rng):
    """Image-level bootstrap of (internal - external) accuracy, averaged over seeds.

    Images are resampled once per replicate and the same resample is applied to
    every seed, so seed variation and image sampling are both reflected."""
    ci = np.stack([(d["class"] == d["pred"]).to_numpy() for d in test_runs]).mean(0)
    ce = np.stack([(d["class"] == d["pred"]).to_numpy() for d in ext_runs]).mean(0)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        b[i] = ci[rng.integers(0, len(ci), len(ci))].mean() - ce[rng.integers(0, len(ce), len(ce))].mean()
    return 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def external_subsets(test_p, ext_p):
    any_ext = next(iter(ext_p.values()))
    meta = external_meta(any_ext)
    nd = neardup_paths("external")
    subsets = {"all sources": lambda d: d}
    if nd is not None:
        subsets["without DINOv2 near-duplicates of training images"] = lambda d: d[~d.relative_path.isin(nd)]
    subsets.update({
        "SD-198 only": lambda d: d[d.source == "sd198"],
        "independent sources (SD-198, skin_normal)": lambda d: d[~d.source.isin(DERMNET_OVERLAP_SOURCES)],
        "independent, protocol-matched":
            lambda d: d[~d.source.isin(DERMNET_OVERLAP_SOURCES) & ~d.source_label.isin(PROTOCOL_MISMATCH)],
    })
    rows = []
    rng = np.random.default_rng(0)
    for mid in KEY_MODELS:
        t_runs = [df for (m, s), df in sorted(test_p.items()) if m == mid]
        e_runs = [df.merge(meta, on="relative_path", how="left") for (m, s), df in sorted(ext_p.items()) if m == mid]
        if not t_runs or not e_runs:
            continue
        for name, f in subsets.items():
            subs = [f(d).sort_values("relative_path") for d in e_runs]
            classes = sorted(subs[0]["class"].unique())
            # internal reference restricted to the classes present in the subset
            t_sub = [d[d["class"].isin(classes)].sort_values("relative_path") for d in t_runs]
            gm = [global_metrics(x["class"].to_numpy(), x["pred"].to_numpy()) for x in subs]
            ia = [acc(x) for x in t_sub]
            ea = [g["accuracy"] for g in gm]
            lo, hi = _boot_drop(t_sub, subs, rng)
            rows.append({"model": mid, "subset": name, "n_external": len(subs[0]), "classes": len(classes),
                         "internal_acc": mean_sd(ia), "external_acc": mean_sd(ea),
                         "external_bacc": mean_sd([g["balanced_accuracy"] for g in gm]),
                         "external_f1_macro": mean_sd([g["f1_macro"] for g in gm]),
                         "drop_pp": mean_sd(np.array(ia) - np.array(ea)), "drop_ci_lo": lo, "drop_ci_hi": hi,
                         "external_acc_mean": float(np.mean(ea)), "internal_acc_mean": float(np.mean(ia))})
    out = pd.DataFrame(rows)
    save_table(out, "external_subsets")
    # per sub-label accuracy of the primary model (SD-198)
    e_runs = [df.merge(meta, on="relative_path", how="left") for (m, s), df in sorted(ext_p.items()) if m == PRIMARY]
    sl = []
    for lab in sorted(meta.loc[meta.source == "sd198", "source_label"].unique()):
        subs = [d[(d.source == "sd198") & (d.source_label == lab)] for d in e_runs]
        m, sd = summarise([acc(x) for x in subs])
        top = subs[0]["pred"].value_counts().idxmax() if len(subs[0]) else ""
        sl.append({"source_label": lab, "mapped_class": subs[0]["class"].iloc[0] if len(subs[0]) else "", "n": len(subs[0]),
                   "accuracy_mean": m, "accuracy_sd": sd, "most_frequent_prediction_seed42": top})
    save_table(pd.DataFrame(sl), "external_sd198_sublabels_M4")
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def mcnemar_exact(a, b):
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b)); n = n01 + n10
    return min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)) if n else 1.0


def pairwise_holm(test_p):
    rows = []
    per_seed = {}
    for s in SEEDS:
        ps = []
        for a, b in PAIRS:
            if (a, s) not in test_p or (b, s) not in test_p:
                ps.append(np.nan); continue
            da = test_p[(a, s)].sort_values("relative_path"); db = test_p[(b, s)].sort_values("relative_path")
            ps.append(mcnemar_exact((da["class"] == da["pred"]).to_numpy(), (db["class"] == db["pred"]).to_numpy()))
        ps = np.array(ps)
        ok = np.isfinite(ps)
        adj = np.full_like(ps, np.nan); adj[ok] = holm(ps[ok])
        per_seed[s] = (ps, adj)
    for i, (a, b) in enumerate(PAIRS):
        raw = [per_seed[s][0][i] for s in SEEDS]; adj = [per_seed[s][1][i] for s in SEEDS]
        rows.append({"model_a": a, "model_b": b, "p_raw_per_seed": ";".join(f"{x:.4f}" for x in raw),
                     "p_holm_per_seed": ";".join(f"{x:.4f}" for x in adj),
                     "seeds_significant_holm_0.05": int(np.sum(np.array(adj) < 0.05))})
    out = pd.DataFrame(rows)
    save_table(out, "stats_pairwise_holm")
    return out


def aurc_of(df):
    correct = (df["class"] == df["pred"]).to_numpy()
    order = np.argsort(-df["confidence"].to_numpy(), kind="stable")
    risk = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    aurc = risk.mean()
    r = 1 - correct.mean()
    optimal = np.mean(np.concatenate([np.zeros(int(correct.sum())), np.cumsum(np.ones(len(correct) - int(correct.sum())))
                                      / (np.arange(int(correct.sum()) + 1, len(correct) + 1))])) if len(correct) else np.nan
    return 1000 * aurc, 1000 * (aurc - optimal), 100 * r


def aurc(test_p, ext_p):
    rows = []
    for split, P in (("test", test_p), ("external", ext_p)):
        for mid in MODELS:
            vals = [aurc_of(df) for (m, s), df in P.items() if m == mid]
            if not vals:
                continue
            rows.append({"model": mid, "split": split, "aurc_x1000": mean_sd([v[0] for v in vals]),
                         "excess_aurc_x1000": mean_sd([v[1] for v in vals]), "error_rate": mean_sd([v[2] for v in vals])})
    out = pd.DataFrame(rows)
    save_table(out, "aurc")
    return out


def color_baseline_ablation():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, recall_score
    st = RESULTS / "image_stats"
    if not (st / "train.csv").exists():
        log("image_stats missing - skip colour baseline ablation"); return None
    tr = pd.read_csv(st / "train.csv")
    colour = ["mean_L", "sd_L", "mean_a", "sd_a", "mean_b", "sd_b", "mean_sat"]
    size = ["height", "width", "aspect"]
    rows = []
    for fs_name, feats in (("colour + size", colour + size), ("colour only", colour), ("size only", size)):
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
        clf.fit(tr[feats].to_numpy(), tr["class"].to_numpy())
        for split in ("test", "external"):
            d = pd.read_csv(st / f"{split}.csv")
            yp = clf.predict(d[feats].to_numpy()); yt = d["class"].to_numpy()
            r = {"features": fs_name, "split": split, "n": len(d), "accuracy": 100 * np.mean(yp == yt),
                 "balanced_accuracy": 100 * balanced_accuracy_score(yt, yp)}
            rec = recall_score(yt, yp, labels=CLASSES, average=None, zero_division=0)
            for c, v in zip(CLASSES, rec):
                r[f"recall_{CLASS_LABELS[c]}"] = 100 * v
            rows.append(r)
    out = pd.DataFrame(rows)
    save_table(out, "color_baseline_ablation")
    return out


def _load_emb(seed, split):
    p = RUNS / PRIMARY / f"seed{seed}" / f"emb_{split}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, np.asarray(z["path"]).astype(str), np.asarray(z["label"])


def embedding_neighbours(test_p, ext_p):
    rows, bands = [], []
    for s in SEEDS:
        tr = _load_emb(s, "train")
        if tr is None:
            continue
        Xtr, ptr, _ = tr
        train_cls = np.array([p.split("/")[-2] for p in ptr])
        for split, P in (("test", test_p), ("external", ext_p)):
            q = _load_emb(s, split)
            if q is None or (PRIMARY, s) not in P:
                continue
            Xq, pq, _ = q
            sims = np.empty(len(Xq)); nn = np.empty(len(Xq), int)
            for i in range(0, len(Xq), 1024):
                S = Xq[i:i + 1024] @ Xtr.T
                nn[i:i + 1024] = S.argmax(1); sims[i:i + 1024] = S.max(1)
            df = P[(PRIMARY, s)].set_index("relative_path").loc[pq]
            correct = (df["class"] == df["pred"]).to_numpy()
            same_cls = train_cls[nn] == df["class"].to_numpy()
            rows.append({"seed": s, "split": split, "n": len(Xq), "median_max_cos": np.median(sims),
                         "q95_max_cos": np.percentile(sims, 95), "n_ge_0.95": int((sims >= 0.95).sum()),
                         "n_ge_0.98": int((sims >= 0.98).sum()), "nn_same_class_pct": 100 * same_cls.mean()})
            edges = [-1, 0.80, 0.90, 0.95, 1.01]
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (sims >= lo) & (sims < hi)
                bands.append({"seed": s, "split": split, "band": f"[{max(lo, 0):.2f}, {min(hi, 1):.2f})", "n": int(m.sum()),
                              "accuracy": 100 * correct[m].mean() if m.any() else np.nan,
                              "nn_same_class_pct": 100 * same_cls[m].mean() if m.any() else np.nan})
    if not rows:
        log("no embeddings - skip neighbour audit"); return None
    out = pd.DataFrame(rows)
    save_table(out, "embedding_neighbours", float_format="%.3f")
    b = pd.DataFrame(bands).groupby(["split", "band"]).agg(n=("n", "mean"), accuracy=("accuracy", "mean"),
                                                          nn_same_class_pct=("nn_same_class_pct", "mean")).reset_index()
    save_table(b, "embedding_neighbour_bands")
    return out


def origin_probe(la):
    """Can image origin be read from the embedding beyond what the class explains?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    rows = []
    org = la.set_index("key")["origin"]
    for s in SEEDS:
        for split in ("test", "train"):
            e = _load_emb(s, split)
            if e is None:
                continue
            X, paths, _ = e
            keys = pd.Series(paths)
            y = org.reindex(keys).to_numpy()
            cls = np.array([p.split("/")[-2] for p in paths])
            for c in CLASSES:
                m = (cls == c) & pd.notna(y)
                yc = y[m]
                if len(np.unique(yc)) < 2 or min(np.bincount(pd.factorize(yc)[0])) < 5:
                    continue
                skf = StratifiedKFold(5, shuffle=True, random_state=s)
                yp = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"), X[m], yc, cv=skf)
                rows.append({"seed": s, "split": split, "class": CLASS_LABELS[c], "n": int(m.sum()),
                             "n_dermnet": int((yc == "dermnet").sum()), "probe_bacc": 100 * balanced_accuracy_score(yc, yp)})
    if not rows:
        return None
    out = pd.DataFrame(rows).groupby(["split", "class"]).agg(n=("n", "first"), n_dermnet=("n_dermnet", "first"),
                                                             probe_bacc_mean=("probe_bacc", "mean"),
                                                             probe_bacc_sd=("probe_bacc", "std")).reset_index()
    out["chance"] = 50.0
    save_table(out, "origin_probe_M4")
    return out


def trainable_parameters():
    try:
        import yaml
        sys.path.insert(0, str(PROJECT))
        from src.models.factory import build_model
    except Exception as e:  # pragma: no cover - torch/timm missing
        log(f"skip trainable-parameter table ({e})"); return None
    rows = []
    for mid, cfg_path in MODEL_CONFIGS.items():
        with open(PROJECT / cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg)
        tot = sum(p.numel() for p in model.parameters())
        trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bb = getattr(model, "backbone", None)
        bb_trn = sum(p.numel() for p in bb.parameters() if p.requires_grad) if bb is not None else np.nan
        rows.append({"model": mid, "backbone": cfg["model"]["backbone"],
                     "input": "x".join(map(str, cfg.get("data", {}).get("input_size", []))),
                     "freeze": f"{cfg['model'].get('freeze_strategy', 'last_n')}:{cfg['model'].get('freeze_except_last_n', '')}",
                     "params_M": tot / 1e6, "trainable_M": trn / 1e6, "trainable_backbone_M": bb_trn / 1e6,
                     "trainable_pct": 100 * trn / tot})
    out = pd.DataFrame(rows)
    save_table(out, "trainable_parameters", float_format="%.3f")
    return out


def main() -> None:
    test_p, ext_p = preds("test"), preds("external")
    la = label_audit()
    origin_accuracy(test_p, la)
    strict_subset(test_p, la)
    external_subsets(test_p, ext_p)
    pairwise_holm(test_p)
    aurc(test_p, ext_p)
    color_baseline_ablation()
    embedding_neighbours(test_p, ext_p)
    origin_probe(la)
    trainable_parameters()
    log("stage 16 done")


if __name__ == "__main__":
    main()
