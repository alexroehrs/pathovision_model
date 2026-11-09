"""Stage 23 - tables of the data-integrity and robustness experiments (CPU, from the outputs of 17-22).

Each part is skipped when its inputs do not exist yet, so the stage can be run
after any subset of stages 17-22.

* ``cv_nested_folds`` / ``cv_nested_summary`` - nested CV per fold and summary
  next to the non-nested CV of stage 06 (optimism = non-nested - nested), with
  the Nadeau-Bengio corrected t-test for M4 vs M1;
* ``matched_finetune`` - backbones under the matched protocol (test, external,
  external independent sources), McNemar vs ConvNeXt-Tiny with
  Holm correction;
* ``strict_retrain`` - strict-label retraining vs. the original models;
* ``leaky_split`` - inherited non-deduplicated split: test accuracy overall and
  for test images with / without an identical image in training, external;
* ``probe_frozen`` - logistic regression on frozen DINOv2 / ImageNet features;
* ``dinov2_neighbours`` (+ ``dinov2_top_pairs.csv``) - model-independent
  near-duplicate audit; ``origin_probe_dinov2`` - origin probe on DINOv2;
* ``border_mask`` and ``deletion_test`` - stage 22.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_LABELS, CLASSES, MODEL_CONFIGS, PRIMARY, RESULTS, RUNS, SEEDS, ensure_dir, global_metrics, log, mean_sd, save_table  # noqa: E402
from s16_reanalysis import DERMNET_OVERLAP_SOURCES, PROTOCOL_MISMATCH, external_meta, holm, mcnemar_exact, neardup_paths  # noqa: E402


def _read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def _gm(df):
    return global_metrics(df["class"].to_numpy(), df["pred"].to_numpy())


def _ext_subsets(df, meta):
    d = df.merge(meta, on="relative_path", how="left")
    out = {"external (all)": d}
    nd = neardup_paths("external")
    if nd is not None:
        out["external, no near-duplicates"] = d[~d.relative_path.isin(nd)]
    out["external, independent"] = d[~d.source.isin(DERMNET_OVERLAP_SOURCES)]
    out["external, independent protocol-matched"] = d[~d.source.isin(DERMNET_OVERLAP_SOURCES) & ~d.source_label.isin(PROTOCOL_MISMATCH)]
    return out


def _meta():
    for mid in MODEL_CONFIGS:
        for s in SEEDS:
            p = RUNS / mid / f"seed{s}" / "pred_external.csv"
            if p.exists():
                return external_meta(pd.read_csv(p))
    return None


# --------------------------------------------------------------------------
def nested_cv():
    rows = []
    for mid in ("M1", "M4"):
        for k in range(10):
            d = _read(RUNS / "ncv" / mid / f"fold{k}" / "pred_heldout.csv")
            if d is None:
                continue
            m = _gm(d); t = _read(RUNS / "ncv" / mid / f"fold{k}" / "pred_test.csv")
            r = {"model": mid, "fold": k, "design": "nested", "n": m["n"], "accuracy": m["accuracy"],
                 "balanced_accuracy": m["balanced_accuracy"], "f1_macro": m["f1_macro"],
                 "test_accuracy": _gm(t)["accuracy"] if t is not None else np.nan}
            rows.append(r)
            o = _read(RUNS / "cv" / mid / f"fold{k}" / "pred_val.csv")
            if o is not None:
                mo = _gm(o)
                rows.append({"model": mid, "fold": k, "design": "non-nested (stage 06)", "n": mo["n"], "accuracy": mo["accuracy"],
                             "balanced_accuracy": mo["balanced_accuracy"], "f1_macro": mo["f1_macro"], "test_accuracy": np.nan})
    if not rows:
        return
    df = pd.DataFrame(rows)
    save_table(df, "cv_nested_folds")
    summ = []
    for (mid, des), g in df.groupby(["model", "design"]):
        v = g["accuracy"].to_numpy(); k = len(v)
        half = stats.t.ppf(0.975, k - 1) * v.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
        summ.append({"model": mid, "design": des, "folds": k, "accuracy": mean_sd(v), "ci_lo": v.mean() - half, "ci_hi": v.mean() + half,
                     "balanced_accuracy": mean_sd(g["balanced_accuracy"]), "f1_macro": mean_sd(g["f1_macro"]),
                     "fold_models_on_test": mean_sd(g["test_accuracy"]) if g["test_accuracy"].notna().any() else ""})
    s = pd.DataFrame(summ)
    nest = df[df.design == "nested"].pivot(index="fold", columns="model", values="accuracy").dropna()
    if {"M1", "M4"} <= set(nest.columns) and len(nest) > 1:
        d = (nest["M4"] - nest["M1"]).to_numpy(); k = len(d)
        n_test, n_train = 1, k - 1                     # each fold: 1/k held out, (k-1)/k for training
        var_c = d.var(ddof=1) * (1 / k + n_test / n_train)
        t = d.mean() / np.sqrt(var_c); p = 2 * stats.t.sf(abs(t), k - 1)
        s = pd.concat([s, pd.DataFrame([{"model": "M4 - M1", "design": "nested", "folds": k, "accuracy": mean_sd(d),
                                         "ci_lo": np.nan, "ci_hi": np.nan, "balanced_accuracy": f"Nadeau-Bengio t={t:.2f}, p={p:.4f}",
                                         "f1_macro": "", "fold_models_on_test": ""}])], ignore_index=True)
    save_table(s, "cv_nested_summary")


def matched(meta):
    bbs = ["efficientnet_b2", "efficientnetv2_s", "swin_tiny", "convnext_tiny"]
    names = {"efficientnet_b2": "EfficientNet-B2", "efficientnetv2_s": "EfficientNetV2-S", "swin_tiny": "Swin-Tiny", "convnext_tiny": "ConvNeXt-Tiny"}
    rows, preds = [], {}
    for bb in bbs:
        res = {"test": [], **{k: [] for k in ("external (all)", "external, independent")}}
        for s in SEEDS:
            t = _read(RUNS / "matched" / bb / f"seed{s}" / "pred_test.csv")
            if t is None:
                continue
            preds[(bb, s)] = t.sort_values("relative_path")
            res["test"].append(_gm(t))
            e = _read(RUNS / "matched" / bb / f"seed{s}" / "pred_external.csv")
            if e is not None and meta is not None:
                sub = _ext_subsets(e, meta)
                for k in ("external (all)", "external, independent"):
                    res[k].append(_gm(sub[k]))
        if not res["test"]:
            continue
        r = {"backbone": names[bb], "seeds": len(res["test"])}
        for k, v in res.items():
            if v:
                r[f"{k} acc"] = mean_sd([g["accuracy"] for g in v])
                r[f"{k} bacc"] = mean_sd([g["balanced_accuracy"] for g in v])
                r[f"{k} F1-M"] = mean_sd([g["f1_macro"] for g in v])
        rows.append(r)
    if not rows:
        return
    save_table(pd.DataFrame(rows), "matched_finetune")
    tests = []
    for s in SEEDS:
        ref = preds.get(("convnext_tiny", s))
        if ref is None:
            continue
        ps, labels = [], []
        for bb in bbs[:-1]:
            o = preds.get((bb, s))
            if o is None:
                continue
            ps.append(mcnemar_exact((ref["class"] == ref["pred"]).to_numpy(), (o["class"] == o["pred"]).to_numpy())); labels.append(bb)
        if ps:
            for bb, p, a in zip(labels, ps, holm(ps)):
                tests.append({"seed": s, "comparison": f"ConvNeXt-Tiny vs {names[bb]}", "p_raw": p, "p_holm": a})
    if tests:
        save_table(pd.DataFrame(tests), "matched_finetune_mcnemar", float_format="%.4f")


def strict(meta):
    la = _read(RESULTS / "dataset_qc" / "label_audit.csv")
    rows = []
    for mid in ("M4", "M1"):
        for design, base in (("strict retrain", RUNS / "strict" / mid), ("original", RUNS / mid)):
            res = {}
            for s in SEEDS:
                rd = base / f"seed{s}"
                t = _read(rd / "pred_test.csv")
                if t is None:
                    continue
                st = _read(rd / "pred_strict_test.csv")
                if st is None and la is not None:
                    keep = set(la.loc[la["strict"] == True, "relative_path"])  # noqa: E712
                    st = t[t.relative_path.isin(keep)]
                res.setdefault("strict test", []).append(_gm(st))
                res.setdefault("full test", []).append(_gm(t))
                e = _read(rd / "pred_external.csv")
                if e is not None and meta is not None:
                    for k, v in _ext_subsets(e, meta).items():
                        res.setdefault(k, []).append(_gm(v))
            if design == "strict retrain" and not res:
                continue
            for k, v in res.items():
                rows.append({"model": mid, "training data": design, "evaluation": k, "n": v[0]["n"], "seeds": len(v),
                             "accuracy": mean_sd([g["accuracy"] for g in v]), "balanced_accuracy": mean_sd([g["balanced_accuracy"] for g in v]),
                             "f1_macro": mean_sd([g["f1_macro"] for g in v])})
    if any(r["training data"] == "strict retrain" for r in rows):
        save_table(pd.DataFrame(rows), "strict_retrain")


def leaky(meta):
    man = _read(RESULTS / "dataset_qc" / "leaky_manifest.csv")
    if man is None:
        return
    t = man[man.split == "test"]
    qc = [{"item": "inherited split images", "value": len(man)}, {"item": "inherited test images", "value": len(t)},
          {"item": "test images with an identical image (MD5 or dHash) in train/val", "value": int(t.dup_in_fit.sum())},
          {"item": "test images with a byte-identical image in train/val", "value": int(t.dup_in_fit_md5.sum())},
          {"item": "validation images with an identical image in train", "value": int(man.val_dup_in_train.sum())}]
    dup = t.set_index("relative_path")["dup_in_fit"]
    # does the identical training/validation copy carry the same class label?
    fit = man[man.split.isin(["train", "val"])]
    cls_md5 = fit.groupby("md5")["class"].agg(set); cls_dh = fit.groupby("dhash")["class"].agg(set)
    def _kind(r):
        k = cls_md5.get(r.md5, set()) | cls_dh.get(r.dhash, set())
        return "none" if not k else ("same" if k == {r["class"]} else "cross")
    kind = t.apply(_kind, axis=1); kind.index = t["relative_path"]
    qc.append({"item": "of which with the same class label", "value": int((kind == "same").sum())})
    qc.append({"item": "of which with another class label", "value": int((kind == "cross").sum())})
    save_table(pd.DataFrame(qc), "leaky_split_qc", float_format="%d")
    rows, adj = [], {}
    for mid in ("M4", "M1"):
        res = {}
        for s in SEEDS:
            p = _read(RUNS / "leaky" / mid / f"seed{s}" / "pred_test.csv")
            if p is None:
                continue
            p["dup"] = p.relative_path.map(dup).fillna(False).astype(bool)
            res.setdefault("inherited test (all)", []).append(_gm(p))
            res.setdefault("inherited test, duplicated in training", []).append(_gm(p[p.dup]))
            res.setdefault("inherited test, not duplicated", []).append(_gm(p[~p.dup]))
            p["kind"] = p.relative_path.map(kind)
            res.setdefault("inherited test, copy with the same label", []).append(_gm(p[p.kind == "same"]))
            res.setdefault("inherited test, copy with another label", []).append(_gm(p[p.kind == "cross"]))
            g = p[p.kind.isin(["none", "same"])].assign(c=p["class"] == p["pred"]).groupby(["class", "kind"])["c"].mean().unstack().dropna()
            w = p[(p.kind == "same") & p["class"].isin(g.index)].groupby("class").size()
            adj.setdefault(mid, []).append(100 * float(((g["same"] - g["none"]) * w).sum() / w.sum()))
            e = _read(RUNS / "leaky" / mid / f"seed{s}" / "pred_external.csv")
            if e is not None and meta is not None:
                for k, v in _ext_subsets(e, meta).items():
                    if k in ("external (all)", "external, independent"):
                        res.setdefault(k, []).append(_gm(v))
        for k, v in res.items():
            rows.append({"model": mid, "evaluation": k, "n": v[0]["n"], "seeds": len(v), "accuracy": mean_sd([g["accuracy"] for g in v]),
                         "balanced_accuracy": mean_sd([g["balanced_accuracy"] for g in v]), "f1_macro": mean_sd([g["f1_macro"] for g in v])})
    for mid, v in adj.items():
        rows.append({"model": mid, "evaluation": "class-adjusted difference, same-label copy - not duplicated (pp)", "n": np.nan,
                     "seeds": len(v), "accuracy": mean_sd(v), "balanced_accuracy": "", "f1_macro": ""})
    if rows:
        save_table(pd.DataFrame(rows), "leaky_split")


def _emb(name, split):
    p = RUNS / "probe" / name / f"emb_{split}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["X"].astype(np.float32), np.asarray(z["path"]).astype(str), np.asarray(z["label"]).astype(str)


def probes(meta):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    rows = []
    for name in ("dinov2_vitb14", "convnext_tiny_in1k"):
        tr, va, te = _emb(name, "train"), _emb(name, "val"), _emb(name, "test")
        if tr is None or va is None or te is None:
            continue
        sc = StandardScaler().fit(tr[0])
        best = None
        for C in (0.001, 0.01, 0.1, 1.0):
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(tr[0]), tr[2])
            b = balanced_accuracy_score(va[2], clf.predict(sc.transform(va[0])))
            if best is None or b > best[0]:
                best = (b, C, clf)
        _, C, clf = best
        evals = {"test": pd.DataFrame({"relative_path": te[1], "class": te[2], "pred": clf.predict(sc.transform(te[0]))})}
        ex = _emb(name, "external")
        if ex is not None and meta is not None:
            e = pd.DataFrame({"relative_path": ex[1], "class": ex[2], "pred": clf.predict(sc.transform(ex[0]))})
            evals.update(_ext_subsets(e, meta))
        for k, d in evals.items():
            m = _gm(d)
            rows.append({"encoder": name, "C": C, "evaluation": k, "n": m["n"], "accuracy": m["accuracy"],
                         "balanced_accuracy": m["balanced_accuracy"], "f1_macro": m["f1_macro"]})
    if rows:
        save_table(pd.DataFrame(rows), "probe_frozen")


def dinov2_audit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    name = "dinov2_vitb14"
    tr = _emb(name, "train")
    if tr is None:
        return
    def norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    parts = {s: _emb(name, s) for s in ("train", "val", "test", "external")}
    ref = np.concatenate([norm(parts[s][0]) for s in ("train", "val") if parts[s] is not None])
    ref_p = np.concatenate([parts[s][1] for s in ("train", "val") if parts[s] is not None])
    ref_c = np.concatenate([parts[s][2] for s in ("train", "val") if parts[s] is not None])
    rows, pairs = [], []
    for split in ("test", "external"):
        if parts[split] is None:
            continue
        X, P, Y = parts[split]
        X = norm(X)
        sims = np.empty(len(X)); nn = np.empty(len(X), int)
        for i in range(0, len(X), 1024):
            S = X[i:i + 1024] @ ref.T
            nn[i:i + 1024] = S.argmax(1); sims[i:i + 1024] = S.max(1)
        for thr in (0.90, 0.95, 0.98):
            m = sims >= thr
            rows.append({"query": split, "reference": "train+val", "threshold": thr, "n_query": len(X), "n_above": int(m.sum()),
                         "pct_above": 100 * m.mean(), "same_class_pct": 100 * (ref_c[nn[m]] == Y[m]).mean() if m.any() else np.nan})
        top = np.argsort(-sims)[:300]
        pairs += [{"query_split": split, "query": P[i], "query_class": Y[i], "neighbour": ref_p[nn[i]], "neighbour_class": ref_c[nn[i]],
                   "cosine": sims[i]} for i in top]
        np.save(ensure_dir(RESULTS / "dataset_qc") / f"dinov2_maxcos_{split}.npy", sims.astype(np.float32))
    save_table(pd.DataFrame(rows), "dinov2_neighbours")
    pd.DataFrame(pairs).to_csv(RESULTS / "dataset_qc" / "dinov2_top_pairs.csv", index=False)
    # origin probe on model-independent features
    la = _read(RESULTS / "dataset_qc" / "label_audit.csv")
    if la is None:
        return
    org = la.set_index("relative_path")["origin"]
    pr = []
    for split in ("test", "train"):
        X, P, Y = parts[split]
        y = org.reindex(P).to_numpy()
        for c in CLASSES:
            m = (Y == c) & pd.notna(y)
            yc = y[m]
            if len(np.unique(yc)) < 2 or np.bincount(pd.factorize(yc)[0]).min() < 5:
                continue
            yp = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"), norm(X[m]), yc,
                                   cv=StratifiedKFold(5, shuffle=True, random_state=0))
            pr.append({"split": split, "class": CLASS_LABELS[c], "n": int(m.sum()), "n_dermnet": int((yc == "dermnet").sum()),
                       "probe_bacc": 100 * balanced_accuracy_score(yc, yp)})
    if pr:
        save_table(pd.DataFrame(pr), "origin_probe_dinov2")


def faithfulness():
    rows = []
    for mid in MODEL_CONFIGS:
        for split in ("test", "external"):
            res = {}
            for s in SEEDS:
                rd = RUNS / mid / f"seed{s}"
                for tag in ("", "_border10", "_border15", "_centercrop"):
                    d = _read(rd / f"pred_{split}{tag}.csv")
                    if d is not None:
                        res.setdefault(tag, []).append(_gm(d)["accuracy"])
            if res.get("_border10"):
                r = {"model": mid, "split": split}
                for tag, lab in (("", "original"), ("_border10", "10 % border masked"), ("_border15", "15 % border masked"), ("_centercrop", "central 70 % crop")):
                    if res.get(tag):
                        r[lab] = mean_sd(res[tag])
                rows.append(r)
    if rows:
        save_table(pd.DataFrame(rows), "border_mask")
    fd = RESULTS / "faithfulness"
    files = sorted(fd.glob("deletion_*.csv")) if fd.exists() else []
    if not files:
        return
    out = []
    for f in files:
        d = pd.read_csv(f)
        _, split, mid, seed = f.stem.split("_", 3)
        base = d[d.fraction == 0].groupby("path")["prob_pred"].first()
        d["rel_prob"] = d["prob_pred"] / d["path"].map(base).clip(lower=1e-6)
        curves = d.groupby(["order", "fraction"])["rel_prob"].mean().unstack(0)
        piv = d.pivot_table(index=["order", "path"], columns="fraction", values="rel_prob").sort_index(axis=1)
        trap = getattr(np, "trapezoid", None) or np.trapz
        auc = pd.Series(trap(piv.to_numpy(), piv.columns.to_numpy(float), axis=1), index=piv.index).unstack(0)
        w = stats.wilcoxon(auc["gradcam"], auc["random"])
        r = {"split": split, "model": mid, "seed": seed.replace("seed", ""), "n_images": len(auc),
             "auc_gradcam": auc["gradcam"].mean(), "auc_random": auc["random"].mean(),
             "pct_images_gradcam_lower": 100 * (auc["gradcam"] < auc["random"]).mean(), "wilcoxon_p": w.pvalue}
        for fr in curves.index:
            r[f"gradcam@{fr:.2f}"] = curves.loc[fr, "gradcam"]; r[f"random@{fr:.2f}"] = curves.loc[fr, "random"]
        out.append(r)
    save_table(pd.DataFrame(out), "deletion_test", float_format="%.4f")


def main() -> None:
    meta = _meta()
    nested_cv()
    matched(meta)
    strict(meta)
    leaky(meta)
    probes(meta)
    dinov2_audit()
    faithfulness()
    log("stage 23 done")


if __name__ == "__main__":
    main()
