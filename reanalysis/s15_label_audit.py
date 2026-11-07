"""Stage 15 - provenance by origin and file-name label audit (CPU).

The two internal Kaggle collections overlap: many "Human Skin Diseases" (HSD)
files are byte-identical copies of DermNet files, so the source recorded for a
deduplicated image (stage 01) reflects the deduplication tie-break rather than
the image's origin.  This stage re-derives provenance and audits labels:

* ``origin`` - ``dermnet`` if the image's MD5 occurs among the raw DermNet
  files (whatever package it was retained from), otherwise ``hsd_only``;
* ``label_conflict`` - the same MD5 appears under two different target
  classes in the raw downloads;
* ``dx_tag`` / ``target_status`` - DermNet file names encode the specific
  diagnosis (e.g. ``lichen-planus-12.jpg`` inside the Psoriasis folder).  A
  transparent rule table maps file-name stems to ``on`` (the target entity),
  ``off`` (a different entity grouped under the same folder), ``nonclinical``
  (histology or laboratory images) or ``unknown`` (uninformative names, e.g.
  numeric HSD names);
* a **strict-label subset**: images that are not label-conflicted and whose
  status is ``on`` or ``unknown``, keeping their original split assignment.
  Its manifest drives the retraining of stage 19 and the strict-subset
  evaluation of stage 16.

Outputs in ``results/dataset_qc/``: ``label_audit.csv`` (one row per image),
``label_audit_summary.csv`` (class x status counts), ``origin_by_class.csv``,
``strict_manifest.csv``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, CLASS_LABELS, DATA_DIR, RESULTS, ensure_dir, log, save_table  # noqa: E402

# --------------------------------------------------------------------------
# File-name rules.  Patterns are matched (re.search) against the lower-cased
# file name; the first matching rule wins.  ``nonclinical`` rules apply to all
# classes.
# --------------------------------------------------------------------------
NONCLINICAL = [r"histolog", r"tinea-laboratory", r"-laboratory"]
RULES: dict[str, list[tuple[str, str]]] = {
    "Acne": [
        (r"rosacea|rhinophym|rhnophym|perioral|steroidperioral|hidradenitis|milia|minocycline|fordyce|"
         r"nevus-comedonicus|hyperhidrosis|sebaceous|sebderm|sebderem|vessels|vascular|steroidros|anal-comedones", "off"),
        (r"acne|comedo", "on"),
    ],
    "Candidiasis": [
        (r"tinea", "off"),
        (r"candid|can-toe|monilia", "on"),
    ],
    "Eczema": [
        (r"stasis|lichen-simplex|neurotic|keratolysis|prurigo|chapped|heels|factitial|fiberglass|id-reaction|"
         r"milroy|desquamation|maceration|notalgia|phlebitis|parasitosis|emersion|fissure|drug|hyperkeratosis|"
         r"iododerma|lichen-sclerosus|lichen", "off"),
        (r"eczema|atopic|dyshidros|pompholyx|nummular|asteatotic|contactderm|dermatitislids", "on"),
    ],
    "NailFungus": [
        (r"onychomycosis|tinea|fungal", "on"),
        (r"bed-ridges|bedridges|psoriasis|lichen|mucous|alopecia|onycholysis|paronychia|habit|pseudomonas|trauma|ridging|median|"
         r"beau|eczema|koilonychia|pincer|yellow|biting|splitting|ingrown|clubbing|wart|twenty|pigment|"
         r"onychogryph|splinter|hematoma|congenital|leukonychia|pachyonychia|dry-nails|terry|normal-var|"
         r"melanoma|exostosis|fibroma|cyst|minocycline|hang-nail|chemotherapy|color-change|fissure|"
         r"lines|dystrophy|nail", "off"),
    ],
    "Psoriasis": [
        (r"pityriasis|lichen|seborrh|sebderm|sebkerm|granular-parakeratosis|palmoplantar|amiantacea|reiter|"
         r"biteline|glutealpinking|acne|lsa|eczema|erythrokeratoderm", "off"),
        (r"psoriasis|psoriatic", "on"),
    ],
    "Tinea": [
        (r"versicolor|intertri|intertirg|perleche|erosio|erythrasma|candid|monialis|pityrosporum|pitted|"
         r"confluent|coccidio|mycetoma|blastomycosis|actinomycosis|balanitis|id-reaction|fissure|cheilitis|contactderm", "off"),
        (r"tinea|tinped|tincorp|kerion|majocchi|ringworm|dermatophyt|tonsurans", "on"),
    ],
    "Normal": [
        (r"exterior|front-three-quarter|side-view|pendant|homesn", "off"),
    ],
}


def status_of(cls: str, fname: str) -> tuple[str, str]:
    f = fname.lower()
    for pat in NONCLINICAL:
        if re.search(pat, f):
            return "nonclinical", pat
    for pat, status in RULES.get(cls, []):
        m = re.search(pat, f)
        if m:
            return status, m.group(0)
    return "unknown", ""


def main() -> None:
    qc = RESULTS / "dataset_qc"
    prov = pd.read_csv(DATA_DIR / "provenance.csv") if (DATA_DIR / "provenance.csv").exists() else pd.read_csv(qc / "provenance.csv")
    raw = pd.read_csv(qc / "raw_provenance.csv")

    dermnet_md5 = set(raw.loc[raw["source_dataset"] == "dermnet_goel", "md5"])
    hsd_md5 = set(raw.loc[raw["source_dataset"] == "skin_diseases_youssef", "md5"])
    n_classes = raw.groupby("md5")["class"].nunique()
    conflict_md5 = set(n_classes[n_classes > 1].index)
    # file names of the DermNet copy are the informative ones; collect all raw names per md5
    names = raw.groupby("md5")["source_file"].apply(lambda s: " ".join(sorted(set(s))))

    prov["origin"] = prov["md5"].map(lambda h: "dermnet" if h in dermnet_md5 else "hsd_only")
    prov["in_both_packages"] = prov["md5"].map(lambda h: h in dermnet_md5 and h in hsd_md5)
    prov["label_conflict"] = prov["md5"].isin(conflict_md5)
    prov["all_raw_names"] = prov["md5"].map(names).fillna(prov["source_file"])
    st = [status_of(c, n) for c, n in zip(prov["class"], prov["all_raw_names"])]
    prov["target_status"] = [s for s, _ in st]
    prov["matched_rule"] = [r for _, r in st]
    # A conflict caused by the DermNet super-folder (e.g. "Tinea, Ringworm, Candidiasis and other Fungal
    # Infections") is resolved when the file name supports the retained class; otherwise it is a true conflict.
    prov["true_conflict"] = prov["label_conflict"] & (prov["target_status"] != "on")
    prov["strict"] = (~prov["true_conflict"]) & prov["target_status"].isin(["on", "unknown"])

    ensure_dir(qc)
    prov.drop(columns=["all_raw_names"]).to_csv(qc / "label_audit.csv", index=False)
    prov.loc[prov["strict"], ["relative_path", "split", "class", "origin"]].to_csv(qc / "strict_manifest.csv", index=False)

    # summaries
    rows = []
    for cls in CLASSES:
        g = prov[prov["class"] == cls]
        r = {"class": CLASS_LABELS[cls], "n": len(g), "dermnet_origin": int((g["origin"] == "dermnet").sum()),
             "in_both_packages": int(g["in_both_packages"].sum()), "label_conflict": int(g["label_conflict"].sum()),
             "true_conflict": int(g["true_conflict"].sum())}
        for s in ("on", "off", "nonclinical", "unknown"):
            r[s] = int((g["target_status"] == s).sum())
        r["strict"] = int(g["strict"].sum())
        r["strict_test"] = int((g["strict"] & (g["split"] == "test")).sum())
        r["test"] = int((g["split"] == "test").sum())
        rows.append(r)
    summ = pd.DataFrame(rows)
    tot = {"class": "Total", **{c: int(summ[c].sum()) for c in summ.columns if c != "class"}}
    summ = pd.concat([summ, pd.DataFrame([tot])], ignore_index=True)
    save_table(summ, "label_audit_summary", subdir="dataset_qc", float_format="%d")

    off = prov[prov["target_status"].isin(["off", "nonclinical"])]
    top = off.groupby(["class", "matched_rule"]).size().reset_index(name="n").sort_values(["class", "n"], ascending=[True, False])
    save_table(top, "label_audit_offtarget_terms", subdir="dataset_qc", float_format="%d")

    org = prov.pivot_table(index="class", columns="origin", values="md5", aggfunc="count", fill_value=0).reindex(CLASSES)
    save_table(org.reset_index(), "origin_by_class", subdir="dataset_qc", float_format="%d")
    log(f"label audit: {len(prov)} images; DermNet origin {int((prov.origin=='dermnet').sum())}; "
        f"conflicts {int(prov.label_conflict.sum())} (true {int(prov.true_conflict.sum())}); strict subset {int(prov.strict.sum())} "
        f"(test {int((prov.strict & (prov.split=='test')).sum())})")
    log("stage 15 done")


if __name__ == "__main__":
    main()
