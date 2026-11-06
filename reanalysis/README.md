# Re-execution package

Complete, scripted re-execution of the PathoVision experiments with the
additional analyses requested during peer review. Training recipes, model
code and data pipeline are those of the original repository; this folder only
orchestrates them, adds seeds, and derives the extra tables and figures.

Nothing here modifies `src/`, `scripts/`, `configs/` or `datasets/*.py`.

## What it produces

| Stage | Script | Needs | Output |
|---|---|---|---|
| 01 | `s01_build_internal.py` | Kaggle CLI | `datasets/data/{train,val,test}` (seed-42 split), `provenance.csv` (source dataset/folder of every image), exact-hash dedup, near-duplicate audit (Hamming <= 10) |
| 02 | `s02_build_external.py` | Kaggle CLI, HuggingFace | `datasets/external_unified/`, source x class counts, external-vs-test near-duplicate audit |
| 03 | `s03_train_all.py` | GPU | 9 configurations x 3 seeds (42, 43, 44) with `scripts/train.py`; resumable |
| 04 | `s04_predict_all.py` | GPU | per-image probabilities on val/test/external, brightness-normalised and centre-cropped re-evaluations, M4 embeddings, FLOPs/latency |
| 05 | `s05_ensemble_hybrid.py` | CPU | M6 soft-voting ensemble (M2+M3+M4) and H1 random forest on M4 embeddings |
| 06 | `s06_cross_validate_full.py` | GPU | 5-fold CV of M1 and M4 with the full recipe |
| 07 | `s07_metrics.py` | CPU | overall / per-class / external / balancing / disease-only / variant / efficiency tables |
| 08 | `s08_calibration.py` | CPU | ECE, MCE, NLL, Brier, temperature scaling, selective prediction, reliability data |
| 09 | `s09_brightness_baseline.py` | CPU | shallow classifiers on global colour statistics (Normal-vs-disease and 7-class) |
| 10 | `s10_source_probe.py` | CPU | source x class confounding, per-source accuracy, linear probe of source identity on embeddings |
| 11 | `s11_ita.py` | CPU | corrected ITA (CIELAB scale), strata composition, accuracy by stratum, omnibus chi-square, class-adjusted logistic model |
| 12 | `s12_gradcam_quant.py` | GPU | quantitative Grad-CAM localisation (central/border/bottom activation mass) |
| 13 | `s13_stats.py` | CPU | McNemar, paired tests across seeds, CV confidence intervals and Nadeau-Bengio test, bootstrap CI of the external drop |
| 14 | `s14_figures.py` | CPU | journal-ready figures (84/174 mm, 9-pt, EPS + 600-dpi TIFF + PDF) |

All tables are written as CSV and LaTeX under `results/tables/`.

## Running

```bash
pip install -r requirements.txt statsmodels
python reanalysis/run_all.py                 # everything
python reanalysis/run_all.py --from 07       # CPU stages from existing predictions
python reanalysis/run_all.py --smoke         # synthetic end-to-end test (CPU, ~5 min)
```

Two environment variables control where large artefacts live:

* `REANALYSIS_ROOT` - `runs/` (checkpoints, predictions) and `results/`;
  on Colab point it to a Google Drive folder so they survive session resets.
* `REANALYSIS_DATA_ROOT` - the `datasets/` tree; keep it on a fast local disk.

The Colab notebook `PathoVision_Reanalysis.ipynb` wires this up (Drive for
runs/results, local disk for images, Kaggle credentials entered by the user).

Every stage is idempotent. A training run is complete only when
`runs/<model>/seed<seed>/done.json` exists; interrupted runs restart from
scratch on the next call, so `s03_train_all.py` can simply be re-run after a
session limit until all 27 runs are done.

## Layout of the outputs

```
runs/<model>/seed<seed>/
    config.yaml            exact configuration used
    checkpoints/best.pt    best-validation checkpoint
    done.json              training time, device, best epoch and metrics
    pred_<split>.csv       path, class, pred, confidence, proba_<class>...
    emb_<split>.npz        penultimate embeddings (M4 only)
    efficiency.json        parameters, GFLOPs, latency
runs/cv/<model>/fold<k>/   cross-validation runs
results/tables/*.csv|tex   every table of the paper and supplement
results/figures/Fig_*.{eps,tif,pdf}
results/dataset_qc/        provenance, duplicate audits, source counts
results/calibration/, results/ita/, results/gradcam/, results/image_stats/
```

Predictions and tables are small (tens of MB); checkpoints are not needed for
any downstream stage once stage 04 and 12 have run.
