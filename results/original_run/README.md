# Preliminary-run results

Output files of a preliminary single-seed run of the training and evaluation pipeline (cluster runs with seed 42), kept for the record. The results reported in the article come from the seed-replicated re-execution in `reanalysis/`; the numbers in this folder are not those of the article. The independent re-execution on consumer hardware is documented separately in `reproduction/`.

| Folder | Contents |
|---|---|
| `01_dataset_qc/` | dHash duplicate audit of the merged 13,298-image corpus: `dataset_qc_report.json` (summary and per-class brightness statistics) and `duplicate_pairs.txt` (every duplicate pair found, with the inherited partition of each copy) |
| `02_baseline/` | M1, EfficientNetB2, cross-entropy, unbalanced |
| `02b_baseline_balanced/` | M1b, EfficientNetB2, cross-entropy + balanced sampling |
| `05_focal_balanced/` | M5, EfficientNetB2, focal loss + balanced sampling |
| `03_m2_v2s_balanced/`, `03_m2b_v2s_nobal/` | M2 / M2b, EfficientNetV2-S |
| `03_m3_swin_balanced/`, `03_m3b_swin_nobal/` | M3 / M3b, Swin-Tiny |
| `03_m4_convnext_balanced/`, `03_m4b_convnext_nobal/` | M4 / M4b, ConvNeXt-Tiny |
| `04_cross_validation/` | Five-fold stratified cross-validation of the baseline M1 (mean, standard deviation and 95% CI per metric) |
| `05_ablation/` | Best-validation accuracy, loss and epoch of every training run (one row per model, plus the five M1 folds). The `ce_loss_only_(no_weights)` row and the table caption come from a single-factor run |
| `07_fairness/` | Image-level ITA estimates of the 929 test images (`skin_tone_annotations.csv`), the accuracy per stratum (`fairness_metrics.csv`) and the corresponding plots |
| `08_external_validation/` | Internal versus external accuracy and weighted F1 of all nine end-to-end models on the 4,804-image external set (`unified_external_validation.csv`) |

Each model folder contains the normalized and absolute confusion matrices, the ROC and PR curves, the per-class metrics (`*_per_class.csv`), a LaTeX table and a `*_comparison.csv` that lists the per-class F1 next to reference values of the EfficientNetB2 baseline.

Notes. The per-class CSVs are the authoritative source of the internal metrics; the `Int Acc`/`Int F1` columns of `08_external_validation/unified_external_validation.csv` were copied from an interim summary and differ for M2 (86.65 vs. 83.85 from the per-class file) and M3 (85.47 vs. 85.36). The ITA values in `07_fairness/` are on the 8-bit OpenCV CIELab scale; the stratum labels are nominal. Per-sample prediction files of these runs were not retained. Grad-CAM panels of individual test photographs are not redistributed here because they contain the source images.
