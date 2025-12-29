# Radiomics → labels → XGBoost (no leakage)

This document lists **all scripts used to prepare datasets** for training and the exact commands to reproduce outputs.

## Inputs (what must exist)

- **Radiomics per-series JSONs** (produced by PyRadiomics batch):
  - `models/vista-3d/outputs/radiomics_per_series/*.json`
  - Produced by: `models/vista-3d/scripts/run_pyradiomics_batch.py`

- **TCIA imaging tree** (used to infer `patient_id` / `series_uid` mapping):
  - `models/vista-3d/data/tcia/NSCLC Radiogenomics/<PatientID>/<StudyUID>/<SeriesUID>/`

## 0) Radiomics extraction (PyRadiomics) + speed notes

Script:
- `models/vista-3d/scripts/run_pyradiomics_batch.py`

Typical command:

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
python /home/shadeform/models/vista-3d/scripts/run_pyradiomics_batch.py \
  --inputs /home/shadeform/models/vista-3d/outputs \
  --params /home/shadeform/models/vista-3d/radiomics_params.yaml \
  --workers 8 \
  --itk-threads 1 \
  --lib-threads 1 \
  --out-csv /home/shadeform/models/vista-3d/outputs/radiomics_features.csv
```

Speed notes (what dominates runtime):
- `imageType: Wavelet` and `imageType: LoG` (especially multiple `sigma`) can easily make extraction **~10x+ slower**, because features are computed on many filtered images (e.g., 1 original + 8 wavelets + N LoG sigmas).
- `setting: resampledPixelSpacing` can be expensive. In this repo, `scripts/run_inference_on_series.py --save-ct` already writes `*_ct_resampled.nii.gz` at **0.7mm isotropic**, so re-resampling inside PyRadiomics is usually redundant.

For quick iteration, use the faster params file:
- `radiomics_params_fast.yaml` (Original-only, no redundant resampling, pre-crop enabled)

## 1) NSCLC Radiogenomics labels (EGFR/KRAS + clinical)

Script:
- `models/vista-3d/scripts/derive_nsclc_radiogenomics_labels.py`

What it does:
- Downloads TCIA “data labels” CSV (EGFR/KRAS mutation status, smoking, TNM, survival, histology…)
- Joins labels to your local TCIA series directories
- Writes a **series-level** labels CSV for joining to radiomics by `series_uid`

Command (recommended: restrict to segmented series present in `outputs/`):

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
python /home/shadeform/models/vista-3d/scripts/derive_nsclc_radiogenomics_labels.py \
  --tcia-root /home/shadeform/models/vista-3d/data/tcia \
  --dataset 'NSCLC Radiogenomics' \
  --outputs-dir /home/shadeform/models/vista-3d/outputs \
  --out-csv /home/shadeform/models/vista-3d/outputs/nsclc_radiogenomics_labels.csv
```

Output:
- `models/vista-3d/outputs/nsclc_radiogenomics_labels.csv`

## 2) Build ML-ready table (radiomics + metadata + splits + optional labels)

Script:
- `models/vista-3d/scripts/prepare_xgb_dataset.py`

What it does:
- Reads `outputs/radiomics_per_series/*.json`
- Infers `patient_id` and `dataset` from the `data/tcia/` directory structure
- Creates leakage-safe `split` (train/val/test) **by patient** (group split)
- Optionally joins labels CSV

Command (NSCLC radiomics + labels):

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
python /home/shadeform/models/vista-3d/scripts/prepare_xgb_dataset.py \
  --radiomics-per-series-dir /home/shadeform/models/vista-3d/outputs/radiomics_per_series \
  --labels-csv /home/shadeform/models/vista-3d/outputs/nsclc_radiogenomics_labels.csv \
  --tcia-root /home/shadeform/models/vista-3d/data/tcia \
  --dataset-filter 'NSCLC Radiogenomics' \
  --out-csv /home/shadeform/models/vista-3d/outputs/ml/nsclc_radiomics_ml_table_labeled.csv
```

Output:
- `models/vista-3d/outputs/ml/nsclc_radiomics_ml_table_labeled.csv`

Leakage note:
- This script creates `split` at the **patient** level. Do **not** re-split by row.

## 3) Feature selection (train-only; no leakage)

Script:
- `models/vista-3d/scripts/feature_selection.py`

What it does (on **train split only**):
- Missingness filter
- Constant feature removal
- Univariate ranking
- Correlation pruning
- Optional XGBoost gain ranking (uses `xgboost.train`, no sklearn)

Command (EGFR):

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
python /home/shadeform/models/vista-3d/scripts/feature_selection.py \
  --data /home/shadeform/models/vista-3d/outputs/ml/nsclc_radiomics_ml_table_labeled.csv \
  --label egfr_mutated \
  --task binary \
  --use-xgb \
  --topk-univariate 300 \
  --corr-thresh 0.95 \
  --topk-final 50 \
  --out-dir /home/shadeform/models/vista-3d/outputs/ml/feature_selection/egfr_mutated
```

Outputs:
- `.../selected_features.txt`
- `.../reduced_table.csv`
- `.../report.json`

## 4) Train XGBoost (leakage-safe)

Script:
- `models/vista-3d/scripts/train_xgboost.py`

What it does:
- Uses the existing `split` column
- Checks **patient_id overlap** across splits and errors if any overlap is found
- Trains with early stopping
- Avoids AUC warnings by using `logloss` for early stopping when validation has a single class
- Writes `metrics.json`, `test_predictions.csv`, and gain importances

Command (EGFR smoke-test using selected features):

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
python /home/shadeform/models/vista-3d/scripts/train_xgboost.py \
  --data /home/shadeform/models/vista-3d/outputs/ml/feature_selection/egfr_mutated/reduced_table.csv \
  --label egfr_mutated \
  --task binary \
  --num-boost-round 800 \
  --early-stopping-rounds 50 \
  --out-dir /home/shadeform/models/vista-3d/outputs/ml/models/nsclc_egfr_smoketest
```

Outputs:
- `models/vista-3d/outputs/ml/models/nsclc_egfr_smoketest/metrics.json`
- `models/vista-3d/outputs/ml/models/nsclc_egfr_smoketest/model.json`
- `models/vista-3d/outputs/ml/models/nsclc_egfr_smoketest/test_predictions.csv`
- `models/vista-3d/outputs/ml/models/nsclc_egfr_smoketest/feature_importance_gain.csv`

## Dependencies

Optional ML deps (XGBoost):
- `models/vista-3d/requirements-ml.txt`

Install:

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
pip install -r /home/shadeform/models/vista-3d/requirements-ml.txt
```


