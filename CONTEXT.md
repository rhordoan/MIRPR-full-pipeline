# Project context (Vista3D radiomics → XGBoost)

This file captures the working context to accompany the codebase and help new collaborators understand what is present vs. intentionally excluded from git.

## Scope and goal
- Segment CT series with Vista3D, clean masks, extract radiomics, derive labels, and train XGBoost models for EGFR/KRAS (and optionally Ki-67).
- Datasets handled: TCIA NSCLC Radiogenomics and TCGA-LUAD (imaging + clinical/mutation/expression labels).

## Key scripts (all under `scripts/`)
- Data acquisition: `tcia_download_collection.py`, `download_nsclc_radiogenomics.py`, `download_tcga_luad_tcia.py`, `gdc_tcga_luad_manifest.py`, `install_gdc_client.sh`, `start_downloads_bg.sh`, `stop_downloads_bg.sh`.
- Inference: `run_inference_on_series.py` (mask cleaning, PNG overlays, optional CT save, optional auto-crop to lung z-range, optional AMP), `run_all_inference_bg.sh`, `run_all_inference_parallel.py`.
- Radiomics: `run_pyradiomics_one.py`, `run_pyradiomics_batch.py` (subprocess workers, resume, sanitization), `normalize_radiomics_jsons.py`.
- Labels: `derive_tcga_luad_labels.py`, `derive_nsclc_radiogenomics_labels.py`.
- ML prep & training: `prepare_xgb_dataset.py`, `feature_selection.py`, `train_xgboost.py`, `start_xgb_training_bg.sh`.

## Reference docs
- `implementation plan.md`: original high-level plan.
- `DATASETS.md`: how to download datasets and run inference/radiomics.
- `ML_PIPELINE.md`: end-to-end commands (labels → ML table → feature selection → training) with leakage-safe splits.

## What is intentionally not pushed (large/derived artifacts)
- Model weights: `weights/`, `best_model.zip`, `best_model.pth` (skip).
- Data: `data/` (TCIA/GDC downloads), `outputs/` (masks, PNGs, radiomics JSON/CSV, metrics), `logs/`.
- Virtual envs: `.venvs/` (e.g., `/home/shadeform/.venvs/vista-3d`).

## Current status (at time of capture)
- Weights downloaded and usable locally.
- Inference scripts ready; parallel runner exists.
- Radiomics extraction stabilized (subprocess per case, JSON sanitization).
- Labels derived for NSCLC Radiogenomics and TCGA-LUAD.
- ML prep done; feature selection and XGBoost training scripts working (native xgboost, leakage checks).
- SSH key generated for repo pushes: `/home/shadeform/.ssh/id_ed25519.pub` (added to host manually).

## Next suggested steps
- Add remote (SSH) and push code/docs only (exclude data/outputs/weights/logs).
- If rerunning feature selection/training, consider stratified patient splits for very low positive counts (e.g., EGFR positives).


