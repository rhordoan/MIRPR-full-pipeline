## Datasets: download scripts

This folder contains scripts to download the datasets referenced in `implementation plan.md`:

- **NSCLC Radiogenomics** (TCIA imaging collection)
- **TCGA-LUAD** (TCIA imaging collection + optional GDC gene expression files)

All scripts assume you’re using the venv you already created:

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
```

### Inference (Vista3D) — speed/quality knobs

Script:
- `scripts/run_inference_on_series.py`

Notes:
- Some CT series include many slices with little/no chest (e.g., long coverage, positioning, or extra recon ranges).
  Running 3D sliding-window inference over the entire z-extent wastes a lot of compute.
- `run_inference_on_series.py` now includes an **optional heuristic auto-crop along z** to likely lung-containing slices
  (with a safety margin). If it can’t confidently detect lungs, it falls back to the full scan.
- On NVIDIA GPUs, enabling **AMP** (mixed precision) usually improves throughput.

Recommended command (fast on GPU, safe defaults):

```bash
python /home/shadeform/models/vista-3d/scripts/run_inference_on_series.py \
  --series-dir /path/to/<SeriesUID> \
  --weights /home/shadeform/models/vista-3d/weights/best_model.pth \
  --out-dir /home/shadeform/models/vista-3d/outputs \
  --save-ct \
  --clean \
  --amp
```

Flags:
- `--disable-lung-crop`: disable the z-cropping and run on the full scan (use this if you suspect missed anatomy).
- `--lung-crop-margin-mm`: margin (mm) added above/below the detected lung range (default: 30mm).
- `--lung-crop-min-lung-frac`: how “lung-like” a slice must be to be included (default: 0.12). Lower = more inclusive.
- `--amp`: enable CUDA autocast for inference (ignored on CPU).

### Label coverage sanity check (for deciding single-task vs multi-task DL)

If you’re deciding whether to train a **multi-task** deep model (EGFR/KRAS/MKI67), you should first check how many **labeled patients** you have per task and how much **overlap** exists (patients with all labels).

Use:
- `scripts/label_stats.py`

Example (NSCLC Radiogenomics EGFR/KRAS):

```bash
python scripts/label_stats.py \
  --csv /home/shadeform/models/vista-3d/outputs/nsclc_radiogenomics_labels.csv \
  --label-cols egfr_mutated,kras_mutated \
  --group-col patient_id
```

Example (TCGA-LUAD MKI67 expression proxy):

```bash
python scripts/label_stats.py \
  --csv /home/shadeform/models/vista-3d/outputs/tcga_luad_labels.csv \
  --label-cols mki67_expr \
  --group-col patient_id
```

### 1) NSCLC Radiogenomics (TCIA)

Downloads CT series from TCIA’s public NBIA API and extracts DICOMs into:
`models/vista-3d/data/tcia/NSCLC Radiogenomics/<PatientID>/<StudyUID>/<SeriesUID>/`

```bash
python /home/shadeform/models/vista-3d/scripts/download_nsclc_radiogenomics.py
```

Smoke-test (download only first 3 series):

```bash
python /home/shadeform/models/vista-3d/scripts/download_nsclc_radiogenomics.py --max-series 3
```

### 2) TCGA-LUAD imaging (TCIA)

Downloads CT series from the `TCGA-LUAD` TCIA collection and extracts DICOMs into:
`models/vista-3d/data/tcia/TCGA-LUAD/<PatientID>/<StudyUID>/<SeriesUID>/`

```bash
python /home/shadeform/models/vista-3d/scripts/download_tcga_luad_tcia.py
```

### 3) TCGA-LUAD gene expression (GDC) — optional but recommended for Ki-67 (MKI67)

This creates a GDC manifest (TSV) for **open-access** TCGA-LUAD gene expression quantification files

### 7) XGBoost training (radiomics → ML table → model)

For a complete end-to-end (labels → ML table → feature selection → training) guide, see:
`models/vista-3d/ML_PIPELINE.md`

Once you have per-series radiomics JSONs under:
`models/vista-3d/outputs/radiomics_per_series/*.json`

Create an ML-ready table (adds `patient_id`, `dataset`, and leakage-safe `split` by patient):

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
python /home/shadeform/models/vista-3d/scripts/prepare_xgb_dataset.py \
  --radiomics-per-series-dir /home/shadeform/models/vista-3d/outputs/radiomics_per_series \
  --tcia-root /home/shadeform/models/vista-3d/data/tcia \
  --out-csv /home/shadeform/models/vista-3d/outputs/ml/radiomics_ml_table.csv
```

If you have labels CSV (example: TCGA-LUAD labels), you can join it at prep time:

```bash
python /home/shadeform/models/vista-3d/scripts/prepare_xgb_dataset.py \
  --radiomics-per-series-dir /home/shadeform/models/vista-3d/outputs/radiomics_per_series \
  --labels-csv /home/shadeform/models/vista-3d/outputs/tcga_luad_labels_v3.csv \
  --tcia-root /home/shadeform/models/vista-3d/data/tcia \
  --out-csv /home/shadeform/models/vista-3d/outputs/ml/tcga_luad_radiomics_ml_table.csv \
  --dataset-filter 'TCGA-LUAD'
```

Train an XGBoost model (example: EGFR mutation status):

```bash
python /home/shadeform/models/vista-3d/scripts/train_xgboost.py \
  --data /home/shadeform/models/vista-3d/outputs/ml/radiomics_ml_table.csv \
  --label egfr_mutated \
  --out-dir /home/shadeform/models/vista-3d/outputs/ml/models/egfr_mutated
```

Run training in the background with logs:

```bash
bash /home/shadeform/models/vista-3d/scripts/start_xgb_training_bg.sh \
  --data /home/shadeform/models/vista-3d/outputs/ml/radiomics_ml_table.csv \
  --label egfr_mutated \
  --out-dir /home/shadeform/models/vista-3d/outputs/ml/models/egfr_mutated
```

### 8) NSCLC Radiogenomics labels (EGFR/KRAS + clinical)

TCIA provides an official “data labels” CSV with EGFR/KRAS mutation status and clinical fields.
This script downloads + converts it to a **series-level** CSV (joinable to your radiomics by `series_uid`):

```bash
source /home/shadeform/.venvs/vista-3d/bin/activate
python /home/shadeform/models/vista-3d/scripts/derive_nsclc_radiogenomics_labels.py \
  --tcia-root /home/shadeform/models/vista-3d/data/tcia \
  --dataset 'NSCLC Radiogenomics' \
  --out-csv /home/shadeform/models/vista-3d/outputs/nsclc_radiogenomics_labels.csv
```

Optional: restrict to only series already segmented (those with `*_mask_clean.nii.gz` in outputs):

```bash
python /home/shadeform/models/vista-3d/scripts/derive_nsclc_radiogenomics_labels.py \
  --tcia-root /home/shadeform/models/vista-3d/data/tcia \
  --dataset 'NSCLC Radiogenomics' \
  --outputs-dir /home/shadeform/models/vista-3d/outputs \
  --out-csv /home/shadeform/models/vista-3d/outputs/nsclc_radiogenomics_labels.csv
```
(default: `Gene Expression Quantification` + `STAR - Counts`).

```bash
mkdir -p /home/shadeform/models/vista-3d/data/gdc
python /home/shadeform/models/vista-3d/scripts/gdc_tcga_luad_manifest.py --out-dir /home/shadeform/models/vista-3d/data/gdc
```

Install `gdc-client` (optional helper):

```bash
bash /home/shadeform/models/vista-3d/scripts/install_gdc_client.sh
```

Download the files (uses the manifest created above):

```bash
/home/shadeform/models/vista-3d/bin/gdc-client download -m /home/shadeform/models/vista-3d/data/gdc/manifest.tsv -d /home/shadeform/models/vista-3d/data/gdc/files
```

If you’re downloading *controlled* file types, you will need a token:

```bash
/home/shadeform/models/vista-3d/bin/gdc-client download -t /path/to/gdc_token.txt -m /home/shadeform/models/vista-3d/data/gdc/manifest.tsv -d /home/shadeform/models/vista-3d/data/gdc/files
```

### 4) TCGA-LUAD labels (clinical + optional MKI67 + optional EGFR/KRAS)

Important: **TCIA imaging alone does not include biomarker labels** (EGFR/KRAS/Ki-67).

This script creates a **series-level** labels table (so it can be merged with your
per-series inference outputs and per-series radiomics JSONs):

- **Clinical + smoking + survival + stage**: from the locally cached TCGA clinical “patient” table (nationwidechildrens export)
- **EGFR/KRAS flags (when available)**: also from that clinical table
- **MKI67 expression (Ki-67 proxy)**: optional, if you downloaded STAR-counts expression files from GDC
- **Mutation flags from MAF**: optional, if you provide `--maf`

Run:

```bash
python /home/shadeform/models/vista-3d/scripts/derive_tcga_luad_labels.py \
  --tcia-collection-dir /home/shadeform/models/vista-3d/data/tcia/TCGA-LUAD \
  --out-csv /home/shadeform/models/vista-3d/outputs/tcga_luad_labels.csv
```

If you only want labels for the subset you already ran inference on, pass `--outputs-dir`:

```bash
python /home/shadeform/models/vista-3d/scripts/derive_tcga_luad_labels.py \
  --tcia-collection-dir /home/shadeform/models/vista-3d/data/tcia/TCGA-LUAD \
  --outputs-dir /home/shadeform/models/vista-3d/outputs \
  --out-csv /home/shadeform/models/vista-3d/outputs/tcga_luad_labels_segmented_only.csv
```

Note: `--outputs-dir` filters by series UIDs present in files like `*_ct_resampled.nii.gz` / `*_mask_clean.nii.gz`.
So make sure `outputs/` corresponds to the TCGA-LUAD inference run you want to label.

### Start everything in the background (logs + pidfiles)

This launches:
- TCIA imaging download for **NSCLC Radiogenomics**
- TCIA imaging download for **TCGA-LUAD**
- GDC manifest generation for **TCGA-LUAD** expression (fast)

```bash
bash /home/shadeform/models/vista-3d/scripts/start_downloads_bg.sh
```

Logs will be in:
`/home/shadeform/models/vista-3d/logs/latest/*.log`

Monitor:

```bash
tail -f /home/shadeform/models/vista-3d/logs/latest/tcia_nsclc_radiogenomics.log
```

Stop all started jobs:

```bash
bash /home/shadeform/models/vista-3d/scripts/stop_downloads_bg.sh
```


