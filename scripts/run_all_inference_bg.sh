#!/usr/bin/env bash
set -euo pipefail

# Run Vista3D inference + cleaning over all DICOM series under a data root.
# Usage:
#   nohup bash scripts/run_all_inference_bg.sh \
#     /home/shadeform/models/vista-3d/data/tcia \
#     /home/shadeform/models/vista-3d/outputs \
#     /home/shadeform/models/vista-3d/outputs/metrics.csv \
#     /home/shadeform/models/vista-3d/outputs/batch_inference_bg.log &
#
# Args (positional, all optional with defaults):
#   $1 DATA_ROOT   (default: /home/shadeform/models/vista-3d/data/tcia)
#   $2 OUT_DIR     (default: /home/shadeform/models/vista-3d/outputs)
#   $3 METRICS_CSV (default: $OUT_DIR/metrics.csv)
#   $4 LOG_FILE    (default: $OUT_DIR/batch_inference_bg.log)
#
# Cleaning defaults: min_vol_mm3=50, closing=1, opening=1, dilate=2. Adjust below if needed.

DATA_ROOT="${1:-/home/shadeform/models/vista-3d/data/tcia}"
OUT_DIR="${2:-/home/shadeform/models/vista-3d/outputs}"
METRICS_CSV="${3:-${OUT_DIR}/metrics.csv}"
LOG_FILE="${4:-${OUT_DIR}/batch_inference_bg.log}"

MIN_VOL_MM3=50
CLOSING_ITERS=1
OPENING_ITERS=1
DILATE_ITERS=2
MAX_ELONGATION=""  # e.g., 8.0 to drop very elongated shapes; leave empty to disable

VENV_PY="/home/shadeform/.venvs/vista-3d/bin/python"

mkdir -p "${OUT_DIR}"
echo "series,series_dir,raw_voxels,raw_mm3,clean_voxels,clean_mm3,min_vol_mm3,closing_iters,opening_iters,dilate_iters,max_elongation" > "${METRICS_CSV}"

SERIES_LIST="$(mktemp)"
trap 'rm -f "${SERIES_LIST}"' EXIT

# Find all series dirs that contain at least one DICOM
find "${DATA_ROOT}" -type f -name "*.dcm" -printf '%h\n' | sort -u > "${SERIES_LIST}"

echo "Found $(wc -l < "${SERIES_LIST}") series under ${DATA_ROOT}"
echo "Writing metrics to ${METRICS_CSV}"
echo "Logging to ${LOG_FILE}"

{
  while IFS= read -r SERIES_DIR; do
    echo "==> Processing ${SERIES_DIR}"
    "${VENV_PY}" /home/shadeform/models/vista-3d/scripts/run_inference_on_series.py \
      --series-dir "${SERIES_DIR}" \
      --out-dir "${OUT_DIR}" \
      --png \
      --clean \
    --save-ct \
      --min-vol-mm3 "${MIN_VOL_MM3}" \
      --closing-iters "${CLOSING_ITERS}" \
      --opening-iters "${OPENING_ITERS}" \
      --dilate-iters "${DILATE_ITERS}" \
      $( [[ -n "${MAX_ELONGATION}" ]] && echo --max-elongation "${MAX_ELONGATION}" ) \
      --metrics-csv "${METRICS_CSV}"
  done < "${SERIES_LIST}"
} >> "${LOG_FILE}" 2>&1

echo "Done. See ${LOG_FILE} and ${METRICS_CSV}"


