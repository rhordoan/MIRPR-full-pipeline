#!/usr/bin/env bash
set -euo pipefail

# Starts dataset downloads in the background with logs + pidfiles.
#
# It launches:
# - TCIA imaging: NSCLC Radiogenomics
# - TCIA imaging: TCGA-LUAD
# - GDC manifest generation: TCGA-LUAD expression (fast)
# - Optional: GDC file download (requires bin/gdc-client) if RUN_GDC_DOWNLOAD=1
#
# Logs are written to: <repo>/logs/<timestamp>/*.log
# PIDs are written to: <repo>/logs/<timestamp>/*.pid
# A symlink <repo>/logs/latest -> <repo>/logs/<timestamp> is created for convenience.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="/home/shadeform/.venvs/vista-3d/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Expected venv python not found/executable at: ${VENV_PY}" >&2
  echo "Create it first (see requirements.txt / previous steps)." >&2
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT_DIR}/logs/${TS}"
mkdir -p "${LOG_DIR}"
mkdir -p "${ROOT_DIR}/data"

ln -sfn "${LOG_DIR}" "${ROOT_DIR}/logs/latest"

TCIA_OUT_DIR="${TCIA_OUT_DIR:-${ROOT_DIR}/data/tcia}"
GDC_OUT_DIR="${GDC_OUT_DIR:-${ROOT_DIR}/data/gdc}"

TCIA_WORKERS="${TCIA_WORKERS:-2}"
TCIA_TIMEOUT="${TCIA_TIMEOUT:-180}"
TCIA_RETRIES="${TCIA_RETRIES:-2}"

echo "Log dir: ${LOG_DIR}"
echo "TCIA out: ${TCIA_OUT_DIR}"
echo "GDC out : ${GDC_OUT_DIR}"

start_job() {
  local name="$1"; shift
  local log="${LOG_DIR}/${name}.log"
  local pidfile="${LOG_DIR}/${name}.pid"

  # shellcheck disable=SC2068
  nohup "$@" >"${log}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pidfile}"
  echo "Started ${name}: pid=${pid} log=${log}"
}

# TCIA imaging downloads (long-running)
start_job "tcia_nsclc_radiogenomics" \
  "${VENV_PY}" "${ROOT_DIR}/scripts/download_nsclc_radiogenomics.py" \
    --out-dir "${TCIA_OUT_DIR}" \
    --workers "${TCIA_WORKERS}" \
    --timeout "${TCIA_TIMEOUT}" \
    --retries "${TCIA_RETRIES}"

start_job "tcia_tcga_luad" \
  "${VENV_PY}" "${ROOT_DIR}/scripts/download_tcga_luad_tcia.py" \
    --out-dir "${TCIA_OUT_DIR}" \
    --workers "${TCIA_WORKERS}" \
    --timeout "${TCIA_TIMEOUT}" \
    --retries "${TCIA_RETRIES}"

# GDC manifest generation (fast)
mkdir -p "${GDC_OUT_DIR}"
start_job "gdc_tcga_luad_manifest" \
  "${VENV_PY}" "${ROOT_DIR}/scripts/gdc_tcga_luad_manifest.py" \
    --out-dir "${GDC_OUT_DIR}"

# Optional: download the GDC files (potentially very large)
RUN_GDC_DOWNLOAD="${RUN_GDC_DOWNLOAD:-0}"
GDC_CLIENT="${ROOT_DIR}/bin/gdc-client"
if [[ "${RUN_GDC_DOWNLOAD}" == "1" ]]; then
  if [[ ! -x "${GDC_CLIENT}" ]]; then
    echo "RUN_GDC_DOWNLOAD=1 but gdc-client not found at ${GDC_CLIENT}." >&2
    echo "Install it with: bash ${ROOT_DIR}/scripts/install_gdc_client.sh" >&2
  else
    mkdir -p "${GDC_OUT_DIR}/files"
    start_job "gdc_tcga_luad_download" \
      "${GDC_CLIENT}" download \
        -m "${GDC_OUT_DIR}/manifest.tsv" \
        -d "${GDC_OUT_DIR}/files"
  fi
fi

echo
echo "To monitor:"
echo "  tail -f ${ROOT_DIR}/logs/latest/tcia_nsclc_radiogenomics.log"
echo "  tail -f ${ROOT_DIR}/logs/latest/tcia_tcga_luad.log"
echo "  tail -f ${ROOT_DIR}/logs/latest/gdc_tcga_luad_manifest.log"
echo
echo "To stop:"
echo "  bash ${ROOT_DIR}/scripts/stop_downloads_bg.sh"




