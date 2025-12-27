#!/usr/bin/env bash
set -euo pipefail

# Stops background dataset download jobs started by scripts/start_downloads_bg.sh
# using pidfiles in <repo>/logs/latest/*.pid.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST="${ROOT_DIR}/logs/latest"

if [[ ! -d "${LATEST}" ]]; then
  echo "No logs/latest directory found at: ${LATEST}" >&2
  exit 1
fi

shopt -s nullglob
PIDS=( "${LATEST}"/*.pid )
if (( ${#PIDS[@]} == 0 )); then
  echo "No pidfiles found in: ${LATEST}" >&2
  exit 0
fi

for pf in "${PIDS[@]}"; do
  name="$(basename "${pf}" .pid)"
  pid="$(cat "${pf}" || true)"
  if [[ -z "${pid}" ]]; then
    echo "Skipping ${name}: empty pidfile"
    continue
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${name}: pid=${pid}"
    kill "${pid}" || true
  else
    echo "Not running ${name}: pid=${pid}"
  fi
done




