#!/usr/bin/env bash
set -euo pipefail

# Installs the GDC data transfer tool (gdc-client) into ./bin/ (relative to repo root).
# This is useful for downloading TCGA files from the Genomic Data Commons.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/bin"
mkdir -p "${BIN_DIR}"

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "${OS}" != "Linux" ]]; then
  echo "This installer currently supports Linux only (detected: ${OS})." >&2
  exit 2
fi

if [[ "${ARCH}" != "x86_64" ]]; then
  echo "This installer expects x86_64 (detected: ${ARCH})." >&2
  exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# NOTE: GDC occasionally revs filenames/versions. If this URL changes, update it.
URL="https://gdc.cancer.gov/system/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip"

echo "Downloading gdc-client from: ${URL}"
curl -L --fail "${URL}" -o "${TMP}/gdc-client.zip"

unzip -o "${TMP}/gdc-client.zip" -d "${TMP}" >/dev/null

FOUND="$(find "${TMP}" -type f -name 'gdc-client' | head -n 1 || true)"
if [[ -z "${FOUND}" ]]; then
  echo "Could not find gdc-client binary in downloaded zip." >&2
  exit 1
fi

install -m 0755 "${FOUND}" "${BIN_DIR}/gdc-client"

echo "Installed: ${BIN_DIR}/gdc-client"
echo "Test with: ${BIN_DIR}/gdc-client --version"




