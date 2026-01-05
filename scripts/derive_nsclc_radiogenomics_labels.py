#!/usr/bin/env python3
"""
Derive labels for the TCIA NSCLC Radiogenomics collection.

Source of truth (TCIA "Data Labels" CSV):
  https://www.cancerimagingarchive.net/wp-content/uploads/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv

This script:
  - Downloads the labels CSV (optional/auto)
  - Parses clinical/genomic columns (EGFR/KRAS mutation status, survival, TNM, etc.)
  - Joins labels to your local TCIA series directories to produce SERIES-level labels:
      series_uid, patient_id, dataset, ... labels ...

Why series-level?
  You extract radiomics per DICOM series (SeriesInstanceUID), so labels must align at that granularity.
  For this collection, labels are patient-level; we replicate them to each series for that patient.

Notes:
  - NSCLC Radiogenomics does NOT include Ki-67 (MKI67) in this labels CSV.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_LABELS_URL = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/"
    "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
)

SERIES_UID_RE = re.compile(r"^\d+(?:\.\d+)+$")


def build_tcia_series_index(tcia_root: str) -> pd.DataFrame:
    """
    Build a table of series under:
      <tcia_root>/<Collection>/<PatientID>/<StudyUID>/<SeriesUID>/
    """
    cols = ["series_uid", "patient_id", "study_uid", "dataset", "series_dir"]
    rows: List[dict] = []
    tcia_root_p = Path(tcia_root).resolve()
    if not tcia_root_p.exists():
        return pd.DataFrame(rows)

    for dirpath, dirnames, _filenames in os.walk(str(tcia_root_p)):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        base = os.path.basename(dirpath)
        if not SERIES_UID_RE.match(base):
            continue
        # TCIA downloader writes <series_dir>/.done when extraction is complete.
        # This lets us safely run labels/radiomics while downloads are still in progress.
        if not os.path.exists(os.path.join(dirpath, ".done")):
            continue
        series_uid = base
        rel = Path(dirpath).resolve().relative_to(tcia_root_p)
        parts = rel.parts
        # expected: <dataset>/<patient_id>/<study_uid>/<series_uid>
        if len(parts) < 4:
            continue
        dataset, patient_id, study_uid = parts[0], parts[1], parts[2]
        rows.append(
            dict(
                series_uid=series_uid,
                patient_id=patient_id,
                study_uid=study_uid,
                dataset=dataset,
                series_dir=str(Path(dirpath).resolve()),
            )
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["series_uid"])


def _norm_str(x: object) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    return s


def _mut_to_bool(x: object) -> Optional[bool]:
    s = _norm_str(x)
    if not s:
        return None
    sl = s.strip().lower()
    if sl in {"mutant", "mutated", "yes", "true", "1"}:
        return True
    if sl in {"wildtype", "wild-type", "wild type", "no", "false", "0"}:
        return False
    # Not collected / Not assessed / Not recorded
    return None


def ensure_labels_csv(local_path: str, url: str) -> str:
    """
    Ensure labels CSV exists at local_path. Downloads with curl if missing.
    """
    lp = Path(local_path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    if lp.exists() and lp.stat().st_size > 0:
        return str(lp)

    import subprocess

    print(f"Downloading NSCLC Radiogenomics labels CSV -> {lp}", flush=True)
    cmd = ["curl", "-L", "-o", str(lp), url]
    proc = subprocess.run(cmd)
    if proc.returncode != 0 or not lp.exists() or lp.stat().st_size == 0:
        raise RuntimeError(f"Failed to download labels CSV from {url}")
    return str(lp)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcia-root", default="/home/shadeform/models/vista-3d/data/tcia")
    ap.add_argument(
        "--dataset",
        default="NSCLC Radiogenomics",
        help="TCIA collection name as it appears under data/tcia/",
    )
    ap.add_argument("--labels-url", default=DEFAULT_LABELS_URL)
    ap.add_argument(
        "--labels-csv",
        default="/home/shadeform/models/vista-3d/data/labels/nsclc_radiogenomics/nsclc_radiogenomics_labels.csv",
        help="Local path for cached labels CSV (downloaded if missing).",
    )
    ap.add_argument(
        "--outputs-dir",
        default=None,
        help="Optional: filter to only series that exist in this outputs dir (e.g., vista-3d/outputs).",
    )
    ap.add_argument(
        "--out-csv",
        default="/home/shadeform/models/vista-3d/outputs/nsclc_radiogenomics_labels.csv",
        help="Output series-level labels CSV.",
    )
    args = ap.parse_args(argv)

    labels_csv = ensure_labels_csv(args.labels_csv, args.labels_url)
    raw = pd.read_csv(labels_csv)

    # Normalize column names a bit (keep originals too for traceability)
    # Canonical keys
    if "Case ID" not in raw.columns:
        raise SystemExit(f"Expected 'Case ID' column in labels CSV: {labels_csv}")

    df = raw.copy()
    df["patient_id"] = df["Case ID"].astype(str).str.strip()
    df["egfr_mutated"] = df["EGFR mutation status"].map(_mut_to_bool) if "EGFR mutation status" in df.columns else None
    df["kras_mutated"] = df["KRAS mutation status"].map(_mut_to_bool) if "KRAS mutation status" in df.columns else None
    df["survival_status"] = df["Survival Status"].map(_norm_str) if "Survival Status" in df.columns else None

    # Keep a curated subset + some useful clinical fields
    keep = [
        "patient_id",
        "Patient affiliation",
        "Age at Histological Diagnosis",
        "Weight (lbs)",
        "Gender",
        "Ethnicity",
        "Smoking status",
        "Pack Years",
        "Quit Smoking Year",
        "Histology ",
        "Pathological T stage",
        "Pathological N stage",
        "Pathological M stage",
        "Histopathological Grade",
        "Lymphovascular invasion",
        "Pleural invasion (elastic, visceral, or parietal)",
        "Adjuvant Treatment",
        "Chemotherapy",
        "Radiation",
        "Recurrence",
        "Recurrence Location",
        "Date of Recurrence",
        "Date of Last Known Alive",
        "Survival Status",
        "Date of Death",
        "Time to Death (days)",
        "CT Date",
        "Days between CT and surgery",
        "PET Date",
        "egfr_mutated",
        "kras_mutated",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Build series index for the dataset and join by patient_id
    idx = build_tcia_series_index(args.tcia_root)
    idx = idx[idx["dataset"].astype(str) == str(args.dataset)].copy()
    if len(idx) == 0:
        raise SystemExit(f"No series found under {args.tcia_root}/{args.dataset}. Have you downloaded the images?")

    if args.outputs_dir:
        outdir = Path(args.outputs_dir)
        # Filter to series that were actually processed (presence of mask_clean)
        # If outputs_dir doesn't have masks, we keep all.
        series_have = set()
        for p in outdir.glob("*_mask_clean.nii.gz"):
            s = p.name.replace("_mask_clean.nii.gz", "")
            if SERIES_UID_RE.match(s):
                series_have.add(s)
        if series_have:
            idx = idx[idx["series_uid"].isin(series_have)].copy()

    merged = idx.merge(df, on="patient_id", how="left", suffixes=("", "_label"))
    merged.to_csv(args.out_csv, index=False)
    print(f"Wrote NSCLC Radiogenomics series labels -> {args.out_csv} (rows={len(merged)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


