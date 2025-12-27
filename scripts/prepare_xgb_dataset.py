#!/usr/bin/env python3
"""
Prepare an ML-ready table for XGBoost (or any tabular model) from radiomics outputs.

Inputs:
  - Per-series PyRadiomics JSONs (recommended): outputs/radiomics_per_series/*.json
    (produced by scripts/run_pyradiomics_batch.py)
  - Optional labels CSV (e.g., TCGA-LUAD labels): outputs/tcga_luad_labels_v3.csv

Outputs:
  - A single CSV with:
      series_uid, patient_id, dataset, split, (optional label columns), radiomics feature columns...

Notes:
  - This script can infer (patient_id, dataset) by indexing your TCIA directory:
      data/tcia/<Collection>/<PatientID>/<StudyUID>/<SeriesUID>/
  - If labels CSV contains patient_id, it will be preferred.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SERIES_UID_RE = re.compile(r"^\d+(?:\.\d+)+$")


@dataclass(frozen=True)
class SeriesIndexRow:
    series_uid: str
    patient_id: str
    study_uid: str
    dataset: str  # TCIA collection name
    series_dir: str


def _safe_read_json(fp: str) -> Optional[dict]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_tcia_series_index(tcia_root: str) -> Dict[str, SeriesIndexRow]:
    """
    Build a mapping: series_uid -> (patient_id, dataset, study_uid, series_dir)
    by walking tcia_root and identifying directories that look like SeriesInstanceUIDs.
    """
    idx: Dict[str, SeriesIndexRow] = {}
    tcia_root_p = Path(tcia_root)
    if not tcia_root_p.exists():
        return idx

    # We only care about directory names that match a DICOM UID pattern.
    for dirpath, dirnames, _filenames in os.walk(tcia_root):
        # prune hidden dirs
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        base = os.path.basename(dirpath)
        if not SERIES_UID_RE.match(base):
            continue

        series_uid = base
        rel = Path(dirpath).resolve().relative_to(tcia_root_p.resolve())
        parts = rel.parts
        # Expected: <dataset>/<patient_id>/<study_uid>/<series_uid>
        if len(parts) < 4:
            continue
        dataset, patient_id, study_uid = parts[0], parts[1], parts[2]
        idx[series_uid] = SeriesIndexRow(
            series_uid=series_uid,
            patient_id=patient_id,
            study_uid=study_uid,
            dataset=dataset,
            series_dir=str(Path(dirpath).resolve()),
        )
    return idx


def assign_group_splits(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> pd.Series:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r} for split assignment")

    groups = df[group_col].astype(str)
    uniq = groups.dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # remainder goes to test
    train_g = set(uniq[:n_train])
    val_g = set(uniq[n_train : n_train + n_val])
    test_g = set(uniq[n_train + n_val :])

    def _bucket(g: str) -> str:
        if g in train_g:
            return "train"
        if g in val_g:
            return "val"
        return "test"

    return groups.map(_bucket)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--radiomics-per-series-dir",
        default="/home/shadeform/models/vista-3d/outputs/radiomics_per_series",
        help="Directory containing per-series radiomics JSONs (*.json).",
    )
    ap.add_argument(
        "--labels-csv",
        default=None,
        help="Optional labels CSV to join (must contain series_uid or series_uid column).",
    )
    ap.add_argument(
        "--tcia-root",
        default="/home/shadeform/models/vista-3d/data/tcia",
        help="TCIA root folder used to infer dataset/patient_id by SeriesUID path.",
    )
    ap.add_argument(
        "--out-csv",
        default="/home/shadeform/models/vista-3d/outputs/ml/radiomics_ml_table.csv",
        help="Output CSV path.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument(
        "--group-col",
        default="patient_id",
        help="Group column for leakage-safe splits (default: patient_id).",
    )
    ap.add_argument(
        "--require-label-cols",
        default=None,
        help="Comma-separated label columns that must be present (rows missing these will be dropped).",
    )
    ap.add_argument(
        "--dataset-filter",
        default=None,
        help="Optional filter by dataset/collection name (e.g., 'NSCLC Radiogenomics' or 'TCGA-LUAD').",
    )
    args = ap.parse_args(argv)

    per_dir = Path(args.radiomics_per_series_dir)
    json_paths = sorted(per_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No radiomics JSONs found in {per_dir}")

    rows: List[dict] = []
    for fp in json_paths:
        d = _safe_read_json(str(fp))
        if not d:
            continue
        rows.append(d)

    if not rows:
        raise SystemExit("All per-series JSONs failed to read (0 rows).")

    df = pd.DataFrame(rows)
    # Avoid pathological fragmentation as we clean/coerce many feature columns.
    df = df.copy()
    # Normalize key column name to series_uid
    if "series_uid" not in df.columns and "series" in df.columns:
        df = df.rename(columns={"series": "series_uid"})
    if "series_uid" not in df.columns:
        raise SystemExit("Radiomics JSONs must contain 'series' or 'series_uid' field.")

    # Index TCIA for patient/dataset inference
    idx = build_tcia_series_index(args.tcia_root)
    def _idx_get(series_uid: object) -> Optional[SeriesIndexRow]:
        return idx.get(str(series_uid))

    idx_rows = df["series_uid"].map(_idx_get)
    df = df.assign(
        patient_id_inferred=idx_rows.map(lambda r: r.patient_id if r else None),
        dataset_inferred=idx_rows.map(lambda r: r.dataset if r else None),
        series_dir_inferred=idx_rows.map(lambda r: r.series_dir if r else None),
    )

    # Optional labels join
    if args.labels_csv:
        labels = pd.read_csv(args.labels_csv)
        if "series_uid" not in labels.columns:
            # fallbacks some pipelines use
            if "series" in labels.columns:
                labels = labels.rename(columns={"series": "series_uid"})
            elif "SeriesInstanceUID" in labels.columns:
                labels = labels.rename(columns={"SeriesInstanceUID": "series_uid"})
        if "series_uid" not in labels.columns:
            raise SystemExit(f"Labels CSV must contain 'series_uid' (or 'series'/'SeriesInstanceUID'): {args.labels_csv}")
        df = df.merge(labels, on="series_uid", how="left", suffixes=("", "_label"))

    # Unify patient_id and dataset: prefer label-provided if present
    if "patient_id" not in df.columns:
        df["patient_id"] = df["patient_id_inferred"]
    else:
        df["patient_id"] = df["patient_id"].fillna(df["patient_id_inferred"])
    if "dataset" not in df.columns:
        df["dataset"] = df["dataset_inferred"]
    else:
        df["dataset"] = df["dataset"].fillna(df["dataset_inferred"])

    if args.dataset_filter:
        df = df[df["dataset"].astype(str) == str(args.dataset_filter)].copy()

    # Coerce radiomics feature columns to numeric; keep metadata/labels as-is.
    feature_prefixes = ("original", "wavelet", "log")
    feature_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(feature_prefixes)]
    if feature_cols:
        # Coerce in one shot to reduce fragmentation.
        df.loc[:, feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Replace inf -> nan (xgboost can handle missing; inf can break some flows)
    if feature_cols:
        feat = df[feature_cols].to_numpy(dtype=float, copy=False)
        feat[~np.isfinite(feat)] = np.nan
        df.loc[:, feature_cols] = feat

    # Basic column pruning: drop all-null and constant features (helps stability)
    non_null = df[feature_cols].notna().any(axis=0)
    feature_cols = [c for c in feature_cols if bool(non_null.get(c, False))]
    nunique = df[feature_cols].nunique(dropna=True)
    feature_cols = [c for c in feature_cols if int(nunique.get(c, 0)) > 1]

    # Assign splits (patient-level by default)
    df = df.assign(
        split=assign_group_splits(
        df=df,
        group_col=args.group_col,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        )
    )

    # Optional: require certain label columns for downstream training
    if args.require_label_cols:
        req = [c.strip() for c in args.require_label_cols.split(",") if c.strip()]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise SystemExit(f"Required label columns not present: {missing}")
        df = df.dropna(subset=req).copy()

    # Reorder columns: metadata, (labels), features
    meta_cols = [
        "series_uid",
        "patient_id",
        "dataset",
        "split",
        "mask_path",
        "image_path",
        "series_dir",
        "series_dir_inferred",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in set(meta_cols) and c not in set(feature_cols)]
    df = df[meta_cols + other_cols + feature_cols]

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Wrote ML table: {out_p} (rows={len(df)}, features={len(feature_cols)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


