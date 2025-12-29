#!/usr/bin/env python3
"""
Compute per-task label counts and overlaps to decide whether multitask training is viable.

Examples:
  python scripts/label_stats.py --csv outputs/nsclc_radiogenomics_labels.csv --label-cols egfr_mutated,kras_mutated
  python scripts/label_stats.py --csv outputs/tcga_luad_labels.csv --label-cols mki67_expr --group-col patient_id

You can also point it at the ML table produced by prepare_xgb_dataset.py (it will just treat it as a CSV):
  python scripts/label_stats.py --csv outputs/ml/nsclc_radiomics_ml_table_labeled.csv --label-cols egfr_mutated,kras_mutated
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _as_bool_series(s: pd.Series) -> Optional[pd.Series]:
    """
    Convert common binary encodings to boolean.
    Returns None if it doesn't look binary.
    """
    if s.dtype == bool:
        return s
    # Try numeric 0/1
    s_num = pd.to_numeric(s, errors="coerce")
    uniq = pd.unique(s_num.dropna())
    if len(uniq) <= 2 and set(map(float, uniq)).issubset({0.0, 1.0}):
        return s_num.astype("Int64").astype(bool)
    # Try string-ish True/False/0/1
    s_str = s.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    mapped = s_str.map(mapping)
    if mapped.notna().any():
        uniq2 = pd.unique(mapped.dropna())
        if len(uniq2) <= 2:
            return mapped.astype("boolean")
    return None


@dataclass(frozen=True)
class LabelSummary:
    name: str
    n_rows: int
    n_labeled_rows: int
    n_groups: Optional[int]
    n_labeled_groups: Optional[int]
    kind: str  # "binary" or "regression"
    positives_rows: Optional[int] = None
    positives_groups: Optional[int] = None
    regression_stats: Optional[Dict[str, float]] = None


def summarize_label(df: pd.DataFrame, label: str, group_col: Optional[str]) -> LabelSummary:
    if label not in df.columns:
        raise SystemExit(f"Label column not found: {label!r}")

    n_rows = int(len(df))
    y = df[label]
    labeled_mask = y.notna()
    n_labeled_rows = int(labeled_mask.sum())

    n_groups = None
    n_labeled_groups = None
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str)
        n_groups = int(groups.nunique(dropna=True))
        n_labeled_groups = int(groups[labeled_mask].nunique(dropna=True))

    y_bool = _as_bool_series(y)
    if y_bool is not None:
        # binary
        pos_rows = int((y_bool[labeled_mask] == True).sum())  # noqa: E712
        pos_groups = None
        if group_col and group_col in df.columns:
            groups = df[group_col].astype(str)
            # positive if any row in group is positive
            gdf = pd.DataFrame({"g": groups, "y": y_bool})
            gpos = gdf[labeled_mask].groupby("g")["y"].max()
            pos_groups = int((gpos == True).sum())  # noqa: E712
        return LabelSummary(
            name=label,
            n_rows=n_rows,
            n_labeled_rows=n_labeled_rows,
            n_groups=n_groups,
            n_labeled_groups=n_labeled_groups,
            kind="binary",
            positives_rows=pos_rows,
            positives_groups=pos_groups,
        )

    # regression-ish
    y_num = pd.to_numeric(y, errors="coerce")
    labeled_mask = y_num.notna()
    n_labeled_rows = int(labeled_mask.sum())
    stats = None
    if n_labeled_rows > 0:
        v = y_num[labeled_mask].to_numpy(dtype=float, copy=False)
        stats = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "p25": float(np.quantile(v, 0.25)),
            "median": float(np.quantile(v, 0.50)),
            "p75": float(np.quantile(v, 0.75)),
            "max": float(np.max(v)),
        }
    # recompute group counts for regression based on numeric labeled mask
    n_labeled_groups = None
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str)
        n_labeled_groups = int(groups[labeled_mask].nunique(dropna=True))
    return LabelSummary(
        name=label,
        n_rows=n_rows,
        n_labeled_rows=n_labeled_rows,
        n_groups=n_groups,
        n_labeled_groups=n_labeled_groups,
        kind="regression",
        regression_stats=stats,
    )


def overlap_table(df: pd.DataFrame, label_cols: List[str], group_col: Optional[str]) -> Dict[str, object]:
    for c in label_cols:
        if c not in df.columns:
            raise SystemExit(f"Label column not found: {c!r}")
    mask_all = df[label_cols].notna().all(axis=1)
    out: Dict[str, object] = {
        "rows_with_all_labels": int(mask_all.sum()),
        "rows_total": int(len(df)),
    }
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str)
        out["groups_total"] = int(groups.nunique(dropna=True))
        out["groups_with_all_labels"] = int(groups[mask_all].nunique(dropna=True))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to labels CSV or ML table CSV")
    ap.add_argument(
        "--label-cols",
        required=True,
        help="Comma-separated label columns to summarize (e.g., egfr_mutated,kras_mutated,mki67_expr)",
    )
    ap.add_argument(
        "--group-col",
        default="patient_id",
        help="Group column for patient-level counts/overlap (default: patient_id). If missing, group stats are omitted.",
    )
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]
    if not label_cols:
        raise SystemExit("No --label-cols provided.")

    group_col = str(args.group_col) if args.group_col else None
    if group_col and group_col not in df.columns:
        group_col = None

    summaries = [summarize_label(df, c, group_col) for c in label_cols]
    overlap = overlap_table(df, label_cols, group_col)

    out = {
        "csv": args.csv,
        "group_col_used": group_col,
        "labels": [s.__dict__ for s in summaries],
        "overlap": overlap,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


