#!/usr/bin/env python3
"""
Build a unified radiomics table across Radiogenomics, TCGA-LUAD, and the private cohort.

Outputs (default: outputs/unified_radiomics/):
  - unified_table.csv          Combined rows with shared features + labels + group_id + fold.
  - feature_columns.txt        Shared feature list (intersection across datasets).
  - splits.csv                 group_kfold assignment (series, group_id, fold, source, patient_id).
  - tcga_missing_series.txt    Series present in TCGA labels but missing features (failed radiomics).

Assumptions:
  - Feature CSVs share `series`, `mask_path`, `image_path` columns and radiomics_* timing columns.
  - Radiogenomics labels CSV contains columns: series_uid, patient_id, egfr_mutated, kras_mutated.
  - TCGA labels CSV contains series_uid, patient_id, egfr_mutated, kras_mutated (from MAF).
  - Private manifest is used only to detect presence of a patient for weak labels; EGFR/KRAS remain None.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupKFold


META_COLS = {
    "series",
    "mask_path",
    "image_path",
    "radiomics_init_seconds",
    "radiomics_execute_seconds",
    "radiomics_total_seconds",
    "radiomics_status",
    "radiomics_error_type",
    "radiomics_error",
}


def _load_features(path: str, shared_features: Optional[list[str]] = None, meta_keep: Optional[list[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if shared_features is None:
        return df
    cols = (meta_keep or []) + shared_features
    return df[cols].copy()


def _shared_feature_list(dfs: list[pd.DataFrame]) -> list[str]:
    feat_sets = []
    for df in dfs:
        feat_sets.append(set(df.columns) - META_COLS)
    shared = set.intersection(*feat_sets)
    return sorted(shared)


def _parse_private_patient_id(mask_or_img_path: str) -> Optional[str]:
    # Expect path contains .../private_infer_ct/<patient_code>__<series_uid>/...
    m = re.search(r"/private_infer_ct/([^/_]+)__", str(mask_or_img_path))
    return m.group(1) if m else None


def build_unified(
    rad_feats_path: str,
    rad_labels_path: str,
    tcga_feats_path: str,
    tcga_labels_path: str,
    priv_feats_path: str,
    priv_manifest_path: str,
    out_dir: str,
    weak_weight: float = 0.5,
    n_splits: int = 5,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    rad_df = pd.read_csv(rad_feats_path)
    tcga_df = pd.read_csv(tcga_feats_path)
    priv_df = pd.read_csv(priv_feats_path)

    meta_keep = [c for c in ["series", "mask_path", "image_path"] if c in rad_df.columns]
    shared_features = _shared_feature_list([rad_df, tcga_df, priv_df])

    rad_feats = _load_features(rad_feats_path, shared_features, meta_keep)
    tcga_feats = _load_features(tcga_feats_path, shared_features, meta_keep)
    priv_feats = _load_features(priv_feats_path, shared_features, meta_keep)

    # Labels
    rad_labels = pd.read_csv(rad_labels_path)
    tcga_labels = pd.read_csv(tcga_labels_path)
    priv_manifest = pd.read_csv(priv_manifest_path)
    priv_labels_idx = priv_manifest.set_index("patient_code")

    # Radiogenomics
    rad_merge = rad_feats.merge(rad_labels, left_on="series", right_on="series_uid", how="left")
    rad_subset = rad_feats.copy()
    rad_subset["source"] = "radiogenomics"
    rad_subset["patient_id"] = rad_merge["patient_id"]
    rad_subset["egfr_mutated"] = rad_merge.get("egfr_mutated")
    rad_subset["kras_mutated"] = rad_merge.get("kras_mutated")
    rad_subset["egfr_mutated_weak"] = None
    rad_subset["kras_mutated_weak"] = None
    rad_subset["weak_weight"] = None

    # TCGA
    tcga_merge = tcga_feats.merge(tcga_labels, left_on="series", right_on="series_uid", how="left")
    tcga_subset = tcga_feats.copy()
    tcga_subset["source"] = "tcga"
    tcga_subset["patient_id"] = tcga_merge.get("patient_id")
    tcga_subset["egfr_mutated"] = tcga_merge.get("egfr_mutated")
    tcga_subset["kras_mutated"] = tcga_merge.get("kras_mutated")
    tcga_subset["egfr_mutated_weak"] = None
    tcga_subset["kras_mutated_weak"] = None
    tcga_subset["weak_weight"] = None

    # Private
    patient_ids = [_parse_private_patient_id(p) for p in priv_feats.get("mask_path", priv_feats.get("image_path"))]
    priv_feats["patient_id"] = patient_ids
    weak_weights = []
    for pid in priv_feats["patient_id"]:
        weak_weights.append(weak_weight if pid in priv_labels_idx.index else None)
    priv_feats["source"] = "private"
    priv_feats["egfr_mutated"] = None
    priv_feats["kras_mutated"] = None
    priv_feats["egfr_mutated_weak"] = None
    priv_feats["kras_mutated_weak"] = None
    priv_feats["weak_weight"] = weak_weights

    feature_cols = shared_features
    meta_cols_out = meta_keep
    label_cols = ["egfr_mutated", "kras_mutated", "egfr_mutated_weak", "kras_mutated_weak", "weak_weight"]
    other_cols = ["patient_id", "source"]
    ordered_cols = meta_cols_out + feature_cols + other_cols + label_cols

    unified = pd.concat(
        [
            rad_subset[ordered_cols],
            tcga_subset[ordered_cols],
            priv_feats[ordered_cols],
        ],
        ignore_index=True,
    )

    unified["group_id"] = unified.apply(
        lambda r: f"{r['source']}:{r['patient_id'] if pd.notnull(r['patient_id']) else r['series']}", axis=1
    )

    # GroupKFold splits (skip if only 1 group)
    unique_groups = unified["group_id"].unique()
    folds = [0] * len(unified)
    if len(unique_groups) >= 2:
        splits = min(n_splits, len(unique_groups))
        gkf = GroupKFold(n_splits=splits)
        for fold_idx, (_, val_idx) in enumerate(gkf.split(unified, groups=unified["group_id"])):
            for i in val_idx:
                folds[i] = fold_idx
    unified["fold"] = folds

    # Save outputs
    unified_path = os.path.join(out_dir, "unified_table.csv")
    unified.to_csv(unified_path, index=False)

    features_path = os.path.join(out_dir, "feature_columns.txt")
    with open(features_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_cols))

    splits_path = os.path.join(out_dir, "splits.csv")
    unified[["series", "group_id", "fold", "source", "patient_id"]].to_csv(splits_path, index=False)

    # Note missing TCGA series present in labels but absent in features
    tcga_in_feats = set(tcga_feats["series"].unique())
    tcga_label_series = set(tcga_labels["series_uid"].unique())
    missing_tcga = sorted(list(tcga_label_series - tcga_in_feats))
    missing_path = os.path.join(out_dir, "tcga_missing_series.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for s in missing_tcga:
            f.write(f"{s}\n")

    return {
        "unified_rows": len(unified),
        "features": len(feature_cols),
        "folds": len(set(folds)),
        "missing_tcga": len(missing_tcga),
        "unified_path": unified_path,
        "splits_path": splits_path,
        "features_path": features_path,
        "missing_tcga_path": missing_path,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rad-features", required=True, help="Radiogenomics radiomics features CSV")
    ap.add_argument("--rad-labels", required=True, help="Radiogenomics labels CSV (series_uid, egfr_mutated, kras_mutated)")
    ap.add_argument("--tcga-features", required=True, help="TCGA radiomics features CSV")
    ap.add_argument("--tcga-labels", required=True, help="TCGA labels CSV (series_uid, patient_id, egfr_mutated, kras_mutated)")
    ap.add_argument("--priv-features", required=True, help="Private cohort radiomics features CSV")
    ap.add_argument("--priv-manifest", required=True, help="Private manifest CSV (patient_code to detect presence for weak labels)")
    ap.add_argument("--out-dir", default="outputs/unified_radiomics", help="Output directory (default: outputs/unified_radiomics)")
    ap.add_argument("--weak-weight", type=float, default=0.5, help="Sample weight for private weak labels (default: 0.5)")
    ap.add_argument("--folds", type=int, default=5, help="GroupKFold splits (default: 5)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stats = build_unified(
        rad_feats_path=args.rad_features,
        rad_labels_path=args.rad_labels,
        tcga_feats_path=args.tcga_features,
        tcga_labels_path=args.tcga_labels,
        priv_feats_path=args.priv_features,
        priv_manifest_path=args.priv_manifest,
        out_dir=args.out_dir,
        weak_weight=args.weak_weight,
        n_splits=args.folds,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









