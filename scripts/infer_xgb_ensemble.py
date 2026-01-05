#!/usr/bin/env python3
"""
Infer with an XGBoost ensemble (probability averaging) and a fixed threshold.

Usage:
  python3 scripts/infer_xgb_ensemble.py \
    --label egfr_mutated \
    --models outputs/ml/models/final_ensembles/egfr_mutated/model_0.json \
             outputs/ml/models/final_ensembles/egfr_mutated/model_1.json \
             outputs/ml/models/final_ensembles/egfr_mutated/model_2.json \
    --threshold 0.3894757300615311 \
    --features-file outputs/ml/models/final_ensembles/egfr_mutated/metadata.json \
    --input-csv /path/to/radiomics_table.csv \
    --out-csv /path/to/preds.csv

Notes:
  - Expects the input CSV to contain the radiomics feature columns listed in the metadata file.
  - Missing/non-numeric values in features are coerced to numeric and filled with 0.
  - Output columns: series (if present), prob, pred (binary), threshold.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb


def load_models(model_paths: List[str]) -> List[xgb.Booster]:
    models: List[xgb.Booster] = []
    for p in model_paths:
        bst = xgb.Booster()
        bst.load_model(p)
        models.append(bst)
    return models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Label name (for metadata tracking).")
    ap.add_argument("--models", nargs="+", required=True, help="List of XGBoost model JSON paths.")
    ap.add_argument("--threshold", type=float, required=True, help="Decision threshold to binarize probabilities.")
    ap.add_argument("--features-file", required=True, help="Path to metadata JSON containing feature list.")
    ap.add_argument("--input-csv", required=True, help="Input CSV with radiomics features.")
    ap.add_argument("--out-csv", required=True, help="Output CSV for predictions.")
    args = ap.parse_args()

    with open(args.features_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_cols = meta.get("features")
    if not feat_cols:
        raise SystemExit("features not found in metadata file.")

    df = pd.read_csv(args.input_csv)
    if not set(feat_cols).issubset(df.columns):
        missing = sorted(set(feat_cols) - set(df.columns))
        raise SystemExit(f"Missing required feature columns: {missing}")

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    dmat = xgb.DMatrix(X, feature_names=feat_cols)

    models = load_models(args.models)
    probs_list = [m.predict(dmat) for m in models]
    probs = np.mean(np.vstack(probs_list), axis=0)
    preds = (probs >= args.threshold).astype(int)

    out_df = pd.DataFrame({"prob": probs, "pred": preds})
    if "series" in df.columns:
        out_df.insert(0, "series", df["series"])
    out_df["threshold_used"] = args.threshold
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(
        json.dumps(
            {
                "label": args.label,
                "input_rows": len(df),
                "models": args.models,
                "threshold": args.threshold,
                "out_csv": args.out_csv,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









