#!/usr/bin/env python3
"""
Feature selection for radiomics -> XGBoost.

Pipeline (train split only, leakage-safe if your CSV has `split`):
  1) Select radiomics features by prefix: original*, wavelet*, log*
  2) Drop features with too much missingness (train-only)
  3) Drop constant features (train-only)
  4) Univariate ranking (abs Pearson corr with label; binary uses 0/1)
  5) Correlation pruning (keep highest-scoring feature within corr clusters)
  6) Optional: train XGBoost and rank by gain importance; output top-K

Outputs:
  - <out-dir>/selected_features.txt
  - <out-dir>/selected_features.json
  - <out-dir>/reduced_table.csv  (same rows, only metadata + label + selected features)
  - <out-dir>/report.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _import_xgboost():
    try:
        import xgboost as xgb  # type: ignore

        return xgb
    except Exception:
        return None


def _to_binary(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    # handle strings like "True"/"False"
    if y.dtype == object:
        yl = y.astype(str).str.strip().str.lower()
        m = yl.map({"true": 1, "false": 0, "1": 1, "0": 0})
        if m.notna().any():
            y2 = pd.to_numeric(m, errors="coerce")
            return y2
    return pd.to_numeric(y, errors="coerce")


def _feature_columns(df: pd.DataFrame, prefixes: Tuple[str, ...]) -> List[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith(prefixes)]


def _train_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        tr = df[df["split"] == "train"].copy()
        va = df[df["split"] == "val"].copy()
        te = df[df["split"] == "test"].copy()
        return tr, va, te
    # fallback: all train
    return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()


def _pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size == 0:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(abs(c))


def _univariate_scores(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Score each feature by abs Pearson correlation with label.
    NaNs are imputed with median per-feature (train-only).
    """
    yv = y.to_numpy(dtype=float, copy=False)
    scores: Dict[str, float] = {}
    # median impute for correlation computation
    med = X.median(axis=0, numeric_only=True)
    for c in X.columns:
        xv = X[c].to_numpy(dtype=float, copy=False)
        xv = xv.copy()
        # impute NaN
        m = float(med.get(c, np.nan))
        if np.isnan(m):
            # if column all-NaN, score is NaN
            scores[c] = float("nan")
            continue
        xv[~np.isfinite(xv)] = np.nan
        xv[np.isnan(xv)] = m
        scores[c] = _pearson_abs(xv, yv)
    return scores


def _corr_prune(
    X: pd.DataFrame,
    scores: Dict[str, float],
    corr_thresh: float,
) -> List[str]:
    """
    Greedy correlation pruning:
      - Sort by score desc
      - Keep a feature, drop any remaining with |corr| >= thresh to kept feature
    Uses median-imputed values for stable correlations.
    """
    feats = list(X.columns)
    # replace NaN/inf with medians for correlation calculations
    med = X.median(axis=0, numeric_only=True)
    Xm = X.copy()
    for c in feats:
        v = Xm[c].to_numpy(dtype=float, copy=False)
        v = v.copy()
        v[~np.isfinite(v)] = np.nan
        m = float(med.get(c, np.nan))
        if np.isnan(m):
            # all-NaN: fill 0
            m = 0.0
        v[np.isnan(v)] = m
        Xm.loc[:, c] = v

    # precompute correlation matrix (abs)
    C = np.corrcoef(Xm.to_numpy(dtype=float, copy=False), rowvar=False)
    C = np.abs(C)
    np.fill_diagonal(C, 0.0)

    # score order (nan -> -inf)
    svals = np.array([scores.get(c, float("nan")) for c in feats], dtype=float)
    svals = np.where(np.isfinite(svals), svals, -np.inf)
    order = np.argsort(-svals)

    kept: List[int] = []
    dropped = np.zeros(len(feats), dtype=bool)
    for i in order:
        if dropped[i]:
            continue
        kept.append(i)
        # drop all features highly correlated with i
        hi = C[i, :] >= corr_thresh
        dropped |= hi
        dropped[i] = False

    return [feats[i] for i in kept]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Labeled ML table CSV (radiomics + labels).")
    ap.add_argument("--label", required=True, help="Target column to select features for.")
    ap.add_argument("--task", default="auto", choices=["auto", "binary", "regression"])
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--feature-prefixes", default="original,wavelet,log")
    ap.add_argument("--max-missing-frac", type=float, default=0.2, help="Drop features with > this missing frac on train.")
    ap.add_argument("--topk-univariate", type=int, default=300, help="Keep top-K by univariate score before corr pruning.")
    ap.add_argument("--corr-thresh", type=float, default=0.95, help="Correlation pruning threshold (abs Pearson).")
    ap.add_argument("--topk-final", type=int, default=50, help="Final number of selected features to output.")
    ap.add_argument("--use-xgb", action="store_true", help="Use XGBoost gain importance for final ranking (recommended).")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        raise SystemExit(f"Label column not found: {args.label!r}")

    prefixes = tuple(p.strip() for p in args.feature_prefixes.split(",") if p.strip())
    feat_cols = _feature_columns(df, prefixes)
    if not feat_cols:
        raise SystemExit(f"No radiomics feature columns found with prefixes={prefixes}")

    # Split
    tr, va, te = _train_val_test(df)
    if len(tr) == 0:
        raise SystemExit("Train split has 0 rows.")

    # Drop rows with missing label (train-only used for feature selection)
    tr = tr[tr[args.label].notna()].copy()
    if len(tr) < 5:
        raise SystemExit(f"Too few labeled training rows after dropping NaNs: {len(tr)}")

    # Task inference
    task = args.task
    if task == "auto":
        ytmp = tr[args.label]
        uniq = pd.unique(ytmp.dropna())
        if ytmp.dtype == bool or (len(uniq) <= 2 and set(map(str, uniq)).issubset({"0", "1", "False", "True"})):
            task = "binary"
        else:
            task = "regression"

    y_tr = tr[args.label]
    if task == "binary":
        y_tr = _to_binary(y_tr)
        if y_tr.dropna().nunique() < 2:
            raise SystemExit("Binary label has <2 classes in train; cannot do feature selection.")
    else:
        y_tr = pd.to_numeric(y_tr, errors="coerce")

    # Coerce features numeric
    X_tr = tr[feat_cols].apply(pd.to_numeric, errors="coerce")

    # Missingness filter (train only)
    miss = X_tr.isna().mean(axis=0)
    keep_miss = miss[miss <= args.max_missing_frac].index.tolist()
    X_tr = X_tr[keep_miss]

    # Drop constant features (train only)
    nun = X_tr.nunique(dropna=True)
    keep_var = nun[nun > 1].index.tolist()
    X_tr = X_tr[keep_var]

    # Univariate scores
    scores = _univariate_scores(X_tr, y_tr.astype(float))
    scored = sorted([(c, scores.get(c, float("nan"))) for c in X_tr.columns], key=lambda x: (-(x[1] if np.isfinite(x[1]) else -1), x[0]))
    scored = [x for x in scored if np.isfinite(x[1])]
    if not scored:
        raise SystemExit("All univariate scores are NaN; cannot proceed.")
    pre = [c for (c, _s) in scored[: max(1, min(args.topk_univariate, len(scored)))]]
    X_pre = X_tr[pre]

    # Correlation pruning
    kept_corr = _corr_prune(X_pre, scores, corr_thresh=args.corr_thresh)

    # Final ranking
    selected: List[str]
    xgb = _import_xgboost() if args.use_xgb else None
    if args.use_xgb and xgb is None:
        raise SystemExit("Requested --use-xgb but xgboost is not installed.")

    if xgb is None:
        # fall back to univariate score order after corr pruning
        kept_scored = sorted([(c, scores.get(c, float("nan"))) for c in kept_corr], key=lambda x: -(x[1] if np.isfinite(x[1]) else -1))
        selected = [c for (c, _s) in kept_scored[: min(args.topk_final, len(kept_scored))]]
        model_info = {"used": False}
    else:
        # Train quick XGB with early stopping using native xgboost.train (no sklearn dependency)
        def _xy(part: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
            X = part[kept_corr].apply(pd.to_numeric, errors="coerce")
            y = part[args.label]
            y = _to_binary(y) if task == "binary" else pd.to_numeric(y, errors="coerce")
            m = y.notna()
            return X[m], y[m].to_numpy()

        Xtr, ytr = _xy(tr)
        Xva, yva = _xy(va) if len(va) else (pd.DataFrame(columns=kept_corr), np.array([]))
        if len(Xva) < 5:
            # no/too-small val: reuse train as eval (not ideal, but keeps flow moving)
            Xva, yva = Xtr, ytr

        if task == "binary":
            params: Dict[str, object] = dict(
                objective="binary:logistic",
                eval_metric="auc",
            )
        else:
            params = dict(
                objective="reg:squarederror",
                eval_metric="rmse",
            )

        params.update(
            dict(
                eta=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                seed=args.seed,
                nthread=min(30, (os.cpu_count() or 8)),
            )
        )

        dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=kept_corr)
        deval = xgb.DMatrix(Xva, label=yva, feature_names=kept_corr)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=4000,
            evals=[(deval, "eval")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        gain = booster.get_score(importance_type="gain")
        # map to all features
        gain_all = {f: float(gain.get(f, 0.0)) for f in kept_corr}
        ranked = sorted(gain_all.items(), key=lambda kv: kv[1], reverse=True)
        selected = [k for (k, _v) in ranked[: min(args.topk_final, len(ranked))]]
        model_info = {
            "used": True,
            "best_iteration": int(getattr(booster, "best_iteration", -1) or -1),
        }
        # save importances
        pd.DataFrame(ranked, columns=["feature", "gain"]).to_csv(out_dir / "xgb_gain_importance.csv", index=False)

    # Write selected features
    (out_dir / "selected_features.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    with open(out_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "task": task, "selected_features": selected}, f, indent=2)

    # Reduced table (keep useful metadata + label + selected features)
    meta_cols = [c for c in ["series_uid", "patient_id", "dataset", "split", "mask_path", "image_path"] if c in df.columns]
    keep_cols = meta_cols + ([args.label] if args.label in df.columns else []) + selected
    reduced = df[keep_cols].copy()
    reduced.to_csv(out_dir / "reduced_table.csv", index=False)

    report = {
        "data": args.data,
        "label": args.label,
        "task": task,
        "prefixes": list(prefixes),
        "n_rows_total": int(len(df)),
        "n_rows_train_labeled": int(len(tr)),
        "n_features_initial": int(len(feat_cols)),
        "n_features_after_missing": int(len(keep_miss)),
        "n_features_after_constant": int(len(keep_var)),
        "n_features_after_univariate_topk": int(len(pre)),
        "n_features_after_corr_prune": int(len(kept_corr)),
        "n_features_selected_final": int(len(selected)),
        "xgb": model_info,
        "params": {
            "max_missing_frac": args.max_missing_frac,
            "topk_univariate": args.topk_univariate,
            "corr_thresh": args.corr_thresh,
            "topk_final": args.topk_final,
        },
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"Wrote selected features -> {out_dir / 'selected_features.txt'} (n={len(selected)})", flush=True)
    print(f"Wrote reduced table -> {out_dir / 'reduced_table.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


