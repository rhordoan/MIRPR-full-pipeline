#!/usr/bin/env python3
"""
Train an XGBoost model from a radiomics ML table (optionally feature-selected).

Key properties:
  - **No data leakage** by default if your CSV has patient-level `split` (train/val/test).
  - Uses **native** XGBoost training (`xgboost.train` + `DMatrix`) to avoid sklearn dependency.
  - Optional **group CV** by patient_id on train+val (keeps test untouched).

Typical usage (binary classification, EGFR):
  python scripts/train_xgboost.py \
    --data /home/shadeform/models/vista-3d/outputs/ml/nsclc_radiomics_ml_table_labeled.csv \
    --label egfr_mutated \
    --out-dir /home/shadeform/models/vista-3d/outputs/ml/models/nsclc_egfr
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
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "xgboost is not installed in this environment. Install it with:\n"
            "  source /home/shadeform/.venvs/vista-3d/bin/activate\n"
            "  pip install -r /home/shadeform/models/vista-3d/requirements-ml.txt\n"
        ) from e


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUC via rank statistic (no sklearn dependency).
    Returns NaN if undefined (all same label).
    """
    y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    order = np.argsort(y_score)
    y = y_true[order]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    ranks = np.arange(1, len(y) + 1, dtype=float)
    r_pos = float(ranks[y == 1].sum())
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _has_both_classes(y: np.ndarray) -> bool:
    if y.size == 0:
        return False
    y = y.astype(int)
    return int(y.min()) != int(y.max())


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return float((2 * prec * rec) / (prec + rec)) if (prec + rec) else 0.0


def _split_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        tr = df[df["split"] == "train"].copy()
        va = df[df["split"] == "val"].copy()
        te = df[df["split"] == "test"].copy()
        return tr, va, te
    raise SystemExit("Missing 'split' column. Please regenerate your ML table with prepare_xgb_dataset.py for leakage-safe splits.")


def _choose_feature_cols(df: pd.DataFrame, label: str, prefixes: Tuple[str, ...], explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [c for c in explicit if c in df.columns]
    feat = [c for c in df.columns if isinstance(c, str) and c.startswith(prefixes)]
    if feat:
        return feat
    exclude = {label, "series_uid", "series", "patient_id", "dataset", "split", "mask_path", "image_path", "series_dir"}
    feat = [c for c in df.columns if c not in exclude]
    return feat


def _group_kfold(groups: List[str], k: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    out: List[Tuple[List[str], List[str]]] = []
    for i in range(k):
        val = folds[i].tolist()
        tr = np.concatenate([folds[j] for j in range(k) if j != i]).tolist()
        out.append((tr, val))
    return out


def _best_iteration(booster) -> int:
    bi = getattr(booster, "best_iteration", None)
    if bi is None:
        return -1
    try:
        return int(bi)
    except Exception:  # noqa: BLE001
        return -1


def _safe_iteration_range(booster, fallback_num_boost_round: int) -> Tuple[int, int]:
    best_iter = _best_iteration(booster)
    end = best_iter + 1 if best_iter >= 0 else -1
    try:
        if hasattr(booster, "num_boosted_rounds"):
            n_rounds = int(booster.num_boosted_rounds())
        else:
            n_rounds = len(booster.get_dump())
    except Exception:  # noqa: BLE001
        try:
            n_rounds = len(booster.get_dump())
        except Exception:  # noqa: BLE001
            n_rounds = int(fallback_num_boost_round)
    if end <= 0:
        end = n_rounds
    end = min(int(end), int(n_rounds))
    if end <= 0:
        end = 1
    return (0, int(end))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV produced by prepare_xgb_dataset.py")
    ap.add_argument("--label", required=True, help="Label/target column name (e.g., egfr_mutated)")
    ap.add_argument("--out-dir", required=True, help="Output directory for model + metrics")
    ap.add_argument("--task", default="auto", choices=["auto", "binary", "regression"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-boost-round", type=int, default=4000)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--feature-prefixes", default="original,wavelet,log")
    ap.add_argument("--features", default=None, help="Optional comma-separated feature list to use.")
    ap.add_argument("--group-col", default="patient_id", help="Group column for CV/sanity checks (default patient_id).")
    ap.add_argument("--cv-folds", type=int, default=0, help="If >0, run group K-fold CV on train+val (test untouched).")
    args = ap.parse_args(argv)

    xgb = _import_xgboost()

    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        raise SystemExit(f"Label column not found: {args.label!r}")

    prefixes = tuple(p.strip() for p in args.feature_prefixes.split(",") if p.strip())
    explicit_feats = [c.strip() for c in (args.features or "").split(",") if c.strip()] or None
    feature_cols = _choose_feature_cols(df, args.label, prefixes, explicit_feats)
    if not feature_cols:
        raise SystemExit(f"No feature columns found with prefixes={prefixes}")

    d0 = len(df)
    df = df[df[args.label].notna()].copy()
    if len(df) == 0:
        raise SystemExit("After dropping missing labels, 0 rows remain.")
    if len(df) != d0:
        print(f"Dropped {d0-len(df)} rows with missing label {args.label!r}")

    task = args.task
    y_raw = df[args.label]
    if task == "auto":
        if y_raw.dtype == bool:
            task = "binary"
        else:
            uniq = pd.unique(y_raw.dropna())
            if len(uniq) <= 2 and set(map(str, uniq)).issubset({"0", "1", "False", "True"}):
                task = "binary"
            else:
                task = "regression"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = _split_df(df)
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        if len(part) == 0:
            raise SystemExit(f"Split {name!r} has 0 rows. Re-run prepare_xgb_dataset.py or adjust fractions.")

    if args.group_col in df.columns:
        gtr = set(tr[args.group_col].astype(str))
        gva = set(va[args.group_col].astype(str))
        gte = set(te[args.group_col].astype(str))
        if (gtr & gva) or (gtr & gte) or (gva & gte):
            raise SystemExit("Leakage detected: group_col appears in multiple splits. Regenerate split with prepare_xgb_dataset.py.")

    def _xy(d: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        X = d[feature_cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        y = d[args.label]
        if task == "binary":
            y = pd.to_numeric(y.astype(int), errors="coerce")
        else:
            y = pd.to_numeric(y, errors="coerce")
        m = y.notna()
        return X[m], y[m].to_numpy(dtype=float, copy=False)

    X_tr, y_tr = _xy(tr)
    X_va, y_va = _xy(va)
    X_te, y_te = _xy(te)

    if task == "binary":
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        if pos == 0 or neg == 0:
            raise SystemExit("Train split has only one class for binary task; cannot train.")
        scale_pos_weight = neg / pos
    else:
        scale_pos_weight = 1.0

    params: Dict[str, object] = dict(
        eta=args.eta,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        seed=args.seed,
        nthread=min(30, (os.cpu_count() or 8)),
    )
    if task == "binary":
        eval_metric = "auc" if _has_both_classes(y_va) else "logloss"
        params.update(dict(objective="binary:logistic", eval_metric=eval_metric, scale_pos_weight=scale_pos_weight))
    else:
        params.update(dict(objective="reg:squarederror", eval_metric="rmse"))

    cv_summary: Dict[str, object] = {"ran": False}
    if args.cv_folds and args.cv_folds > 1:
        if args.group_col not in df.columns:
            raise SystemExit(f"--cv-folds requires group column {args.group_col!r} in data.")
        base = df[df["split"].isin(["train", "val"])].copy()
        base = base[base[args.label].notna()].copy()
        groups = base[args.group_col].astype(str).tolist()
        splits = _group_kfold(groups=groups, k=int(args.cv_folds), seed=args.seed)
        aucs: List[float] = []
        best_iters: List[int] = []
        for fold_i, (g_tr, g_va) in enumerate(splits):
            tr_f = base[base[args.group_col].astype(str).isin(g_tr)].copy()
            va_f = base[base[args.group_col].astype(str).isin(g_va)].copy()
            Xtr_f, ytr_f = _xy(tr_f)
            Xva_f, yva_f = _xy(va_f)
            dtr = xgb.DMatrix(Xtr_f, label=ytr_f, feature_names=feature_cols)
            dva = xgb.DMatrix(Xva_f, label=yva_f, feature_names=feature_cols)
            params_fold = dict(params)
            if task == "binary":
                params_fold["eval_metric"] = "auc" if _has_both_classes(yva_f) else "logloss"
            booster = xgb.train(
                params=params_fold,
                dtrain=dtr,
                num_boost_round=int(args.num_boost_round),
                evals=[(dva, "eval")],
                early_stopping_rounds=int(args.early_stopping_rounds),
                verbose_eval=False,
            )
            it_range_cv = _safe_iteration_range(booster, fallback_num_boost_round=int(args.num_boost_round))
            p = booster.predict(dva, iteration_range=it_range_cv)
            if task == "binary":
                aucs.append(_binary_auc(yva_f.astype(int), p) if _has_both_classes(yva_f) else float("nan"))
            best_iters.append(_best_iteration(booster))
            msg_auc = aucs[-1] if aucs else float("nan")
            print(f"[cv] fold {fold_i+1}/{len(splits)}: best_iter={best_iters[-1]} auc={msg_auc}", flush=True)
        cv_summary = {
            "ran": True,
            "folds": int(args.cv_folds),
            "aucs": aucs,
            "auc_mean": float(np.nanmean(aucs)) if aucs else None,
            "auc_std": float(np.nanstd(aucs)) if aucs else None,
            "best_iterations": best_iters,
            "best_iteration_median": int(np.median([b for b in best_iters if b >= 0])) if any(b >= 0 for b in best_iters) else -1,
        }

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=feature_cols)
    dte = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols)
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=int(args.num_boost_round),
        evals=[(dva, "val")],
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose_eval=False,
    )
    best_iter = _best_iteration(booster)
    it_range = _safe_iteration_range(booster, fallback_num_boost_round=int(args.num_boost_round))

    if task == "binary":
        p_va = booster.predict(dva, iteration_range=it_range)
        p_te = booster.predict(dte, iteration_range=it_range)
        pred_va = (p_va >= 0.5).astype(int)
        pred_te = (p_te >= 0.5).astype(int)
        metrics = {
            "task": "binary",
            "label": args.label,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "val_auc": _binary_auc(y_va.astype(int), p_va) if _has_both_classes(y_va) else float("nan"),
            "test_auc": _binary_auc(y_te.astype(int), p_te) if _has_both_classes(y_te) else float("nan"),
            "val_acc": _accuracy(y_va.astype(int), pred_va),
            "test_acc": _accuracy(y_te.astype(int), pred_te),
            "val_f1": _f1(y_va.astype(int), pred_va),
            "test_f1": _f1(y_te.astype(int), pred_te),
            "best_iteration": best_iter,
            "scale_pos_weight": float(scale_pos_weight),
            "cv": cv_summary,
            "class_counts": {
                "train_pos": int((y_tr.astype(int) == 1).sum()),
                "train_neg": int((y_tr.astype(int) == 0).sum()),
                "val_pos": int((y_va.astype(int) == 1).sum()),
                "val_neg": int((y_va.astype(int) == 0).sum()),
                "test_pos": int((y_te.astype(int) == 1).sum()),
                "test_neg": int((y_te.astype(int) == 0).sum()),
            },
            "eval_metric_used": str(params.get("eval_metric")),
        }
        pd.DataFrame({"p": p_te, "y": y_te.astype(int)}).to_csv(out_dir / "test_predictions.csv", index=False)
    else:
        pred_va = booster.predict(dva, iteration_range=it_range)
        pred_te = booster.predict(dte, iteration_range=it_range)
        mae_va = float(np.mean(np.abs(pred_va - y_va)))
        mae_te = float(np.mean(np.abs(pred_te - y_te)))
        rmse_va = float(np.sqrt(np.mean((pred_va - y_va) ** 2)))
        rmse_te = float(np.sqrt(np.mean((pred_te - y_te) ** 2)))
        metrics = {
            "task": "regression",
            "label": args.label,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "val_mae": mae_va,
            "test_mae": mae_te,
            "val_rmse": rmse_va,
            "test_rmse": rmse_te,
            "best_iteration": best_iter,
            "cv": cv_summary,
        }

    model_path = out_dir / "model.json"
    booster.save_model(str(model_path))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    try:
        gain = booster.get_score(importance_type="gain")
        fi = pd.DataFrame(
            [{"feature": f, "gain": float(gain.get(f, 0.0))} for f in feature_cols]
        ).sort_values("gain", ascending=False)
        fi.to_csv(out_dir / "feature_importance_gain.csv", index=False)
    except Exception:
        pass

    print(f"Wrote model -> {model_path}", flush=True)
    print(f"Wrote metrics -> {out_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
