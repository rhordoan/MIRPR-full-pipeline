import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run balanced multi-split TTF1 XGBoost evaluation with fixed hyperparams."
    )
    ap.add_argument(
        "--table",
        type=Path,
        default=Path("/home/shadeform/outputs/ml/feature_pruned/private_ttf1/reduced_table.csv"),
        help="Path to pruned radiomics table with columns patient_id, ttf1_positive, and features.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/shadeform/outputs/ml/models/private_ttf1_multi_split.json"),
        help="Where to save the results JSON list.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="101,103,107,109,113",
        help="Comma-separated RNG seeds to sample multiple balanced splits.",
    )
    ap.add_argument("--test-size", type=int, default=18, help="Number of patient groups in test.")
    ap.add_argument("--val-size", type=int, default=14, help="Number of patient groups in val.")
    ap.add_argument("--min-test-pos", type=int, default=10, help="Minimum positives in test split.")
    ap.add_argument("--min-test-neg", type=int, default=6, help="Minimum negatives in test split.")
    ap.add_argument(
        "--max-tries",
        type=int,
        default=3000,
        help="Tries per seed to find a balanced split before giving up.",
    )
    # XGB hyperparameters
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-child-weight", type=float, default=4.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.9)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--reg-lambda", type=float, default=4.0)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--max-delta-step", type=float, default=1.0)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--num-boost-round", type=int, default=1200)
    ap.add_argument("--early-stopping-rounds", type=int, default=80)
    return ap.parse_args()


def discover_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(("original", "wavelet", "log"))]


def make_split(
    df: pd.DataFrame,
    groups: Sequence[str],
    seed: int,
    test_size: int,
    val_size: int,
    min_test_pos: int,
    min_test_neg: int,
    max_tries: int,
) -> Optional[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    groups = list(groups)
    for _ in range(max_tries):
        rng.shuffle(groups)
        test_g = set(groups[:test_size])
        val_g = set(groups[test_size : test_size + val_size])
        assign = {
            g: ("test" if g in test_g else "val" if g in val_g else "train") for g in groups
        }
        d = df.copy()
        d["split"] = d["patient_id"].astype(str).map(assign)
        te = d[d["split"] == "test"]
        pos = int(te["ttf1_positive"].sum())
        neg = len(te) - pos
        if pos >= min_test_pos and neg >= min_test_neg:
            return d
    return None


def train_eval(
    df: pd.DataFrame,
    feats: List[str],
    params: Dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[float, float, int]:
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    dtr = xgb.DMatrix(tr[feats], label=tr["ttf1_positive"].astype(int), feature_names=feats)
    dva = xgb.DMatrix(va[feats], label=va["ttf1_positive"].astype(int), feature_names=feats)
    dte = xgb.DMatrix(te[feats], label=te["ttf1_positive"].astype(int), feature_names=feats)
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dva, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    it_range = (0, booster.best_iteration + 1) if booster.best_iteration is not None else (0, -1)
    p_te = booster.predict(dte, iteration_range=it_range)
    p_va = booster.predict(dva, iteration_range=it_range)
    return (
        float(roc_auc_score(dte.get_label(), p_te)),
        float(roc_auc_score(dva.get_label(), p_va)),
        int(booster.best_iteration) if booster.best_iteration is not None else -1,
    )


def main() -> None:
    args = parse_args()
    df_base = pd.read_csv(args.table)
    feats = discover_features(df_base)
    groups = df_base["patient_id"].astype(str).unique().tolist()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eta": args.eta,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "max_delta_step": args.max_delta_step,
        "seed": 1337,
        "nthread": args.nthread,
    }

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_rows: List[Dict] = []
    for s in seeds:
        split_df = make_split(
            df_base,
            groups,
            seed=s,
            test_size=args.test_size,
            val_size=args.val_size,
            min_test_pos=args.min_test_pos,
            min_test_neg=args.min_test_neg,
            max_tries=args.max_tries,
        )
        if split_df is None:
            continue
        te = split_df[split_df["split"] == "test"]
        pos = int(te["ttf1_positive"].sum())
        neg = len(te) - pos
        auc_te, auc_va, best_it = train_eval(
            split_df, feats, params, args.num_boost_round, args.early_stopping_rounds
        )
        out_rows.append(
            {
                "seed": s,
                "test_auc": auc_te,
                "val_auc": auc_va,
                "test_pos": pos,
                "test_neg": neg,
                "best_iter": best_it,
                "params": params,
                "num_boost_round": args.num_boost_round,
                "early_stopping_rounds": args.early_stopping_rounds,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_rows, f, indent=2)

    if out_rows:
        tas = [r["test_auc"] for r in out_rows]
        print(
            f"Splits completed: {len(out_rows)} | "
            f"test_auc mean {float(np.mean(tas)):.4f} "
            f"std {float(np.std(tas)):.4f} "
            f"max {float(np.max(tas)):.4f}"
        )
        print(f"Saved to {args.out}")
    else:
        print("No balanced splits found under current constraints.")


if __name__ == "__main__":
    main()

