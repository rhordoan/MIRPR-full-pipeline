#!/usr/bin/env python3
"""
Train a mask-aware 3D classifier (DenseNet) on inferred CT+mask pairs.

Key ideas:
- Use the Vista3D mask as an ROI: crop around the mask bounding box + margin.
- Use the mask as an additional input channel (CT + mask).
- Split leakage-safe by patient_id (group split).

Expected inputs:
- Inference outputs directory containing:
    <series_uid>_ct_resampled.nii.gz
    <series_uid>_mask_clean.nii.gz
- Labels CSV with at least columns:
    series_uid, patient_id, egfr_mutated and/or kras_mutated

Example:
  source /home/shadeform/.venvs/vista-3d/bin/activate
  python scripts/train_densenet_roi.py \
    --inputs /home/shadeform/MIRPR-full-pipeline/outputs/nsclc_radiogenomics_inference \
    --labels-csv /home/shadeform/MIRPR-full-pipeline/outputs/nsclc_radiogenomics_labels.csv \
    --label egfr_mutated \
    --out-dir /home/shadeform/MIRPR-full-pipeline/outputs/dl/nsclc_egfr_densenet_roi
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import DenseNet121
from torchvision.models.video import r3d_18
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    Rand3DElasticd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
)


@dataclass(frozen=True)
class Sample:
    series_uid: str
    patient_id: str
    image: str
    mask: str
    y: int  # 0/1


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_boolish(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return 1
    if s in {"false", "0", "no"}:
        return 0
    if s in {"nan", "none", ""}:
        return None
    # tolerate python bool stringification
    if s == "t":
        return 1
    if s == "f":
        return 0
    return None


def load_labels(labels_csv: str, label_col: str) -> Dict[str, Tuple[str, int]]:
    """
    Returns mapping: series_uid -> (patient_id, y)
    """
    out: Dict[str, Tuple[str, int]] = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"Empty labels CSV: {labels_csv}")
        if "series_uid" not in r.fieldnames or "patient_id" not in r.fieldnames:
            raise SystemExit("labels CSV must contain columns: series_uid, patient_id")
        if label_col not in r.fieldnames:
            raise SystemExit(f"Label column not found in labels CSV: {label_col!r}")
        for row in r:
            series_uid = (row.get("series_uid") or "").strip()
            patient_id = (row.get("patient_id") or "").strip()
            y = _parse_boolish(row.get(label_col))
            if not series_uid or not patient_id or y is None:
                continue
            out[series_uid] = (patient_id, int(y))
    return out


def discover_samples(inputs_dir: str, labels_map: Dict[str, Tuple[str, int]]) -> List[Sample]:
    samples: List[Sample] = []
    # drive from CTs (ensures radiomics-ready layout)
    for fn in sorted(os.listdir(inputs_dir)):
        if not fn.endswith("_ct_resampled.nii.gz"):
            continue
        series_uid = fn.replace("_ct_resampled.nii.gz", "")
        ct = os.path.join(inputs_dir, fn)
        mask = os.path.join(inputs_dir, f"{series_uid}_mask_clean.nii.gz")
        if not os.path.exists(mask):
            continue
        lab = labels_map.get(series_uid)
        if not lab:
            continue
        patient_id, y = lab
        samples.append(Sample(series_uid=series_uid, patient_id=patient_id, image=ct, mask=mask, y=y))
    return samples


def stratified_patient_split(
    patient_to_y: Dict[str, int],
    *,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Stratified split at patient level (binary y).
    """
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 1:
        raise SystemExit("Invalid split fractions; require 0 < val,test and val+test < 1.")
    pos = [p for p, y in patient_to_y.items() if y == 1]
    neg = [p for p, y in patient_to_y.items() if y == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_group(xs: List[str]) -> Tuple[List[str], List[str], List[str]]:
        n = len(xs)
        n_test = max(1, int(round(n * test_frac))) if n >= 5 else max(0, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac))) if n >= 5 else max(0, int(round(n * val_frac)))
        # ensure we don't exceed
        if (n_test + n_val) > n:
            n_val = max(0, n - n_test)
        test = xs[:n_test]
        val = xs[n_test : n_test + n_val]
        train = xs[n_test + n_val :]
        return train, val, test

    pos_tr, pos_va, pos_te = split_group(pos)
    neg_tr, neg_va, neg_te = split_group(neg)

    train = pos_tr + neg_tr
    val = pos_va + neg_va
    test = pos_te + neg_te
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def make_transforms(
    *,
    roi_size: Tuple[int, int, int],
    crop_margin_mm: float,
    spacing_mm: float,
    train: bool,
) -> Compose:
    margin_vox = int(round(float(crop_margin_mm) / float(spacing_mm)))
    margin = (margin_vox, margin_vox, margin_vox)
    xforms: List[object] = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        # Image already in HU and clipped during inference; map to [0,1] for the classifier.
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000.0,
            a_max=1000.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "mask"], source_key="mask", margin=margin),
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=roi_size),
    ]
    if train:
        xforms += [
            RandBiasFieldd(keys=["image"], prob=0.25, coeff_range=(0.0, 0.05)),
            RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.05),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "mask"],
                prob=0.25,
                rotate_range=(0.15, 0.15, 0.15),
                translate_range=(8, 8, 8),
                scale_range=(0.10, 0.10, 0.10),
                padding_mode="border",
                mode=("bilinear", "nearest"),
            ),
            Rand3DElasticd(
                keys=["image", "mask"],
                prob=0.15,
                sigma_range=(4, 6),
                magnitude_range=(50, 100),
                padding_mode="border",
                mode=("bilinear", "nearest"),
            ),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        ]
    xforms += [
        ConcatItemsd(keys=["image", "mask"], name="x", dim=0),
        EnsureTyped(keys=["x"]),
    ]
    return Compose(xforms)


def _roc_auc_from_probs(y_true: List[int], y_prob: List[float]) -> Optional[float]:
    """
    Fast AUC computation without sklearn.
    Returns None if AUC is undefined (single class).
    """
    if not y_true or len(y_true) != len(y_prob):
        return None
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    # Rank-based AUC with average ranks for ties.
    order = sorted(range(len(y_prob)), key=lambda i: (y_prob[i], i))
    ranks = [0.0] * len(y_prob)
    i = 0
    r = 1.0
    while i < len(order):
        j = i
        p = y_prob[order[i]]
        while j < len(order) and y_prob[order[j]] == p:
            j += 1
        # average rank for tie block [i, j)
        avg = (r + (r + (j - i) - 1.0)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        r += (j - i)
        i = j
    sum_pos_ranks = sum(ranks[i] for i, y in enumerate(y_true) if y == 1)
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def eval_epoch(model, loader, device) -> Tuple[float, Optional[float], List[Tuple[str, str, float, int]]]:
    model.eval()
    losses: List[float] = []
    y_probs: List[float] = []
    y_true: List[int] = []
    rows: List[Tuple[str, str, float, int]] = []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device).float().view(-1, 1)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits).detach().cpu().view(-1).numpy()
        ys = y.detach().cpu().view(-1).numpy().astype(int)
        y_probs.extend([float(p) for p in probs])
        y_true.extend([int(v) for v in ys])
        for s_uid, p_id, p, yt in zip(batch["series_uid"], batch["patient_id"], probs, ys):
            rows.append((str(s_uid), str(p_id), float(p), int(yt)))

    auc = _roc_auc_from_probs(y_true, y_probs)
    return float(np.mean(losses) if losses else 0.0), auc, rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Dir containing *_ct_resampled.nii.gz and *_mask_clean.nii.gz")
    ap.add_argument("--labels-csv", required=True, help="Series-level labels CSV (must include series_uid, patient_id)")
    ap.add_argument("--label", required=True, help="Label column (e.g., egfr_mutated or kras_mutated)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--roi-size", default="160,160,160", help="ROI size z,y,x after crop (default: 160,160,160)")
    ap.add_argument("--crop-margin-mm", type=float, default=25.0, help="Crop margin around mask bbox (mm). Default: 25")
    ap.add_argument("--spacing-mm", type=float, default=0.7, help="Voxel spacing in mm for inputs (default: 0.7)")
    ap.add_argument("--cache-rate", type=float, default=1.0, help="CacheDataset cache rate (0..1). Lower to reduce preload time.")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="Use torch AMP for training")
    ap.add_argument(
        "--backbone",
        choices=["densenet121", "r3d18"],
        default="densenet121",
        help="Backbone architecture (default: densenet121).",
    )
    ap.add_argument("--dropout", type=float, default=0.2, help="Dropout applied to logits (default: 0.2)")
    ap.add_argument("--focal-gamma", type=float, default=0.0, help="If >0, use focal loss with this gamma (alpha balanced via pos_weight).")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing in [0,1]; applied to targets before loss.")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="If >0, clip grad norm to this value.")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="none", help="LR scheduler: cosine annealing or none.")
    ap.add_argument("--warmup-epochs", type=int, default=0, help="Linear warmup epochs (applied before scheduler).")
    ap.add_argument("--patience", type=int, default=12, help="Early-stop patience on val AUC (epochs without improvement).")
    args = ap.parse_args(argv)

    roi_size = tuple(int(x.strip()) for x in str(args.roi_size).split(","))
    if len(roi_size) != 3:
        raise SystemExit("--roi-size must be 'z,y,x'")

    os.makedirs(args.out_dir, exist_ok=True)
    _seed_all(int(args.seed))

    labels_map = load_labels(args.labels_csv, args.label)
    samples = discover_samples(args.inputs, labels_map)
    if not samples:
        raise SystemExit("No training samples found. Check --inputs and --labels-csv alignment.")

    # Patient-level labels (if multiple series per patient exist, use max label)
    patient_to_y: Dict[str, int] = {}
    for s in samples:
        patient_to_y[s.patient_id] = max(patient_to_y.get(s.patient_id, 0), int(s.y))

    train_p, val_p, test_p = stratified_patient_split(
        patient_to_y,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
    )
    train_set = {p for p in train_p}
    val_set = {p for p in val_p}
    test_set = {p for p in test_p}

    def to_dict(s: Sample) -> dict:
        return {
            "image": s.image,
            "mask": s.mask,
            "y": int(s.y),
            "series_uid": s.series_uid,
            "patient_id": s.patient_id,
        }

    train_items = [to_dict(s) for s in samples if s.patient_id in train_set]
    val_items = [to_dict(s) for s in samples if s.patient_id in val_set]
    test_items = [to_dict(s) for s in samples if s.patient_id in test_set]

    # Basic report
    def _count(items: List[dict]) -> Tuple[int, int, int]:
        ys = [int(d["y"]) for d in items]
        return len(items), sum(ys), len(set(d["patient_id"] for d in items))

    n_tr, pos_tr, p_tr = _count(train_items)
    n_va, pos_va, p_va = _count(val_items)
    n_te, pos_te, p_te = _count(test_items)

    with open(os.path.join(args.out_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "n_samples_total": len(samples),
                "n_patients_total": len(patient_to_y),
                "train": {"samples": n_tr, "patients": p_tr, "positives": pos_tr},
                "val": {"samples": n_va, "patients": p_va, "positives": pos_va},
                "test": {"samples": n_te, "patients": p_te, "positives": pos_te},
            },
            f,
            indent=2,
            sort_keys=True,
        )

    x_tr = make_transforms(roi_size=roi_size, crop_margin_mm=args.crop_margin_mm, spacing_mm=args.spacing_mm, train=True)
    x_ev = make_transforms(roi_size=roi_size, crop_margin_mm=args.crop_margin_mm, spacing_mm=args.spacing_mm, train=False)

    cache_rate = max(0.0, min(1.0, float(args.cache_rate)))
    train_ds = CacheDataset(train_items, transform=x_tr, cache_rate=cache_rate, num_workers=int(args.num_workers))
    val_ds = CacheDataset(val_items, transform=x_ev, cache_rate=cache_rate, num_workers=int(args.num_workers))
    test_ds = CacheDataset(test_items, transform=x_ev, cache_rate=cache_rate, num_workers=int(args.num_workers))

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(args.num_workers))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=int(args.num_workers))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_model(backbone: str, dropout: float) -> torch.nn.Module:
        if backbone == "densenet121":
            base = DenseNet121(spatial_dims=3, in_channels=2, out_channels=1)
            return torch.nn.Sequential(base, torch.nn.Dropout(p=dropout))
        if backbone == "r3d18":
            # TorchVision 3D ResNet-18 (video) adapted for 2 input channels and binary logits.
            m = r3d_18(weights=None, progress=False)
            # Replace stem conv to accept 2 channels.
            m.stem[0] = torch.nn.Conv3d(
                in_channels=2,
                out_channels=64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            )
            in_feats = m.fc.in_features
            m.fc = torch.nn.Sequential(torch.nn.Dropout(p=dropout), torch.nn.Linear(in_feats, 1))
            return m
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = build_model(str(args.backbone), float(args.dropout)).to(device)

    # Pos weight to help imbalance
    pos = sum(1 for d in train_items if int(d["y"]) == 1)
    neg = sum(1 for d in train_items if int(d["y"]) == 0)
    pos_weight = torch.tensor([float(neg / max(1, pos))], device=device)
    focal_gamma = float(args.focal_gamma)
    label_smoothing = max(0.0, min(1.0, float(args.label_smoothing)))

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
        if focal_gamma > 0:
            p = torch.sigmoid(logits)
            pt = targets * p + (1 - targets) * (1 - p)
            bce = ((1 - pt) ** focal_gamma) * bce
        return bce.mean()

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    base_lr = float(args.lr)
    scheduler = None
    if args.scheduler == "cosine":
        t_max = max(1, int(args.epochs) - int(args.warmup_epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    best_auc = -1.0
    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, "model_best.pt")
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    best_epoch = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        # Linear warmup on LR if requested
        if int(args.warmup_epochs) > 0 and epoch <= int(args.warmup_epochs):
            scale = float(epoch) / float(max(1, int(args.warmup_epochs)))
            for g in opt.param_groups:
                g["lr"] = base_lr * scale
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device).float().view(-1, 1)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        tr_loss = float(np.mean(losses) if losses else 0.0)
        va_loss, va_auc, _ = eval_epoch(model, val_loader, device)
        if scheduler and epoch > int(args.warmup_epochs):
            scheduler.step()

        saved = False
        if va_auc is not None:
            if float(va_auc) > best_auc:
                best_auc = float(va_auc)
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": best_auc}, best_path)
                saved = True
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Fallback when val set has a single class: track best by val loss.
            if va_loss < best_val_loss:
                best_val_loss = float(va_loss)
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": best_val_loss}, best_path)
                saved = True
                no_improve = 0
            else:
                no_improve += 1

        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_auc": va_auc,
            "best_val_auc": best_auc if best_auc >= 0 else None,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "saved": saved,
            "elapsed_s": int(time.time() - t0),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if int(args.patience) > 0 and no_improve >= int(args.patience):
            break

    # Load best and evaluate on test
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    te_loss, te_auc, rows = eval_epoch(model, test_loader, device)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "test_loss": te_loss,
                "test_auc": te_auc,
                "best_val_auc": best_auc if best_auc >= 0 else None,
                "best_epoch": best_epoch,
                "roi_size": roi_size,
                "crop_margin_mm": float(args.crop_margin_mm),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    with open(os.path.join(args.out_dir, "test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series_uid", "patient_id", "y_true", "y_prob"])
        for series_uid, patient_id, y_prob, y_true in rows:
            w.writerow([series_uid, patient_id, y_true, y_prob])

    print(f"Wrote: {args.out_dir}/metrics.json and test_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


