#!/usr/bin/env python3
"""
Run inference with the Vista3D model on a single DICOM series directory.

Example:
  python scripts/run_inference_on_series.py \
    --series-dir /home/shadeform/models/vista-3d/data/tcia/TCGA-LUAD/TCGA-17-Z032/1.3.6.1.4.1.14519.5.2.1.7777.9002.103341830096723135986659789551/1.3.6.1.4.1.14519.5.2.1.7777.9002.320705545700604303450136239268 \
    --weights /home/shadeform/models/vista-3d/weights/best_model.pth \
    --out-dir /home/shadeform/models/vista-3d/outputs
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import vista3d132


TARGET_CLASS = 23  # Lung Tumor
TARGET_SPACING = (0.7, 0.7, 0.7)  # (z, y, x) mm
INTENSITY_RANGE = (-1000, 1000)
ROI_SIZE = (192, 192, 192)


def estimate_lung_z_bounds(
    vol_zyx: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    *,
    patient_hu_thresh: float = -900.0,
    lung_hu_thresh: float = -500.0,
    min_patient_frac: float = 0.02,
    min_lung_frac: float = 0.12,
    min_run_slices: int = 10,
    margin_mm: float = 30.0,
) -> Optional[Tuple[int, int]]:
    """
    Heuristic: find a contiguous z-range likely to contain lungs, to avoid running
    sliding-window inference over long regions with little/no lung.

    This is intentionally conservative:
    - "patient" pixels are those > patient_hu_thresh (to exclude outside-air)
    - "lung-like" pixels are those < lung_hu_thresh within patient pixels
    - select slices with lung_fraction >= min_lung_frac
    - expand by margin_mm on both ends

    Returns (z0, z1) inclusive bounds in the original volume index space, or None.
    """
    if vol_zyx.ndim != 3:
        return None

    z, y, x = vol_zyx.shape
    if z < 4:
        return None

    # patient_mask: excludes outside-air padding/background
    patient = vol_zyx > patient_hu_thresh
    patient_counts = patient.reshape(z, -1).sum(axis=1).astype(np.float32)
    patient_frac = patient_counts / float(y * x)

    # Only consider slices with some patient present.
    ok_patient = patient_frac >= float(min_patient_frac)
    if not bool(ok_patient.any()):
        return None

    # lung_like: low-HU regions within patient (lungs tend to dominate in chest slices)
    lung_like = (vol_zyx < lung_hu_thresh) & patient
    lung_counts = lung_like.reshape(z, -1).sum(axis=1).astype(np.float32)
    lung_frac = np.zeros((z,), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        lung_frac[ok_patient] = lung_counts[ok_patient] / np.maximum(patient_counts[ok_patient], 1.0)

    candidates = np.where(ok_patient & (lung_frac >= float(min_lung_frac)))[0]
    if candidates.size == 0:
        return None

    # Keep the largest contiguous run of candidate slices (helps ignore bowel gas).
    runs: List[Tuple[int, int]] = []
    start = int(candidates[0])
    prev = int(candidates[0])
    for idx in candidates[1:]:
        idx_i = int(idx)
        if idx_i == prev + 1:
            prev = idx_i
            continue
        runs.append((start, prev))
        start = idx_i
        prev = idx_i
    runs.append((start, prev))
    runs.sort(key=lambda ab: (ab[1] - ab[0] + 1), reverse=True)
    z0, z1 = runs[0]
    if (z1 - z0 + 1) < int(min_run_slices):
        return None

    # Expand by margin in mm (converted to slices using input z-spacing).
    sz = float(spacing_zyx[0]) if spacing_zyx and spacing_zyx[0] > 0 else 1.0
    margin_slices = int(round(float(margin_mm) / sz))
    z0 = max(0, int(z0) - margin_slices)
    z1 = min(z - 1, int(z1) + margin_slices)
    if z1 <= z0:
        return None
    return int(z0), int(z1)


def load_dicom_series(series_dir: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"No DICOMs found in {series_dir}")

    import pydicom

    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        if getattr(ds, "InstanceNumber", None) is None:
            order_key = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0.0
        else:
            order_key = int(ds.InstanceNumber)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = ds.pixel_array.astype(np.float32) * slope + intercept
        slices.append((order_key, arr))

    slices.sort(key=lambda x: x[0])
    vol = np.stack([s[1] for s in slices], axis=0)  # (z, y, x)

    ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
    pix = ds0.PixelSpacing
    spacing_xy = (float(pix[0]), float(pix[1]))
    spacing_z = float(getattr(ds0, "SliceThickness", 1.0))
    spacing = (spacing_z, spacing_xy[0], spacing_xy[1])  # z, y, x
    return vol, spacing


def resample_volume(vol: torch.Tensor, spacing: Tuple[float, float, float], target_spacing: Tuple[float, float, float]) -> torch.Tensor:
    # vol: (1, 1, z, y, x)
    z, y, x = vol.shape[2:]
    sz = int(round(z * spacing[0] / target_spacing[0]))
    sy = int(round(y * spacing[1] / target_spacing[1]))
    sx = int(round(x * spacing[2] / target_spacing[2]))
    return F.interpolate(vol, size=(sz, sy, sx), mode="trilinear", align_corners=False)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True, help="Path to a single DICOM series directory")
    ap.add_argument("--weights", default="/home/shadeform/models/vista-3d/weights/best_model.pth")
    ap.add_argument("--out-dir", default="/home/shadeform/models/vista-3d/outputs")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--disable-lung-crop",
        action="store_true",
        help=(
            "Disable heuristic z-cropping to likely lung slices. "
            "By default we try to crop to speed up inference on long scans with lots of non-lung slices."
        ),
    )
    ap.add_argument("--lung-crop-margin-mm", type=float, default=30.0, help="Margin (mm) added above/below detected lung z-range.")
    ap.add_argument("--lung-crop-min-lung-frac", type=float, default=0.12, help="Min lung-like fraction (within patient) to mark a slice as lung.")
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA autocast (mixed precision) during sliding window inference (faster on most GPUs).",
    )
    ap.add_argument("--png", action="store_true", help="Also save 2D overlay PNGs (up to 4 informative slices)")
    ap.add_argument("--clean", action="store_true", help="Run mask postprocessing (size filter, morph, optional dilation)")
    ap.add_argument("--min-vol-mm3", type=float, default=50.0, help="Minimum component volume to keep (mm^3). Default: 50")
    ap.add_argument("--closing-iters", type=int, default=1, help="Binary closing iterations (after size filter)")
    ap.add_argument("--opening-iters", type=int, default=1, help="Binary opening iterations (after fill holes)")
    ap.add_argument("--dilate-iters", type=int, default=0, help="Optional dilation iterations for peritumoral margin")
    ap.add_argument("--max-elongation", type=float, default=None, help="Optional max elongation ratio to keep components (e.g., 8.0). Omit to disable.")
    ap.add_argument("--metrics-csv", default=None, help="Optional path to append per-series metrics (raw/clean voxels and mm3)")
    ap.add_argument("--save-ct", dest="save_ct", action="store_true", help="Save resampled CT volume as NIfTI for radiomics")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    vol_np, spacing = load_dicom_series(args.series_dir)
    vol_np = np.clip(vol_np, INTENSITY_RANGE[0], INTENSITY_RANGE[1])

    # Optional: crop along z to likely lung-containing region (saves resample + SW inference work).
    z_crop0 = 0
    z_crop1 = vol_np.shape[0] - 1
    if not args.disable_lung_crop:
        bounds = estimate_lung_z_bounds(
            vol_zyx=vol_np,
            spacing_zyx=spacing,
            min_lung_frac=float(args.lung_crop_min_lung_frac),
            margin_mm=float(args.lung_crop_margin_mm),
        )
        if bounds is not None:
            z_crop0, z_crop1 = bounds
            if z_crop0 != 0 or z_crop1 != (vol_np.shape[0] - 1):
                vol_np = vol_np[z_crop0 : z_crop1 + 1]
                print(f"Auto-cropped z-range to likely lungs: z=[{z_crop0}, {z_crop1}] (slices={vol_np.shape[0]})", flush=True)
        else:
            print("Auto-crop: could not confidently detect lung z-range; using full scan.", flush=True)

    vol_t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)  # (1,1,z,y,x)
    vol_t = resample_volume(vol_t, spacing, TARGET_SPACING)

    device = torch.device(args.device)
    vol_t = vol_t.to(device, dtype=torch.float32)

    model = vista3d132(encoder_embed_dim=48, in_channels=1).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    use_amp = bool(args.amp) and device.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = sliding_window_inference(
                    inputs=vol_t,
                    roi_size=ROI_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                    transpose=True,
                    class_vector=torch.tensor([TARGET_CLASS], device=device),
                )
        else:
            logits = sliding_window_inference(
                inputs=vol_t,
                roi_size=ROI_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                transpose=True,
                class_vector=torch.tensor([TARGET_CLASS], device=device),
            )
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    pred_np = preds.cpu().numpy()[0, 0]
    voxel_vol_mm3 = TARGET_SPACING[0] * TARGET_SPACING[1] * TARGET_SPACING[2]

    # Save NIfTI for easy inspection
    series_name = os.path.basename(args.series_dir.rstrip("/"))
    out_path = os.path.join(args.out_dir, f"{series_name}_mask_raw.nii.gz")
    affine = np.diag(list(TARGET_SPACING)[::-1] + [1.0])  # simple affine
    nib.save(nib.Nifti1Image(pred_np.astype(np.float32), affine), out_path)
    print(f"Saved mask to: {out_path}")

    if args.save_ct:
        ct_out = os.path.join(args.out_dir, f"{series_name}_ct_resampled.nii.gz")
        ct_np = vol_t.cpu().numpy()[0, 0]
        nib.save(nib.Nifti1Image(ct_np.astype(np.float32), affine), ct_out)
        print(f"Saved resampled CT to: {ct_out}")

    raw_voxels = float((pred_np > 0.5).sum())
    raw_mm3 = raw_voxels * voxel_vol_mm3

    mask_for_png = pred_np
    if args.clean:
        min_vox = args.min_vol_mm3 / voxel_vol_mm3

        labeled, n = ndi.label(pred_np > 0.5)
        if n > 0:
            sizes = ndi.sum(pred_np, labeled, index=range(1, n + 1))
            keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_vox]
            mask_clean = np.isin(labeled, keep_labels)
        else:
            mask_clean = pred_np > 0.5

        if args.max_elongation and mask_clean.any():
            labeled, n = ndi.label(mask_clean)
            keep_labels = []
            for lbl in range(1, n + 1):
                coords = np.argwhere(labeled == lbl)
                if coords.shape[0] == 0:
                    continue
                coords_centered = coords - coords.mean(axis=0, keepdims=True)
                cov = np.cov(coords_centered.T)
                evals, _ = np.linalg.eigh(cov)
                evals = np.maximum(evals, 1e-6)
                elong = float(np.sqrt(evals.max() / evals.min()))
                if elong <= args.max_elongation:
                    keep_labels.append(lbl)
            mask_clean = np.isin(labeled, keep_labels)

        if args.closing_iters > 0:
            mask_clean = ndi.binary_closing(mask_clean, iterations=args.closing_iters)
        mask_clean = ndi.binary_fill_holes(mask_clean)
        if args.opening_iters > 0:
            mask_clean = ndi.binary_opening(mask_clean, iterations=args.opening_iters)
        if args.dilate_iters > 0:
            mask_clean = ndi.binary_dilation(mask_clean, iterations=args.dilate_iters)

        mask_for_png = mask_clean.astype(np.uint8)
        out_clean = os.path.join(args.out_dir, f"{series_name}_mask_clean.nii.gz")
        nib.save(nib.Nifti1Image(mask_for_png.astype(np.float32), affine), out_clean)
        print(f"Saved cleaned mask to: {out_clean}")
    else:
        mask_clean = mask_for_png

    clean_voxels = float((mask_for_png > 0.5).sum())
    clean_mm3 = clean_voxels * voxel_vol_mm3

    if args.metrics_csv:
        import csv

        csv_path = args.metrics_csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "series",
                        "series_dir",
                        "raw_voxels",
                        "raw_mm3",
                        "clean_voxels",
                        "clean_mm3",
                        "min_vol_mm3",
                        "closing_iters",
                        "opening_iters",
                        "dilate_iters",
                        "max_elongation",
                    ]
                )
            w.writerow(
                [
                    series_name,
                    os.path.abspath(args.series_dir),
                    raw_voxels,
                    raw_mm3,
                    clean_voxels,
                    clean_mm3,
                    args.min_vol_mm3 if args.clean else "",
                    args.closing_iters if args.clean else "",
                    args.opening_iters if args.clean else "",
                    args.dilate_iters if args.clean else "",
                    args.max_elongation if args.clean else "",
                ]
            )
        print(f"Appended metrics to: {csv_path}")

    if args.png:
        import matplotlib.pyplot as plt  # local import to avoid forcing display backend

        vol_np_resampled = vol_t.cpu().numpy()[0, 0]  # resampled, clipped volume
        nz = np.argwhere(mask_for_png > 0)

        slices_to_save: list[int] = []
        if nz.size > 0:
            z_vals = nz[:, 0]
            z_med = int(np.median(z_vals))
            slices_to_save.append(z_med)
            unique_z = np.unique(z_vals)
            areas = [int((mask_for_png[z] > 0).sum()) for z in unique_z]
            z_max = int(unique_z[int(np.argmax(areas))])
            slices_to_save.append(z_max)
            z_q25 = int(np.quantile(z_vals, 0.25))
            z_q75 = int(np.quantile(z_vals, 0.75))
            slices_to_save.extend([z_q25, z_q75])
        else:
            center = vol_np_resampled.shape[0] // 2
            slices_to_save.extend(
                [center, max(0, center - 10), min(vol_np_resampled.shape[0] - 1, center + 10), max(0, center - 20)]
            )

        clean_slices: list[int] = []
        for z in slices_to_save:
            zc = max(0, min(vol_np_resampled.shape[0] - 1, int(z)))
            if zc not in clean_slices:
                clean_slices.append(zc)

        for z_idx in clean_slices[:4]:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(vol_np_resampled[z_idx], cmap="gray")
            if nz.size > 0:
                ax.imshow(mask_for_png[z_idx], cmap="jet", alpha=0.35)
            ax.set_title(f"Series {series_name}\nSlice z={z_idx}")
            ax.axis("off")
            png_path = os.path.join(args.out_dir, f"{series_name}_z{z_idx}.png")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"Saved overlay PNG to: {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


