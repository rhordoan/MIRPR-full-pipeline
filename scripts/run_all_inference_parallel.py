#!/usr/bin/env python3
"""
Parallel runner for Vista3D inference over DICOM series with filtering and resume.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency
    pydicom = None


def _count_dicoms(series_dir: str) -> int:
    try:
        return len(glob.glob(os.path.join(series_dir, "*.dcm")))
    except Exception:  # noqa: BLE001
        return 0


def _read_description(series_dir: str) -> str:
    meta_path = os.path.join(series_dir, "series.json")
    if not os.path.exists(meta_path):
        return ""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:  # noqa: BLE001
        return ""
    desc = meta.get("description") or meta.get("SeriesDescription") or ""
    return str(desc).strip()


def _should_exclude(description: str, exclude_substrs: List[str]) -> bool:
    d = (description or "").lower()
    return any(s and s.lower() in d for s in exclude_substrs)


def _body_part(series_dir: str) -> str:
    """Read BodyPartExamined (or return empty) from first DICOM without pixel load."""
    if pydicom is None:
        return ""
    try:
        files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
        if not files:
            return ""
        ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        return str(getattr(ds, "BodyPartExamined", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _is_lung_series(description: str, body_part: str, lung_substr: str) -> bool:
    needles = [s.strip().lower() for s in (lung_substr or "lung").split(",") if s.strip()]
    if not needles:
        return True
    d = (description or "").lower()
    b = (body_part or "").lower()
    return any(n in d or n in b for n in needles)


def _hu_sniff_lung(series_dir: str, hu_min: float = -1000, hu_max: float = -300, min_frac: float = 0.001) -> bool:
    """Lightweight voxel check: reads one slice, counts lung-like HU fraction."""
    if pydicom is None:
        return False
    try:
        files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
        if not files:
            return False
        mid = files[len(files) // 2]
        ds = pydicom.dcmread(mid, force=True)
        arr = ds.pixel_array.astype("float32")
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        hu = arr * slope + intercept
        frac = ((hu >= hu_min) & (hu <= hu_max)).mean()
        return frac >= min_frac
    except Exception:  # noqa: BLE001
        return False


def _hu_sniff_lung_multi(series_dir: str, min_frac: float = 0.003) -> bool:
    """Sample multiple slices for higher recall while staying light."""
    if pydicom is None:
        return False
    try:
        files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
        if not files:
            return False
        idxs = [0, len(files) // 4, len(files) // 2, (3 * len(files)) // 4, len(files) - 1]
        for idx in idxs:
            ds = pydicom.dcmread(files[idx], force=True)
            arr = ds.pixel_array.astype("float32")
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            hu = arr * slope + intercept
            lung_frac = ((hu >= -1000) & (hu <= -300)).mean()
            air_frac = ((hu >= -1200) & (hu <= -800)).mean()
            if lung_frac >= min_frac or (air_frac >= 0.001 and lung_frac >= 0.001):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _series_geometry(series_dir: str) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """Return (pixel_spacing_mean, slice_thickness, rows, cols)."""
    if pydicom is None:
        return (None, None, None, None)
    try:
        files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
        if not files:
            return (None, None, None, None)
        ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        spacing = getattr(ds, "PixelSpacing", None)
        px_spacing = None
        if spacing and len(spacing) >= 2:
            px_spacing = float(sum(map(float, spacing[:2]))) / 2.0
        slice_thickness = None
        if hasattr(ds, "SliceThickness"):
            try:
                slice_thickness = float(ds.SliceThickness)
            except Exception:  # noqa: BLE001
                slice_thickness = None
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        return (px_spacing, slice_thickness, rows, cols)
    except Exception:  # noqa: BLE001
        return (None, None, None, None)


def _elite_lung_score(series_dir: str, n_slices: int, description: str) -> Tuple[float, Dict[str, bool]]:
    """
    High-recall scoring for lung CT suitability.
    Returns (score, flags) where higher is better; never hard-rejects without giving a fallback score.
    """
    desc_l = (description or "").lower()
    desc_hits = ["lung", "chest", "thorax", "pe", "ctp", "ctac", "wb", "whole body", "inspiration", "expiration", "spiral", "b20", "b30", "b40", "b60", "routine", "std ctac", "ctac"]
    has_desc_hit = any(k in desc_l for k in desc_hits)

    hu_pass = _hu_sniff_lung_multi(series_dir, min_frac=0.003)
    px_spacing, slice_thk, rows, cols = _series_geometry(series_dir)

    spacing_coarse = px_spacing is not None and px_spacing > 2.5
    thk_coarse = slice_thk is not None and slice_thk > 3.5
    coverage_ok = False
    if rows and cols:
        coverage_ok = rows >= 256 or cols >= 256 or (px_spacing is not None and max(rows, cols) * px_spacing >= 220)

    # Base score
    score = float(n_slices)
    if hu_pass:
        score += 50.0
    if has_desc_hit:
        score += 20.0
    if coverage_ok:
        score += 10.0
    if spacing_coarse:
        score -= 20.0
    if thk_coarse:
        score -= 15.0

    flags = {
        "hu_pass": hu_pass,
        "desc_hit": has_desc_hit,
        "coverage_ok": coverage_ok,
        "spacing_coarse": spacing_coarse,
        "thk_coarse": thk_coarse,
    }
    return score, flags


def _stable_fraction_pick(key: str, frac: float) -> bool:
    if frac <= 0:
        return False
    if frac >= 1:
        return True
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return v < frac


def find_series(data_root: str, require_done: bool, exclude_desc: List[str], min_slices: int) -> List[Tuple[str, int]]:
    series_dirs = set()
    for dcm in glob.glob(os.path.join(data_root, "**", "*.dcm"), recursive=True):
        series_dirs.add(os.path.dirname(dcm))

    filtered: List[Tuple[str, int]] = []
    for sd in series_dirs:
        if require_done and not os.path.exists(os.path.join(sd, ".done")):
            continue
        n_slices = _count_dicoms(sd)
        if n_slices < min_slices:
            continue
        desc = _read_description(sd)
        if _should_exclude(desc, exclude_desc):
            continue
        filtered.append((sd, n_slices))
    return sorted(filtered, key=lambda x: x[0])


def find_series_with_lung_filter(
    data_root: str,
    require_done: bool,
    exclude_desc: List[str],
    min_slices: int,
    require_lung: bool,
    lung_substr: str,
) -> List[Tuple[str, int]]:
    base = find_series(data_root, require_done, exclude_desc, min_slices)
    if not require_lung:
        return base
    filtered: List[Tuple[str, int]] = []
    for sd, n_slices in base:
        desc = _read_description(sd)
        bp = _body_part(sd)
        if _is_lung_series(desc, bp, lung_substr) or _hu_sniff_lung(sd):
            filtered.append((sd, n_slices))
    return sorted(filtered, key=lambda x: x[0])


def find_series_elite_lung(
    data_root: str,
    require_done: bool,
    exclude_desc: List[str],
    min_slices: int,
) -> List[Tuple[str, int, float]]:
    """
    High-recall lung CT selector. Never drops a patient entirely; assigns a score to each series.
    Returns list of (series_dir, n_slices, score).
    """
    series_dirs = set()
    for dcm in glob.glob(os.path.join(data_root, "**", "*.dcm"), recursive=True):
        series_dirs.add(os.path.dirname(dcm))

    scored: List[Tuple[str, int, float]] = []
    for sd in series_dirs:
        if require_done and not os.path.exists(os.path.join(sd, ".done")):
            continue
        n_slices = _count_dicoms(sd)
        if n_slices < max(min_slices, 60):  # hard floor to avoid 2D / scouts
            continue
        desc = _read_description(sd)
        if _should_exclude(desc, exclude_desc):
            continue
        score, _flags = _elite_lung_score(sd, n_slices, desc)
        scored.append((sd, n_slices, score))
    return sorted(scored, key=lambda x: x[0])


def _choose_one_per_patient(series: List[Tuple[str, int]]) -> List[str]:
    by_patient: Dict[str, Tuple[str, int]] = {}
    for sd, n in series:
        parts = sd.strip("/").split("/")
        patient = parts[-3] if len(parts) >= 3 else parts[0]
        cur = by_patient.get(patient)
        if cur is None or n > cur[1]:
            by_patient[patient] = (sd, n)
    return [v[0] for v in by_patient.values()]


def _choose_per_patient_scored(series: List[Tuple[str, int, float]], max_per_patient: int = 1) -> List[str]:
    """
    Choose best (or top-N) series per patient based on score, with a fallback so no patient is dropped.
    """
    by_patient: Dict[str, List[Tuple[str, int, float]]] = {}
    for sd, n, score in series:
        parts = sd.strip("/").split("/")
        patient = parts[-3] if len(parts) >= 3 else parts[0]
        by_patient.setdefault(patient, []).append((sd, n, score))

    chosen: List[str] = []
    for patient, lst in by_patient.items():
        lst_sorted = sorted(lst, key=lambda x: x[2], reverse=True)
        top = lst_sorted[: max_per_patient]
        chosen.extend([sd for sd, _, _ in top])
    return chosen


def safe_name(series_dir: str) -> str:
    return series_dir.rstrip("/").replace("/", "__")


def run_series(series_dir: str, out_dir: str, per_metrics_dir: str, python_bin: str, args) -> tuple[str, int]:
    series_tag = safe_name(series_dir)
    metrics_path = os.path.join(per_metrics_dir, f"{series_tag}.csv")
    log_path = os.path.join(per_metrics_dir, f"{series_tag}.log")

    series_name = os.path.basename(series_dir.rstrip("/"))
    mask_clean = os.path.join(out_dir, f"{series_name}_mask_clean.nii.gz")
    ct_resampled = os.path.join(out_dir, f"{series_name}_ct_resampled.nii.gz")
    save_ct = _stable_fraction_pick(series_name, float(args.save_ct_fraction))

    if os.path.exists(mask_clean):
        if not (save_ct and not os.path.exists(ct_resampled)):
        return series_dir, 0

    cmd = [
        python_bin,
        os.path.join(os.path.dirname(__file__), "run_inference_on_series.py"),
        "--series-dir",
        series_dir,
        "--weights",
        args.weights,
        "--out-dir",
        out_dir,
        "--device",
        args.device,
    ]
    if args.clean:
        cmd.append("--clean")
    if args.amp:
        cmd.append("--amp")
    if args.png:
        cmd.append("--png")
    if save_ct:
        cmd.append("--save-ct")
    if args.min_vol_mm3 is not None:
        cmd.extend(["--min-vol-mm3", str(args.min_vol_mm3)])
    if args.closing_iters is not None:
        cmd.extend(["--closing-iters", str(args.closing_iters)])
    if args.opening_iters is not None:
        cmd.extend(["--opening-iters", str(args.opening_iters)])
    if args.dilate_iters is not None:
        cmd.extend(["--dilate-iters", str(args.dilate_iters)])
    if args.max_elongation is not None:
        cmd.extend(["--max-elongation", str(args.max_elongation)])

    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)
    return series_dir, proc.returncode


def merge_metrics(per_metrics_dir: str, merged_path: str) -> None:
    csv_files = sorted(glob.glob(os.path.join(per_metrics_dir, "*.csv")))
    if not csv_files:
        return
    header = None
    rows = []
    for fp in csv_files:
        with open(fp, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            h = next(r, None)
            if h is None:
                continue
            if header is None:
                header = h
            for row in r:
                rows.append(row)
    if header is None:
        return
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Root containing series subdirs with DICOMs")
    ap.add_argument("--out-dir", required=True, help="Output directory for masks/overlays/CT")
    ap.add_argument("--metrics", required=True, help="Merged metrics CSV output path")
    ap.add_argument("--workers", type=int, default=2, help="Concurrent series to process (default: 2)")
    ap.add_argument("--weights", required=True, help="Path to model weights checkpoint (.pth)")
    ap.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    ap.add_argument("--amp", action="store_true", help="Enable CUDA autocast (mixed precision)")
    ap.add_argument("--png", action="store_true", help="Save overlay PNGs (off by default)")
    ap.add_argument("--save-ct-fraction", type=float, default=1.0, help="Fraction of series to save CT resampled")
    ap.add_argument("--require-done", action="store_true", help="Only process series with .done marker")
    ap.add_argument("--exclude-desc", default="scout,topogram,ct scout,sag,cor,fusion,pet", help="Comma-separated substrings to exclude by SeriesDescription")
    ap.add_argument("--min-slices", type=int, default=30, help="Skip series with fewer slices")
    ap.add_argument("--one-per-patient", action="store_true", help="Keep only one series per patient (best by slices)")
    ap.add_argument("--clean", action="store_true", help="Pass --clean to run_inference_on_series.py")
    ap.add_argument("--min-vol-mm3", type=float, default=None)
    ap.add_argument("--closing-iters", type=int, default=None)
    ap.add_argument("--opening-iters", type=int, default=None)
    ap.add_argument("--dilate-iters", type=int, default=None)
    ap.add_argument("--max-elongation", type=float, default=None)
    ap.add_argument("--python-bin", default=sys.executable, help="Python interpreter to launch workers")
    ap.add_argument(
        "--require-lung",
        action="store_true",
        help="Only keep series whose description or BodyPartExamined mention lung",
    )
    ap.add_argument(
        "--lung-substr",
        default="lung,chest,thorax",
        help="Comma-separated substrings to identify lung series in description/body part",
    )
    ap.add_argument(
        "--lung-heuristic",
        action="store_true",
        help="Use high-recall lung heuristic (multi-slice HU sniff + geometry) instead of simple lung filter",
    )
    ap.add_argument(
        "--max-series-per-patient",
        type=int,
        default=1,
        help="When using lung heuristic, keep up to this many series per patient (default: 1)",
    )
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    per_metrics_dir = os.path.join(args.out_dir, "per_series")
    os.makedirs(per_metrics_dir, exist_ok=True)

    exclude_list = [s.strip() for s in args.exclude_desc.split(",") if s.strip()]
    if args.lung_heuristic:
        scored = find_series_elite_lung(
            args.data_root,
            args.require_done,
            exclude_list,
            args.min_slices,
        )
        if args.one_per_patient:
            series_dirs = _choose_per_patient_scored(scored, max_per_patient=int(args.max_series_per_patient))
        else:
            series_dirs = [s for s, _, _ in scored]
    else:
        series_dirs_counts = find_series_with_lung_filter(
            args.data_root,
            args.require_done,
            exclude_list,
            args.min_slices,
            args.require_lung,
            args.lung_substr,
        )
        if args.one_per_patient:
            series_dirs = _choose_one_per_patient(series_dirs_counts)
        else:
            series_dirs = [s for s, _ in series_dirs_counts]
    print(f"Found {len(series_dirs)} series after filtering")

    completed = 0
    failed = 0

    def _run(series_dir: str) -> tuple[str, int]:
        return run_series(series_dir, args.out_dir, per_metrics_dir, args.python_bin, args)

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(_run, sd) for sd in series_dirs]
        for fut in as_completed(futs):
            s, rc = fut.result()
            if rc == 0:
                completed += 1
            else:
                failed += 1
                print(f"[error] failed {s} rc={rc}", flush=True)
            if (completed + failed) % 10 == 0:
                print(f"Completed {completed+failed}/{len(series_dirs)} (ok={completed}, failed={failed})", flush=True)

    merge_metrics(per_metrics_dir, args.metrics)
    print(f"Done. ok={completed}, failed={failed}. Merged metrics -> {args.metrics}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
