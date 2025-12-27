#!/usr/bin/env python3
"""
Parallel runner for Vista3D inference over all DICOM series under a root.

Strategy:
- Enumerate series dirs containing at least one .dcm.
- Skip series that already produced a cleaned mask (mask_clean) if desired.
- Launch up to N concurrent subprocesses of run_inference_on_series.py.
- Each subprocess writes its own per-series metrics CSV to avoid file contention.
- After all complete, merge per-series metrics into a single CSV.

Example:
  python scripts/run_all_inference_parallel.py \
    --data-root /home/shadeform/models/vista-3d/data/tcia \
    --out-dir /home/shadeform/models/vista-3d/outputs \
    --metrics /home/shadeform/models/vista-3d/outputs/metrics.csv \
    --workers 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional


def find_series(data_root: str) -> List[str]:
    # Find unique directories that contain at least one DICOM
    series_dirs = set()
    for dcm in glob.glob(os.path.join(data_root, "**", "*.dcm"), recursive=True):
        series_dirs.add(os.path.dirname(dcm))
    return sorted(series_dirs)


def safe_name(series_dir: str) -> str:
    return series_dir.rstrip("/").replace("/", "__")


def run_series(series_dir: str, out_dir: str, per_metrics_dir: str, python_bin: str, args) -> tuple[str, int]:
    series_tag = safe_name(series_dir)
    metrics_path = os.path.join(per_metrics_dir, f"{series_tag}.csv")
    log_path = os.path.join(per_metrics_dir, f"{series_tag}.log")

    # Skip if mask_clean already exists (idempotent)
    series_name = os.path.basename(series_dir.rstrip("/"))
    mask_clean = os.path.join(out_dir, f"{series_name}_mask_clean.nii.gz")
    if os.path.exists(mask_clean):
        return series_dir, 0

    cmd = [
        python_bin,
        os.path.join(os.path.dirname(__file__), "run_inference_on_series.py"),
        "--series-dir",
        series_dir,
        "--out-dir",
        out_dir,
        "--png",
        "--clean",
        "--save-ct",
        "--min-vol-mm3",
        str(args.min_vol_mm3),
        "--closing-iters",
        str(args.closing_iters),
        "--opening-iters",
        str(args.opening_iters),
        "--dilate-iters",
        str(args.dilate_iters),
        "--metrics-csv",
        metrics_path,
    ]
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
    ap.add_argument("--min-vol-mm3", type=float, default=50.0)
    ap.add_argument("--closing-iters", type=int, default=1)
    ap.add_argument("--opening-iters", type=int, default=1)
    ap.add_argument("--dilate-iters", type=int, default=2)
    ap.add_argument("--max-elongation", type=float, default=None)
    ap.add_argument("--python-bin", default="/home/shadeform/.venvs/vista-3d/bin/python")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    per_metrics_dir = os.path.join(args.out_dir, "metrics_per_series")
    os.makedirs(per_metrics_dir, exist_ok=True)

    series_list = find_series(args.data_root)
    if not series_list:
        print("No series found.")
        return 1

    print(f"Found {len(series_list)} series. Running with {args.workers} workers.")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(run_series, sd, args.out_dir, per_metrics_dir, args.python_bin, args) for sd in series_list]
        for fut in as_completed(futs):
            series_dir, rc = fut.result()
            status = "ok" if rc == 0 else f"rc={rc}"
            print(f"[{status}] {series_dir}")
            results.append(rc)

    merge_metrics(per_metrics_dir, args.metrics)
    print(f"Merged metrics -> {args.metrics}")

    failed = sum(1 for r in results if r != 0)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


