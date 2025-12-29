#!/usr/bin/env python3
"""
Batch-extract PyRadiomics features for all cleaned masks and matching CT volumes.

Expected files (per series) in outputs directory:
  <series>_mask_clean.nii.gz
  <series>_ct_resampled.nii.gz   (produced by run_inference_on_series.py with --save-ct)

Example:
  python scripts/run_pyradiomics_batch.py \
    --inputs /home/shadeform/models/vista-3d/outputs \
    --params /home/shadeform/models/vista-3d/radiomics_params.yaml \
    --out-csv /home/shadeform/models/vista-3d/outputs/radiomics_features.csv
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import subprocess
import sys
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Directory containing *_mask_clean.nii.gz and *_ct_resampled.nii.gz")
    ap.add_argument("--params", required=True, help="PyRadiomics params YAML")
    ap.add_argument("--out-csv", required=True, help="Output CSV path for aggregated features")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers (subprocesses). "
            "Tip: start with workers≈physical_cores/2 for heavy configs (Wavelet/LoG), "
            "or workers≈cores for fast configs (Original-only)."
        ),
    )
    ap.add_argument(
        "--itk-threads",
        type=int,
        default=1,
        help=(
            "Threads per worker for ITK/SimpleITK (PyRadiomics). "
            "When --workers > 1, keep this low (often 1) to avoid oversubscription."
        ),
    )
    ap.add_argument(
        "--lib-threads",
        type=int,
        default=1,
        help=(
            "Threads per worker for common numeric libs (OpenMP/BLAS/NumExpr). "
            "When --workers > 1, keep this low (often 1)."
        ),
    )
    ap.add_argument(
        "--per-series-dir",
        default=None,
        help="Optional dir to write per-series JSON (recommended for parallel runs). Default: <out-dir>/radiomics_per_series",
    )
    ap.add_argument(
        "--lock-file",
        default=None,
        help="Lock file path to prevent running multiple controllers at once (default: <out-dir>/radiomics_batch.lock)",
    )
    args = ap.parse_args(argv)

    mask_paths = sorted(glob.glob(os.path.join(args.inputs, "*_mask_clean.nii.gz")))
    print(f"Found {len(mask_paths)} masks in {args.inputs}", flush=True)
    jobs: List[Tuple[str, str, str]] = []
    for mask_path in mask_paths:
        series = os.path.basename(mask_path).replace("_mask_clean.nii.gz", "")
        image_path = os.path.join(args.inputs, f"{series}_ct_resampled.nii.gz")
        if not os.path.exists(image_path):
            print(f"Skipping {series}: CT not found at {image_path}", flush=True)
            continue
        jobs.append((series, image_path, mask_path))

    if not jobs:
        print("No series processed. Check input paths.", flush=True)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    per_series_dir = args.per_series_dir or os.path.join(out_dir, "radiomics_per_series")
    os.makedirs(per_series_dir, exist_ok=True)
    lock_file = args.lock_file or os.path.join(out_dir, "radiomics_batch.lock")

    # Acquire an exclusive lock file so we don't run multiple controllers concurrently.
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except Exception:  # noqa: BLE001
            return False

    for _ in range(2):
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            break
        except FileExistsError:
            # Check if lock is stale; if so, remove it and retry once.
            stale = False
            try:
                with open(lock_file, "r", encoding="utf-8") as f:
                    pid_s = (f.read() or "").strip()
                pid = int(pid_s) if pid_s else -1
                stale = pid <= 0 or not _pid_alive(pid)
            except Exception:  # noqa: BLE001
                stale = True

            if stale:
                with suppress(Exception):
                    os.remove(lock_file)
                continue

            print(f"Lock already exists at {lock_file} (pid {pid}). Another batch is running; exiting.", flush=True)
            return 2

    # Resume support: skip any series already written to per_series_dir.
    done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(per_series_dir, "*.json"))}
    pending = [(s, ip, mpth) for (s, ip, mpth) in jobs if s not in done]
    if done:
        print(f"Resuming: {len(done)} already done, {len(pending)} remaining", flush=True)

    def _run_subprocess(series: str, image_path: str, mask_path: str) -> tuple[str, int]:
        out_json = os.path.join(per_series_dir, f"{series}.json")
        out_log = os.path.join(per_series_dir, f"{series}.log")
        env = os.environ.copy()
        # Hard cap threads per subprocess (prevents --workers * many threads).
        # Tune with --itk-threads / --lib-threads.
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(int(args.itk_threads))
        lib_t = str(int(args.lib_threads))
        env["OMP_NUM_THREADS"] = lib_t
        env["OPENBLAS_NUM_THREADS"] = lib_t
        env["MKL_NUM_THREADS"] = lib_t
        env["NUMEXPR_NUM_THREADS"] = lib_t
        env["VECLIB_MAXIMUM_THREADS"] = lib_t

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "run_pyradiomics_one.py"),
            "--image",
            image_path,
            "--mask",
            mask_path,
            "--params",
            args.params,
            "--series",
            series,
            "--out-json",
            out_json,
        ]
        with open(out_log, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=lf, env=env)
        return series, proc.returncode

    try:
        if args.workers <= 1:
            print("Running sequential extraction (workers=1)", flush=True)
            for series, image_path, mask_path in pending:
                s, rc = _run_subprocess(series, image_path, mask_path)
                if rc != 0:
                    print(f"[error] failed {s} (rc={rc})", flush=True)
        else:
            print(f"Running via subprocesses with {args.workers} workers. Per-series JSON -> {per_series_dir}", flush=True)
            completed = 0
            failed = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_run_subprocess, s, ip, mpth) for (s, ip, mpth) in pending]
                for fut in as_completed(futs):
                    series, rc = fut.result()
                    if rc == 0:
                        completed += 1
                    else:
                        failed += 1
                        print(
                            f"[error] failed {series} (rc={rc}) see {os.path.join(per_series_dir, series + '.log')}",
                            flush=True,
                        )
                    if (completed + failed) % 10 == 0:
                        print(f"Completed {completed+failed}/{len(pending)} (ok={completed}, failed={failed})", flush=True)
    finally:
        with suppress(Exception):
            os.remove(lock_file)

    rows: List[Dict[str, float]] = []
    for fp in sorted(glob.glob(os.path.join(per_series_dir, "*.json"))):
        with open(fp, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        print("No features extracted (0 successful cases). Try fewer --workers (e.g., 8) and check inputs.", flush=True)
        return 1

    df = pd.DataFrame(rows)
    # Coerce numeric feature columns (many older JSONs may contain numeric strings).
    feature_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(("original", "wavelet", "log"))]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote features to {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



