#!/usr/bin/env python3
"""
Download a TCIA (NBIA) collection via the public NBIA REST API.

This downloads per-series ZIPs from:
  https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=...

and extracts them into a deterministic folder structure:
  <out_dir>/<collection>/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm

The script is resume-friendly: if a series folder contains a ".done" file, it is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from tqdm import tqdm


NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


@dataclass(frozen=True)
class SeriesInfo:
    collection: str
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str | None = None
    description: str | None = None
    image_count: int | None = None
    file_size: int | None = None


def _http_get_json(url: str, timeout_s: int = 60) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "vista-3d-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def list_series(collection: str, modality: Optional[str] = None) -> list[SeriesInfo]:
    qs = {"Collection": collection}
    if modality:
        qs["Modality"] = modality
    url = f"{NBIA_BASE}/getSeries?{urllib.parse.urlencode(qs)}"
    items = _http_get_json(url)
    series: list[SeriesInfo] = []
    for it in items:
        series.append(
            SeriesInfo(
                collection=it.get("Collection") or collection,
                patient_id=it.get("PatientID") or "UNKNOWN_PATIENT",
                study_uid=it.get("StudyInstanceUID") or "UNKNOWN_STUDY",
                series_uid=it.get("SeriesInstanceUID"),
                modality=it.get("Modality"),
                description=it.get("SeriesDescription"),
                image_count=int(it["ImageCount"]) if it.get("ImageCount") is not None else None,
                file_size=int(it["FileSize"]) if it.get("FileSize") is not None else None,
            )
        )
    # Defensive: drop any malformed entries without a UID
    return [s for s in series if s.series_uid]


def _series_out_dir(out_dir: str, s: SeriesInfo) -> str:
    return os.path.join(out_dir, s.collection, s.patient_id, s.study_uid, s.series_uid)


def _download_and_extract_series(
    s: SeriesInfo,
    out_dir: str,
    keep_zip: bool,
    sleep_s: float,
    retries: int,
    timeout_s: int,
) -> tuple[SeriesInfo, bool, str]:
    """
    Returns: (series, ok, message)
    """
    series_dir = _series_out_dir(out_dir, s)
    done_marker = os.path.join(series_dir, ".done")
    if os.path.exists(done_marker):
        return s, True, "skipped"

    os.makedirs(series_dir, exist_ok=True)

    # download zip to a temp file so partials don't look "complete"
    tmp_dir = tempfile.mkdtemp(prefix="tcia_", dir=series_dir)
    zip_path = os.path.join(tmp_dir, "series.zip")

    try:
        url = f"{NBIA_BASE}/getImage?SeriesInstanceUID={urllib.parse.quote(s.series_uid)}"

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "vista-3d-downloader/1.0"})
                with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(zip_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < retries:
                    time.sleep(max(0.5, sleep_s) * (attempt + 1))
                else:
                    raise

        # Extract into series_dir
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(series_dir)

        if keep_zip:
            shutil.move(zip_path, os.path.join(series_dir, "series.zip"))

        # Write metadata for traceability
        meta_path = os.path.join(series_dir, "series.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "collection": s.collection,
                    "patient_id": s.patient_id,
                    "study_uid": s.study_uid,
                    "series_uid": s.series_uid,
                    "modality": s.modality,
                    "description": s.description,
                    "image_count": s.image_count,
                    "file_size": s.file_size,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        with open(done_marker, "w", encoding="utf-8") as f:
            f.write("ok\n")

        if sleep_s > 0:
            time.sleep(sleep_s)

        return s, True, "downloaded"
    except zipfile.BadZipFile:
        return s, False, "bad_zip"
    except urllib.error.HTTPError as e:
        return s, False, f"http_{e.code}"
    except Exception as e:  # noqa: BLE001
        return s, False, f"error_{type(e).__name__}"
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:  # noqa: BLE001
            pass


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", required=True, help="TCIA collection name (e.g. 'NSCLC Radiogenomics', 'TCGA-LUAD')")
    p.add_argument("--out-dir", required=True, help="Output root directory")
    p.add_argument("--modality", default=None, help="Optional modality filter (e.g. CT)")
    p.add_argument("--max-series", type=int, default=None, help="Optional cap for smoke-testing")
    p.add_argument("--workers", type=int, default=2, help="Parallel downloads (be nice to TCIA). Default: 2")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between successful series downloads per worker")
    p.add_argument("--retries", type=int, default=2, help="Retries per series download")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds per request")
    p.add_argument("--keep-zip", action="store_true", help="Keep the downloaded series.zip next to extracted DICOMs")
    args = p.parse_args(argv)

    series = list_series(args.collection, modality=args.modality)
    if args.max_series is not None:
        series = series[: args.max_series]

    if not series:
        print(f"No series found for collection={args.collection!r} modality={args.modality!r}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    ok = 0
    skipped = 0
    failed = 0

    # Precompute which ones are already done so the progress bar is meaningful.
    pending: list[SeriesInfo] = []
    for s in series:
        done_marker = os.path.join(_series_out_dir(args.out_dir, s), ".done")
        if os.path.exists(done_marker):
            skipped += 1
        else:
            pending.append(s)

    if not pending:
        print("All series already downloaded (nothing to do).")
        return 0

    desc = f"{args.collection}"
    if args.modality:
        desc += f" ({args.modality})"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                _download_and_extract_series,
                s,
                args.out_dir,
                args.keep_zip,
                args.sleep,
                args.retries,
                args.timeout,
            )
            for s in pending
        ]

        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc, unit="series"):
            s, is_ok, msg = fut.result()
            if is_ok:
                ok += 1
            else:
                failed += 1
                # Write a small failure marker for easier triage/resume
                series_dir = _series_out_dir(args.out_dir, s)
                os.makedirs(series_dir, exist_ok=True)
                with open(os.path.join(series_dir, ".failed"), "w", encoding="utf-8") as f:
                    f.write(msg + "\n")

    print(
        json.dumps(
            {
                "collection": args.collection,
                "modality": args.modality,
                "downloaded": ok,
                "skipped": skipped,
                "failed": failed,
                "out_dir": args.out_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())




