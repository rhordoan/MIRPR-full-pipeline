#!/usr/bin/env python3
"""
Convenience wrapper to download the NSCLC Radiogenomics imaging dataset from TCIA.

TCIA collection name: "NSCLC Radiogenomics"
"""

from __future__ import annotations

import argparse
import os
import sys

from tcia_download_collection import main as tcia_main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None, help="Output root directory (default: <repo>/data/tcia)")
    p.add_argument("--modality", default="CT", help="Modality filter (default: CT)")
    p.add_argument("--workers", type=int, default=2, help="Parallel downloads. Default: 2")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between successful series downloads per worker")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--keep-zip", action="store_true")
    p.add_argument("--max-series", type=int, default=None, help="Optional cap for smoke-testing")
    args = p.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = args.out_dir or os.path.join(repo_root, "data", "tcia")

    return tcia_main(
        [
            "--collection",
            "NSCLC Radiogenomics",
            "--out-dir",
            out_dir,
            "--modality",
            args.modality,
            "--workers",
            str(args.workers),
            "--sleep",
            str(args.sleep),
            "--retries",
            str(args.retries),
            "--timeout",
            str(args.timeout),
            *(["--keep-zip"] if args.keep_zip else []),
            *(["--max-series", str(args.max_series)] if args.max_series is not None else []),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))




