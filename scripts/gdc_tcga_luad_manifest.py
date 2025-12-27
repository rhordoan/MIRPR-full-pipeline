#!/usr/bin/env python3
"""
Create a GDC manifest for TCGA-LUAD (open access) gene expression quantification files.

This does NOT require authentication for open-access expression quantification outputs,
but some file types in GDC are controlled and will require a token.

Typical usage:
  python scripts/gdc_tcga_luad_manifest.py --out-dir data/gdc --workflow "HTSeq - FPKM-UQ"
  gdc-client download -m data/gdc/manifest.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def _http_post_json(url: str, payload: dict, timeout_s: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "vista-3d-gdc/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _make_filter(project: str, data_type: str, workflow_type: Optional[str]) -> dict:
    content: list[dict] = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
    ]
    if workflow_type:
        content.append({"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow_type]}})
    return {"op": "and", "content": content}


def fetch_all_files(project: str, data_type: str, workflow_type: Optional[str], page_size: int = 2000) -> List[dict]:
    """
    Returns list of file "hits" from the GDC /files endpoint.
    """
    filt = _make_filter(project, data_type, workflow_type)
    fields = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "data_format",
        "data_type",
        "analysis.workflow_type",
        "cases.case_id",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.samples.sample_id",
        "cases.samples.submitter_id",
        "cases.samples.portions.portion_id",
        "cases.samples.portions.submitter_id",
    ]

    out: list[dict] = []
    frm = 0
    while True:
        payload = {
            "filters": filt,
            "fields": ",".join(fields),
            "format": "JSON",
            "size": page_size,
            "from": frm,
        }
        resp = _http_post_json(GDC_FILES_ENDPOINT, payload, timeout_s=120)
        hits = resp.get("data", {}).get("hits", [])
        out.extend(hits)
        total = resp.get("data", {}).get("pagination", {}).get("total", len(out))
        frm += len(hits)
        if not hits or frm >= total:
            break
    return out


def write_manifest(out_path: str, hits: List[dict]) -> None:
    """
    Writes a GDC manifest TSV compatible with gdc-client.
    The gdc-client manifest uses columns: id, filename, md5, size, state.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "filename", "md5", "size", "state"])
        for h in hits:
            w.writerow(
                [
                    h.get("file_id"),
                    h.get("file_name"),
                    h.get("md5sum") or "",
                    h.get("file_size") or "",
                    "",
                ]
            )


def write_index_csv(out_path: str, hits: List[dict]) -> None:
    """
    Writes a human-friendly index mapping file_id -> case/sample metadata.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file_id",
                "file_name",
                "workflow_type",
                "case_id",
                "case_submitter_id",
                "sample_type",
                "sample_id",
                "sample_submitter_id",
            ]
        )
        for h in hits:
            cases = h.get("cases") or []
            case0 = cases[0] if cases else {}
            samples = case0.get("samples") or []
            sample0 = samples[0] if samples else {}
            w.writerow(
                [
                    h.get("file_id"),
                    h.get("file_name"),
                    (h.get("analysis") or {}).get("workflow_type"),
                    case0.get("case_id"),
                    case0.get("submitter_id"),
                    sample0.get("sample_type"),
                    sample0.get("sample_id"),
                    sample0.get("submitter_id"),
                ]
            )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="Output directory, e.g. data/gdc")
    p.add_argument("--project", default="TCGA-LUAD", help="GDC project id (default: TCGA-LUAD)")
    p.add_argument(
        "--data-type",
        default="Gene Expression Quantification",
        help='GDC data_type filter (default: "Gene Expression Quantification")',
    )
    p.add_argument(
        "--workflow",
        default="STAR - Counts",
        help='analysis.workflow_type filter (default: "STAR - Counts"). Use "" to disable workflow filtering.',
    )
    args = p.parse_args(argv)

    workflow = args.workflow if args.workflow != "" else None

    try:
        hits = fetch_all_files(args.project, args.data_type, workflow)
    except urllib.error.HTTPError as e:
        print(f"GDC API error: HTTP {e.code}", file=sys.stderr)
        return 2

    if not hits:
        print("No files returned from GDC for the requested filters.", file=sys.stderr)
        return 1

    manifest_path = os.path.join(args.out_dir, "manifest.tsv")
    index_path = os.path.join(args.out_dir, "index.csv")
    write_manifest(manifest_path, hits)
    write_index_csv(index_path, hits)

    print(json.dumps({"hits": len(hits), "manifest": manifest_path, "index": index_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


