#!/usr/bin/env python3
"""
Derive usable labels for TCIA TCGA-LUAD imaging by joining:

- TCIA series metadata (from the folder structure + per-series `series.json`)
- GDC clinical/case metadata (via GDC /cases API)
- Optional RNA-seq expression proxy labels (MKI67 as Ki-67 proxy) if STAR-counts files are downloaded
- Optional somatic mutation labels if a MAF file is provided (EGFR/KRAS)

Outputs a single CSV at the *series* level so you can merge it with
radiomics features (which are also produced per DICOM series).

Notes:
- TCIA imaging alone does NOT include EGFR/KRAS/Ki-67 labels.
- For Ki-67, we typically proxy using MKI67 RNA expression from GDC/TCGA.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"


@dataclass(frozen=True)
class SeriesRow:
    series_uid: str
    patient_id: str
    study_uid: str
    modality: str | None = None
    description: str | None = None
    image_count: int | None = None
    file_size: int | None = None
    series_dir: str | None = None


def _http_post_json(url: str, payload: dict, timeout_s: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "vista-3d-labels/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_json_if_exists(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _na(x: Any) -> Any:
    """
    Normalize TCGA clinical placeholders like:
      [Not Available], [Not Applicable], [Unknown], empty strings
    """
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        if s.startswith("[") and s.endswith("]"):
            return None
        if s.lower() in {"na", "n/a", "null", "none"}:
            return None
        return s
    return x


def load_tcga_clinical_patient_table(path: str) -> Dict[str, dict]:
    """
    Loads TCGA clinical "patient" table from the legacy nationwidechildrens export.

    File format:
      - line 1: canonical column names
      - line 2: alternate column names
      - line 3: CDE IDs
      - line 4+: data rows
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        header1 = f.readline().rstrip("\n").split("\t")
        _ = f.readline()  # alt header
        _ = f.readline()  # CDE IDs
        r = csv.DictReader(f, fieldnames=header1, delimiter="\t")
        out: Dict[str, dict] = {}
        for row in r:
            barcode = _na(row.get("bcr_patient_barcode") or row.get("patient_id"))
            if not barcode:
                continue
            out[str(barcode)] = {k: _na(v) for k, v in row.items()}
        return out


def iter_tcia_series_rows(tcia_collection_dir: str) -> Iterable[SeriesRow]:
    """
    Walks the deterministic TCIA downloader layout:
      <collection>/<PatientID>/<StudyUID>/<SeriesUID>/
    and uses series.json when present.
    """
    for patient_id in sorted(os.listdir(tcia_collection_dir)):
        pdir = os.path.join(tcia_collection_dir, patient_id)
        if not os.path.isdir(pdir):
            continue
        for study_uid in sorted(os.listdir(pdir)):
            sdir = os.path.join(pdir, study_uid)
            if not os.path.isdir(sdir):
                continue
            for series_uid in sorted(os.listdir(sdir)):
                ser_dir = os.path.join(sdir, series_uid)
                if not os.path.isdir(ser_dir):
                    continue
                meta = _load_json_if_exists(os.path.join(ser_dir, "series.json")) or {}
                yield SeriesRow(
                    series_uid=series_uid,
                    patient_id=meta.get("patient_id") or patient_id,
                    study_uid=meta.get("study_uid") or study_uid,
                    modality=meta.get("modality"),
                    description=meta.get("description"),
                    image_count=meta.get("image_count"),
                    file_size=meta.get("file_size"),
                    series_dir=ser_dir,
                )


def load_outputs_series_uids(outputs_dir: str) -> set[str]:
    """
    Uses inference outputs naming convention:
      <series_uid>_ct_resampled.nii.gz
      <series_uid>_mask_clean.nii.gz
    """
    uids: set[str] = set()
    for p in glob.glob(os.path.join(outputs_dir, "*_ct_resampled.nii.gz")):
        uids.add(os.path.basename(p).replace("_ct_resampled.nii.gz", ""))
    for p in glob.glob(os.path.join(outputs_dir, "*_mask_clean.nii.gz")):
        uids.add(os.path.basename(p).replace("_mask_clean.nii.gz", ""))
    return uids


def fetch_gdc_cases_by_submitter_ids(
    submitter_ids: List[str],
    project_id: str = "TCGA-LUAD",
    page_size: int = 2000,
) -> Dict[str, dict]:
    """
    Returns mapping: case_submitter_id (TCGA-XX-XXXX) -> case payload.
    """
    # The GDC filter must not exceed some size; if huge, chunk it.
    fields = [
        "case_id",
        "submitter_id",
        "project.project_id",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "diagnoses.age_at_diagnosis",
        "diagnoses.tumor_stage",
        "diagnoses.tumor_grade",
        "diagnoses.primary_diagnosis",
        "diagnoses.morphology",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "exposures.tobacco_smoking_status",
        "exposures.pack_years_smoked",
    ]

    def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
        for i in range(0, len(xs), n):
            yield xs[i : i + n]

    out: Dict[str, dict] = {}
    for chunk in _chunks(sorted(set(submitter_ids)), 200):
        filt = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "submitter_id", "value": chunk}},
                {"op": "in", "content": {"field": "project.project_id", "value": [project_id]}},
            ],
        }
        frm = 0
        while True:
            payload = {"filters": filt, "fields": ",".join(fields), "format": "JSON", "size": page_size, "from": frm}
            resp = _http_post_json(GDC_CASES_ENDPOINT, payload, timeout_s=120)
            hits = resp.get("data", {}).get("hits", []) or []
            for h in hits:
                sid = h.get("submitter_id")
                if sid:
                    out[sid] = h
            total = resp.get("data", {}).get("pagination", {}).get("total", len(hits))
            frm += len(hits)
            if not hits or frm >= total:
                break
    return out


def _first_nonempty(items: list[dict], key: str) -> Any:
    for it in items:
        v = it.get(key)
        if v is not None and v != "":
            return v
    return None


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return None


def _parse_yes_no(x: Any) -> bool | None:
    s = _na(x)
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"yes", "y", "true", "1", "mutated", "mutation detected", "detected", "positive"}:
        return True
    if s in {"no", "n", "false", "0", "wildtype", "wild-type", "wt", "not detected", "negative"}:
        return False
    # sometimes "performed" without result; treat as unknown
    return None


def extract_clinical_fields(case: dict | None, clinical_row: dict | None) -> dict:
    """
    Prefer the local TCGA clinical patient table (richer + includes EGFR/KRAS calls).
    Fall back to GDC /cases when patient-table data isn't available.
    """
    if clinical_row:
        # Core demographics / staging / outcomes
        age_years = _safe_int(clinical_row.get("age_at_initial_pathologic_diagnosis"))
        age_days = int(age_years * 365.25) if age_years is not None else None
        # Prefer "clinical_stage" then "pathologic_stage"
        stage = clinical_row.get("clinical_stage") or clinical_row.get("pathologic_stage") or clinical_row.get("ajcc_clinical_tumor_stage") or clinical_row.get("ajcc_pathologic_tumor_stage")
        vital_status = clinical_row.get("vital_status")
        days_to_death = _safe_int(clinical_row.get("days_to_death") or clinical_row.get("death_days_to"))
        days_to_last_follow_up = _safe_int(clinical_row.get("days_to_last_followup") or clinical_row.get("last_contact_days_to"))

        pack_years = clinical_row.get("number_pack_years_smoked") or clinical_row.get("tobacco_smoking_pack_years_smoked")
        try:
            pack_years = float(pack_years) if pack_years is not None else None
        except Exception:  # noqa: BLE001
            pack_years = None

        # Mutation calls (available in this table for many LUAD cases)
        egfr_raw = clinical_row.get("egfr_mutation_result") or clinical_row.get("egfr_mutation_status") or clinical_row.get("egfr_mutation_identified")
        kras_raw = clinical_row.get("kras_mutation_found")
        egfr_mut = _parse_yes_no(egfr_raw)
        kras_mut = _parse_yes_no(kras_raw)

        return {
            # GDC identifiers (optional, may still be populated from /cases if present)
            "gdc_case_id": case.get("case_id") if case else None,
            "gdc_project_id": (case.get("project") or {}).get("project_id") if case else None,
            "gender": clinical_row.get("gender"),
            "race": clinical_row.get("race"),
            "ethnicity": clinical_row.get("ethnicity"),
            "age_at_diagnosis_days": age_days,
            "age_at_diagnosis_years": age_years,
            "tumor_stage": stage,
            "tumor_grade": clinical_row.get("tumor_grade"),
            "primary_diagnosis": clinical_row.get("histologic_diagnosis") or clinical_row.get("diagnosis") or clinical_row.get("histological_type"),
            "morphology": clinical_row.get("icd_o_3_histology"),
            "vital_status": vital_status,
            "days_to_death": days_to_death,
            "days_to_last_follow_up": days_to_last_follow_up,
            "tobacco_smoking_status": clinical_row.get("tobacco_smoking_history") or clinical_row.get("tobacco_smoking_history_indicator"),
            "pack_years_smoked": pack_years,
            # Mutation flags derived from the legacy clinical table
            "egfr_mutated": egfr_mut,
            "kras_mutated": kras_mut,
            "egfr_mutation_raw": egfr_raw,
            "kras_mutation_raw": kras_raw,
        }

    if not case:
        return {
            "gdc_case_id": None,
            "gdc_project_id": None,
            "gender": None,
            "race": None,
            "ethnicity": None,
            "age_at_diagnosis_days": None,
            "age_at_diagnosis_years": None,
            "tumor_stage": None,
            "tumor_grade": None,
            "primary_diagnosis": None,
            "morphology": None,
            "vital_status": None,
            "days_to_death": None,
            "days_to_last_follow_up": None,
            "tobacco_smoking_status": None,
            "pack_years_smoked": None,
            "egfr_mutated": None,
            "kras_mutated": None,
            "egfr_mutation_raw": None,
            "kras_mutation_raw": None,
        }

    demo = case.get("demographic") or {}
    diagnoses = case.get("diagnoses") or []
    exposures = case.get("exposures") or []

    dx0 = diagnoses[0] if diagnoses else {}
    age_days = _safe_int(dx0.get("age_at_diagnosis"))
    age_years = (age_days / 365.25) if age_days is not None else None

    # Some fields may appear only in later diagnoses; choose first non-empty if possible.
    tumor_stage = _first_nonempty(diagnoses, "tumor_stage") or dx0.get("tumor_stage")
    tumor_grade = _first_nonempty(diagnoses, "tumor_grade") or dx0.get("tumor_grade")
    primary_dx = _first_nonempty(diagnoses, "primary_diagnosis") or dx0.get("primary_diagnosis")
    morphology = _first_nonempty(diagnoses, "morphology") or dx0.get("morphology")
    vital_status = _first_nonempty(diagnoses, "vital_status") or dx0.get("vital_status")

    days_to_death = _safe_int(_first_nonempty(diagnoses, "days_to_death"))
    days_to_last_follow_up = _safe_int(_first_nonempty(diagnoses, "days_to_last_follow_up"))

    smoking = _first_nonempty(exposures, "tobacco_smoking_status")
    pack_years = _first_nonempty(exposures, "pack_years_smoked")
    try:
        pack_years = float(pack_years) if pack_years is not None and pack_years != "" else None
    except Exception:  # noqa: BLE001
        pack_years = None

    return {
        "gdc_case_id": case.get("case_id"),
        "gdc_project_id": (case.get("project") or {}).get("project_id"),
        "gender": demo.get("gender"),
        "race": demo.get("race"),
        "ethnicity": demo.get("ethnicity"),
        "age_at_diagnosis_days": age_days,
        "age_at_diagnosis_years": age_years,
        "tumor_stage": tumor_stage,
        "tumor_grade": tumor_grade,
        "primary_diagnosis": primary_dx,
        "morphology": morphology,
        "vital_status": vital_status,
        "days_to_death": days_to_death,
        "days_to_last_follow_up": days_to_last_follow_up,
        "tobacco_smoking_status": smoking,
        "pack_years_smoked": pack_years,
        "egfr_mutated": None,
        "kras_mutated": None,
        "egfr_mutation_raw": None,
        "kras_mutation_raw": None,
    }


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _find_gdc_downloaded_file(expr_files_dir: str, file_id: str, file_name: str) -> str | None:
    """
    gdc-client typically downloads as: <dir>/<file_id>/<file_name>
    but we also support any recursive layout.
    """
    p1 = os.path.join(expr_files_dir, file_id, file_name)
    if os.path.exists(p1):
        return p1
    # fallback: search (can be slow if huge, but expression dir is usually manageable)
    hits = glob.glob(os.path.join(expr_files_dir, "**", file_id, file_name), recursive=True)
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join(expr_files_dir, "**", file_name), recursive=True)
    for h in hits:
        # prefer paths containing file_id
        if file_id in h:
            return h
    return hits[0] if hits else None


def _parse_star_counts_tsv_for_gene(path: str, gene_symbol: str = "MKI67") -> tuple[float | None, str | None]:
    """
    Parses GDC `*.rna_seq.augmented_star_gene_counts.tsv` for a given gene symbol.
    Returns (value, metric_name).
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        # Sometimes comment/header lines precede the header; handle that.
        while header and header[0].startswith("#"):
            header = f.readline().rstrip("\n").split("\t")
        if not header or len(header) < 2:
            return None, None

        # Find a reasonable numeric column to use.
        preferred_cols = [
            "tpm_unstranded",
            "fpkm_uq_unstranded",
            "fpkm_unstranded",
            "unstranded",
        ]
        col = None
        for c in preferred_cols:
            if c in header:
                col = c
                break
        if col is None:
            # fall back to the last column
            col = header[-1]

        col_idx = header.index(col)
        gene_name_idx = header.index("gene_name") if "gene_name" in header else None

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if gene_name_idx is None or len(parts) <= max(col_idx, gene_name_idx):
                continue
            if parts[gene_name_idx] == gene_symbol:
                try:
                    return float(parts[col_idx]), col
                except Exception:  # noqa: BLE001
                    return None, col
    return None, col


def load_mki67_expression(
    expr_index_csv: str,
    expr_files_dir: str,
    prefer_sample_type: str = "Primary Tumor",
    gene_symbol: str = "MKI67",
) -> Dict[str, dict]:
    """
    Returns mapping: case_submitter_id -> {"mki67": float|None, "mki67_metric": str|None, "expr_file_id": ..., ...}
    """
    if not os.path.exists(expr_index_csv) or not os.path.isdir(expr_files_dir):
        return {}

    rows = _read_csv(expr_index_csv)
    # choose one expression file per case (prefer Primary Tumor)
    by_case: Dict[str, list[dict]] = {}
    for r in rows:
        cid = r.get("case_submitter_id") or r.get("case_submitter") or r.get("case")  # tolerate older schemas
        if not cid:
            continue
        by_case.setdefault(cid, []).append(r)

    out: Dict[str, dict] = {}
    for cid, items in by_case.items():
        chosen = None
        for it in items:
            if (it.get("sample_type") or "") == prefer_sample_type:
                chosen = it
                break
        if chosen is None:
            chosen = items[0]
        file_id = chosen.get("file_id") or ""
        file_name = chosen.get("file_name") or ""
        if not file_id or not file_name:
            out[cid] = {"mki67": None, "mki67_metric": None, "expr_file_id": file_id, "expr_file_name": file_name}
            continue

        fp = _find_gdc_downloaded_file(expr_files_dir, file_id, file_name)
        if not fp:
            out[cid] = {"mki67": None, "mki67_metric": None, "expr_file_id": file_id, "expr_file_name": file_name}
            continue

        v, metric = _parse_star_counts_tsv_for_gene(fp, gene_symbol=gene_symbol)
        out[cid] = {
            "mki67": v,
            "mki67_metric": metric,
            "expr_file_id": file_id,
            "expr_file_name": file_name,
            "expr_sample_type": chosen.get("sample_type"),
        }
    return out


def load_mutation_flags_from_maf(maf_path: str) -> Dict[str, dict]:
    """
    Parses a MAF file (TSV) and returns mapping: case_submitter_id -> mutation flags.

    Expected columns:
      - Hugo_Symbol
      - Tumor_Sample_Barcode  (e.g. TCGA-XX-XXXX-01A-...)
    """
    out: Dict[str, dict] = {}
    with open(maf_path, "r", encoding="utf-8", newline="") as f:
        # skip comment lines
        pos = f.tell()
        line = f.readline()
        while line.startswith("#"):
            pos = f.tell()
            line = f.readline()
        f.seek(pos)
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            gene = (row.get("Hugo_Symbol") or "").strip()
            sample = (row.get("Tumor_Sample_Barcode") or row.get("Tumor_Sample_UUID") or "").strip()
            if not gene or not sample:
                continue
            case_submitter = sample.split("-")[0:3]
            if len(case_submitter) < 3:
                continue
            cid = "-".join(case_submitter)
            d = out.setdefault(cid, {"egfr_mutated": False, "kras_mutated": False})
            if gene.upper() == "EGFR":
                d["egfr_mutated"] = True
            if gene.upper() == "KRAS":
                d["kras_mutated"] = True
    return out


def write_csv(out_csv: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    # union of all keys
    keys: List[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tcia-collection-dir",
        default=None,
        help="Path to TCIA collection dir (default: <repo>/data/tcia/TCGA-LUAD)",
    )
    ap.add_argument(
        "--outputs-dir",
        default=None,
        help="If set, restrict to series found in outputs (by *_ct_resampled/_mask_clean).",
    )
    ap.add_argument("--out-csv", default=None, help="Output CSV path (default: <repo>/outputs/tcga_luad_labels.csv)")
    ap.add_argument("--project", default="TCGA-LUAD", help="GDC project id (default: TCGA-LUAD)")
    ap.add_argument(
        "--clinical-patient-txt",
        default=None,
        help="Optional path to the legacy TCGA clinical patient table (nationwidechildrens export). "
        "Default: <repo>/data/gdc/tcga_luad_clinical_xml_cache/*clinical_patient_luad.txt",
    )
    ap.add_argument(
        "--expr-index-csv",
        default=None,
        help="GDC expression index CSV (default: <repo>/data/gdc/tcga_luad_expression_for_download/index.csv)",
    )
    ap.add_argument(
        "--expr-files-dir",
        default=None,
        help="Directory containing downloaded expression files (gdc-client output). If missing, expression labels are skipped.",
    )
    ap.add_argument(
        "--maf",
        default=None,
        help="Optional path to a MAF file to derive EGFR/KRAS mutation flags. If omitted, mutation labels are left blank.",
    )
    args = ap.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tcia_collection_dir = args.tcia_collection_dir or os.path.join(repo_root, "data", "tcia", "TCGA-LUAD")
    out_csv = args.out_csv or os.path.join(repo_root, "outputs", "tcga_luad_labels.csv")
    outputs_dir = args.outputs_dir

    if not os.path.isdir(tcia_collection_dir):
        print(f"[error] TCIA collection dir not found: {tcia_collection_dir}", file=sys.stderr)
        return 2

    restrict_uids = None
    if outputs_dir:
        restrict_uids = load_outputs_series_uids(outputs_dir)
        print(f"Restricting to {len(restrict_uids)} series found in outputs: {outputs_dir}", flush=True)

    series_rows: List[SeriesRow] = []
    for s in iter_tcia_series_rows(tcia_collection_dir):
        if restrict_uids is not None and s.series_uid not in restrict_uids:
            continue
        series_rows.append(s)

    if not series_rows:
        print("[error] No series found (check --tcia-collection-dir and/or --outputs-dir).", file=sys.stderr)
        return 2

    patient_ids = sorted({s.patient_id for s in series_rows if s.patient_id})
    print(f"Found {len(series_rows)} series across {len(patient_ids)} patients", flush=True)

    # Load local clinical table if present (more complete than modern GDC fields)
    clinical_patient_txt = args.clinical_patient_txt
    if clinical_patient_txt is None:
        # Pick the first matching file in the cache directory (name includes a UUID prefix).
        cache_glob = os.path.join(repo_root, "data", "gdc", "tcga_luad_clinical_xml_cache", "*clinical_patient_luad.txt")
        hits = sorted(glob.glob(cache_glob))
        clinical_patient_txt = hits[0] if hits else None

    clinical_by_barcode: Dict[str, dict] = {}
    if clinical_patient_txt and os.path.exists(clinical_patient_txt):
        clinical_by_barcode = load_tcga_clinical_patient_table(clinical_patient_txt)
        print(f"Clinical: loaded patient table with {len(clinical_by_barcode)} rows from {clinical_patient_txt}", flush=True)
    else:
        print("Clinical: no local patient table found; will rely on GDC /cases only.", flush=True)

    # Fetch clinical info from GDC
    cases_by_submitter: Dict[str, dict] = {}
    try:
        cases_by_submitter = fetch_gdc_cases_by_submitter_ids(patient_ids, project_id=args.project)
        print(f"GDC /cases: matched {len(cases_by_submitter)}/{len(patient_ids)} patients", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[warn] GDC /cases HTTP error {e.code}; continuing without clinical labels.", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] GDC /cases error {type(e).__name__}; continuing without clinical labels.", file=sys.stderr)

    # Expression labels (MKI67)
    expr_index_csv = args.expr_index_csv or os.path.join(repo_root, "data", "gdc", "tcga_luad_expression_for_download", "index.csv")
    expr_files_dir = args.expr_files_dir or os.path.join(repo_root, "data", "gdc", "downloads")
    mki67_by_case = load_mki67_expression(expr_index_csv, expr_files_dir)
    if mki67_by_case:
        print(f"Expression: found MKI67 values for {sum(1 for v in mki67_by_case.values() if v.get('mki67') is not None)} cases", flush=True)
    else:
        print("Expression: skipped (no index CSV or no downloaded expression files dir found).", flush=True)

    # Mutation flags from MAF (optional)
    mut_by_case: Dict[str, dict] = {}
    if args.maf:
        if not os.path.exists(args.maf):
            print(f"[warn] MAF not found at {args.maf}; skipping mutation flags.", file=sys.stderr)
        else:
            mut_by_case = load_mutation_flags_from_maf(args.maf)
            print(f"Mutations: loaded flags from MAF for {len(mut_by_case)} cases", flush=True)

    # Compute MKI67 high/low within available values (median split)
    mki67_vals = [v["mki67"] for v in mki67_by_case.values() if v.get("mki67") is not None]
    mki67_median = median(mki67_vals) if mki67_vals else None

    out_rows: List[dict] = []
    for s in series_rows:
        case = cases_by_submitter.get(s.patient_id)
        clinical_row = clinical_by_barcode.get(s.patient_id) if clinical_by_barcode else None
        clinical = extract_clinical_fields(case, clinical_row)
        expr = mki67_by_case.get(s.patient_id) or {}
        muts = mut_by_case.get(s.patient_id) or {}

        row = {
            "series_uid": s.series_uid,
            "patient_id": s.patient_id,
            "study_uid": s.study_uid,
            "modality": s.modality,
            "description": s.description,
            "image_count": s.image_count,
            "file_size": s.file_size,
            "series_dir": s.series_dir,
            **clinical,
            # MKI67 expression proxy label (if available)
            "mki67_expr": expr.get("mki67"),
            "mki67_expr_metric": expr.get("mki67_metric"),
            "mki67_expr_sample_type": expr.get("expr_sample_type"),
            "mki67_expr_file_id": expr.get("expr_file_id"),
            "mki67_expr_file_name": expr.get("expr_file_name"),
            "mki67_high_median_split": (expr.get("mki67") is not None and mki67_median is not None and expr.get("mki67") >= mki67_median),
            "mki67_median_used": mki67_median,
            # Mutation flags: prefer clinical-table calls; optionally override/augment from MAF.
            # If MAF is provided, it will fill missing values; it won't overwrite non-null clinical-table calls.
            "egfr_mutated": clinical.get("egfr_mutated") if clinical.get("egfr_mutated") is not None else muts.get("egfr_mutated"),
            "kras_mutated": clinical.get("kras_mutated") if clinical.get("kras_mutated") is not None else muts.get("kras_mutated"),
        }
        out_rows.append(row)

    write_csv(out_csv, out_rows)
    print(json.dumps({"out_csv": out_csv, "rows": len(out_rows), "patients": len(patient_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


