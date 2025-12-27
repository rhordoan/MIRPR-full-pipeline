#!/usr/bin/env python3
"""
Normalize existing radiomics JSON files to strict JSON:
- Parse with Python's permissive parser (accepts NaN/Infinity).
- Rewrite with allow_nan=False (converts NaN/Inf to null) and atomic replace.

Skips files that cannot be parsed (likely truncated).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from typing import Any, List, Optional


def sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [sanitize(v) for v in x]
    if isinstance(x, str):
        # Many older outputs stored numeric features as strings; convert when safe.
        try:
            y = float(x)
            if math.isnan(y) or math.isinf(y):
                return None
            return y
        except Exception:  # noqa: BLE001
            return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing per-series radiomics *.json files")
    args = ap.parse_args(argv)

    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".json")]
    fixed = 0
    skipped = 0
    for fp in sorted(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)  # permissive parser
        except Exception:  # noqa: BLE001
            skipped += 1
            continue

        data = sanitize(data)
        fd, tmp = tempfile.mkstemp(prefix=".tmp_norm_", suffix=".json", dir=os.path.dirname(fp))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, allow_nan=False)
        os.replace(tmp, fp)
        fixed += 1

    print(json.dumps({"fixed": fixed, "skipped": skipped, "dir": args.dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


