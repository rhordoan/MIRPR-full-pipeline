#!/usr/bin/env python3
"""
Extract PyRadiomics features for a single (CT, mask) pair and write JSON.

This is designed to be called as a subprocess so that any native crashes
in SimpleITK/PyRadiomics are isolated to that one case.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from radiomics import featureextractor


def _json_sanitize(x: Any) -> Any:
    """
    Convert PyRadiomics outputs to strict JSON-serializable values.
    - Converts numpy scalars to Python scalars
    - Converts 0-d / single-element numpy arrays to Python scalars
    - Replaces NaN/Inf with None (null)
    """
    try:
        import numpy as np  # local import

        if isinstance(x, (np.generic,)):
            x = x.item()
        elif isinstance(x, np.ndarray):
            # PyRadiomics sometimes returns 0-d arrays or singletons; convert those to scalars.
            if x.shape == () or x.size == 1:
                x = x.item()
            else:
                x = x.tolist()
    except Exception:  # noqa: BLE001
        pass

    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]

    if isinstance(x, float):
        # Strict JSON does not allow NaN/Infinity.
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return x
    if isinstance(x, (int, str, bool)) or x is None:
        return x

    # Fallback: stringify unknown types
    return str(x)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    extractor = featureextractor.RadiomicsFeatureExtractor(args.params)
    res: Dict[str, Any] = extractor.execute(args.image, args.mask)
    feats = {k: v for k, v in res.items() if k.startswith(("original", "wavelet", "log"))}
    feats["series"] = args.series
    feats["mask_path"] = os.path.abspath(args.mask)
    feats["image_path"] = os.path.abspath(args.image)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # Write atomically to avoid truncated JSON if the process crashes mid-write.
    out_dir = os.path.dirname(args.out_json)
    sanitized = _json_sanitize(feats)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_radiomics_", suffix=".json", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sanitized, f, allow_nan=False)
        os.replace(tmp_path, args.out_json)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


