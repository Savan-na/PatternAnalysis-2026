"""Real-data preflight for Train/Validation only; no test-array access.

Example:
    python -m hipmri_rewrite.audit --dataset-root /path/to/HipMRI_Study_open --split train --limit 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import LABELS, NUM_CLASSES
from .data import downsample, load_case, one_hot, to_3d_model_arrays, zscore_nonzero
from .splits import discover


def audit(root: str | Path, split: str, limit: int | None = None) -> dict:
    if split not in ("train", "validation"):
        raise ValueError("Only Train and Validation arrays may be audited before final freeze")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    cases = discover(root)
    selected = cases[split] if limit is None else cases[split][:limit]
    rows = []
    for case in selected:
        original = load_case(case)
        smaller = downsample(original)
        processed = zscore_nonzero(smaller)
        image, target, shape = to_3d_model_arrays(processed)
        encoded = one_hot(processed.mask)
        if not np.array_equal(encoded.argmax(axis=0).astype(np.uint8), processed.mask):
            raise RuntimeError(f"{case.identifier}: one-hot roundtrip mismatch")
        if not np.isfinite(image).all() or target.min() < 0 or target.max() >= NUM_CLASSES:
            raise RuntimeError(f"{case.identifier}: invalid model arrays")
        if tuple(processed.mask.shape) != shape:
            raise RuntimeError(f"{case.identifier}: model shape mismatch")
        rows.append({
            "case": case.identifier,
            "source_shape_xyz": list(original.image.shape),
            "downsampled_shape_xyz": list(processed.image.shape),
            "model_shape_dhw": list(image.shape),
            "source_spacing_mm": list(original.spacing_mm),
            "downsampled_spacing_mm": list(processed.spacing_mm),
            "source_counts": {name: int(np.count_nonzero(original.mask == i)) for i, name in enumerate(LABELS)},
            "downsampled_counts": {name: int(np.count_nonzero(processed.mask == i)) for i, name in enumerate(LABELS)},
        })
    return {"split": split, "total_discovered": {key: len(value) for key, value in cases.items()},
            "audited": len(rows), "complete_split_audit": limit is None, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON output inside the new project")
    args = parser.parse_args()
    result = audit(args.dataset_root, args.split, args.limit)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        project = Path(__file__).resolve().parents[2]
        output = args.output.resolve()
        if not output.is_relative_to(project):
            parser.error("Output must be inside the 20261017 project")
        if output.exists():
            parser.error("Refusing to overwrite an audit result")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
