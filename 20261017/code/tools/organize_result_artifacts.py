"""Classify local comparison outputs without changing their contents.

Run after downloading legacy Rangpur outputs or regenerating plots.
Unknown files stay in place; conflicting destinations are never overwritten.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]


def destination(path: Path) -> Path:
    name = path.name
    if path.parent == ROOT / "results/figures/comparisons":
        group = ("training" if name.startswith("g5_three_arm_training_") else
                 "test" if name.startswith("g6_test_") else
                 "validation" if name.startswith(("g5_validation_", "g5_3d_validation_", "g5_three_arm_validation_")) else None)
    elif path.parent == ROOT / "results/case_studies/comparisons":
        group = ("validation/body" if name.startswith("g5_body_validation_") else
                 "validation/small_organs" if name.startswith("g5_validation_") else
                 "test/body" if name.startswith("g6_body_") else
                 "test/small_organs" if name.startswith("g6_test_") else None)
    else:
        group = None
    return path.parent / group / name if group else path


def main() -> None:
    moves = []
    for folder in ("figures", "case_studies"):
        for path in sorted((ROOT / "results" / folder / "comparisons").iterdir()):
            if path.is_file() and path.suffix in {".png", ".svg", ".json"}:
                target = destination(path)
                if target != path:
                    if target.exists():
                        raise FileExistsError(f"Inspect duplicate before moving: {target}")
                    moves.append((path, target, hashlib.sha256(path.read_bytes()).digest()))
    for path, target, digest in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        assert hashlib.sha256(target.read_bytes()).digest() == digest
        print(target.relative_to(ROOT))
    print(f"Classified {len(moves)} files; contents unchanged")


if __name__ == "__main__":
    main()
