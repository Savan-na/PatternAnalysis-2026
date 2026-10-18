"""Build a verified source-only ZIP for manual transfer to Rangpur."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "code" / "transfer" / "20261017_source.zip"
MAP = PROJECT / "PROJECT_MAP.md"
EXCLUDE_PARTS = {"__pycache__", "artifacts", "checkpoints", "predictions"}


def main() -> None:
    entries = {}
    files = sorted(path for path in PROJECT.rglob("*") if path.is_file()
                   and path not in (OUTPUT, MAP) and not EXCLUDE_PARTS.intersection(path.relative_to(PROJECT).parts)
                   and not (path.relative_to(PROJECT).parts[0] in {"logs", "results"}
                            and path.name != "README.md")
                   and path.suffix not in {".pyc", ".pt", ".pth", ".nii"}
                   and not path.name.endswith(".nii.gz"))
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        shared_folders = ("logs/audits/", "logs/profiles/", "logs/training/",
                          "logs/validation/", "results/data_analysis/",
                          "results/data_analysis/historical_train/",
                          "results/data_analysis/g1_selected_cases/",
                          "results/data_analysis/g1_bladder_review/",
                          "results/data_analysis/g1_patient_screen/",
                          "results/tables/", "results/figures/",
                          "results/case_studies/", "results/reports/")
        arms = ("baseline_2d", "baseline_3d", "improved_3d")
        arm_folders = tuple(f"{category}/{arm}/" for category in
                            ("logs/training", "logs/validation", "results/tables",
                             "results/figures", "results/case_studies") for arm in arms)
        comparison_folders = tuple(f"results/{category}/comparisons/" for category in
                                   ("tables", "figures", "case_studies"))
        for folder in (*shared_folders, *arm_folders, *comparison_folders):
            archive.writestr(f"20261017/{folder}", "")
        for path in files:
            relative = path.relative_to(PROJECT).as_posix()
            payload = path.read_bytes()
            entries[relative] = hashlib.sha256(payload).hexdigest()
            archive.writestr(f"20261017/{relative}", payload)
        archive.writestr("20261017/TRANSFER_MANIFEST_SHA256.json",
                         json.dumps(entries, indent=2, sort_keys=True) + "\n")
    with ZipFile(OUTPUT) as archive:
        for relative, digest in entries.items():
            actual = hashlib.sha256(archive.read(f"20261017/{relative}")).hexdigest()
            if actual != digest:
                raise RuntimeError(f"ZIP integrity failure: {relative}")
    print(f"{OUTPUT} | {len(entries)} source files | SHA-256 {hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
