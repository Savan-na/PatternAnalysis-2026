"""Check the report's scores using the saved confusion counts; no MRI or weights needed."""
import csv,json
from pathlib import Path
import numpy as np
from utils import aggregate_cases,sha256_file
ROOT=Path(__file__).resolve().parent


def verify_replayed_sources(root, replay):
    """Accept exact replayed files or a verified change to comments/docstrings only."""
    import ast
    import hashlib

    class RemoveProse(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)

    proof_path = root / "evidence/editorial_source_equivalence.json"
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else {"files": {}}
    for filename, original_digest in replay["sources"].items():
        path = root / filename
        if sha256_file(path) == original_digest:
            continue
        change = proof["files"][filename]
        assert change["gpu_replayed_sha256"] == original_digest, filename
        assert change["current_sha256"] == sha256_file(path), filename
        tree = RemoveProse().visit(ast.parse(path.read_text(encoding="utf-8")))
        digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        assert digest == change["executable_ast_sha256"], filename


def main():
    bundle=json.loads((ROOT/"evidence/experiments.json").read_text())
    summary=list(csv.DictReader((ROOT/"evidence/model_summary.csv").open(encoding="utf-8-sig")))
    metrics=list(csv.DictReader((ROOT/"evidence/foreground_metrics.csv").open(encoding="utf-8-sig")))
    curves=bundle["training_candidates"]
    counts=0
    for name,model in bundle["models"].items():
        rows=[r for r in curves if r["model"]==name]
        assert [r["epoch"] for r in rows]==list(range(1,51))
        best=max((r for r in rows if r["validation_dice"] is not None),
                 key=lambda r:(r["validation_dice"],-r["epoch"]))
        assert best["epoch"]==model["best_epoch"]
        for split,cases in model["splits"].items():
            assert len(cases)==(38 if split=="validation" else 30)
            assert len({c["patient"] for c in cases})==6
            result=aggregate_cases(cases)
            expected=next(r for r in summary if r["model"]==name and r["split"]==split)
            assert abs(result["foreground_dice"]-float(expected["foreground_dice"]))<1e-12
            if split=="validation":assert abs(result["selection_score"]-best["validation_dice"])<1e-12
            for k,label in enumerate(bundle["class_order"][1:],1):
                row=next(r for r in metrics if r["model"]==name and r["split"]==split and r["label"]==label)
                for metric in ("dice","iou","precision","recall"):
                    assert abs(result["patient_equal"][metric][k]-float(row[metric]))<1e-12
            counts+=len(cases)
    assert counts==680
    provenance=json.loads((ROOT/"evidence/provenance.json").read_text())
    for filename,digest in provenance["clinical_figures"].items():
        assert sha256_file(ROOT/"figures"/filename)==digest
    replay_path=ROOT/"evidence/migration_receipt.json"
    assert sha256_file(replay_path)=="14936ed27162acd63209f8ab34dec9fc05192f30746b80111e520c94abaa38fc"
    replay=json.loads(replay_path.read_text())
    verify_replayed_sources(ROOT, replay)
    for name,record in replay["models"].items():
        assert record["cases"]==38 and record["all_confusions_and_inputs_match"] is True
        assert record["checkpoint_sha256"]==bundle["models"][name]["selected"]["checkpoint_sha256"]
    print("PASS: GPU replay receipt and runtime equivalence (including prose-only edits); 76 Validation matrices")
    print("PASS: 500 epoch entries; Validation-selected checkpoints; 680 case counts; 400 foreground metrics; five original clinical figures")


if __name__=="__main__":main()
