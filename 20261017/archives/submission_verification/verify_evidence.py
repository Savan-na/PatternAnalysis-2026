"""Check the report's scores using the saved confusion counts; no MRI or weights needed."""
import csv,json
from pathlib import Path
import numpy as np
import sys
ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[2] / "recognition/HipMRI_UNet_s4906923"
sys.path.insert(0, str(ROOT))
from utils import aggregate_cases, sha256_file


def verify_replayed_sources(root, replay):
    """Accept exact replayed files or a verified change to comments/docstrings only."""
    import ast
    import hashlib

    class RemoveProse(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)

    proof_path = ARCHIVE / "editorial_source_equivalence.json"
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else {"files": {}}
    for filename, original_digest in replay["sources"].items():
        if filename == "verify_replay.py":
            # The original driver is preserved in Git; it is no longer submitted.
            import subprocess
            original = subprocess.check_output([
                "git", "show", proof["baseline_commit"] +
                ":recognition/HipMRI_UNet_s4906923/verify_replay.py"
            ], cwd=ROOT.parents[1])
            assert hashlib.sha256(original).hexdigest() == original_digest
            continue
        path = root / filename
        if sha256_file(path) == original_digest:
            continue
        change = proof["files"][filename]
        assert change["gpu_replayed_sha256"] == original_digest, filename
        assert change["current_sha256"] == sha256_file(path), filename
        tree = RemoveProse().visit(ast.parse(path.read_text(encoding="utf-8")))
        digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        assert digest == change["executable_ast_sha256"], filename



def verify_report_additions(bundle, provenance):
    """Recalculate the diagnostic contrasts and check their saved source identities."""
    research = ROOT.parents[1] / "20261017"
    review = provenance["report_diagnostic_summaries"]
    assert review["source_bundle_sha256"] == sha256_file(ROOT/"evidence/experiments.json")
    for name, expected in review["body_comparison"].items():
        model = bundle["models"][name]
        median = np.median([r["surfaces"]["Body"]["hd95_mm"] for r in model["splits"]["validation"]])
        assert abs(median-expected["validation_median_body_hd95_mm"])<1e-12
        for split, count in expected["distant_body_fp_voxels"].items():
            assert count == sum(r["far_background_to_body_voxels"] for r in model["splits"][split])
    standard = {r["case"]:r for r in bundle["models"]["Standard"]["splits"]["validation"]}
    assert all(r["surfaces"]["Body"]["hd95_mm"] > standard[r["case"]]["surfaces"]["Body"]["hd95_mm"]
               for r in bundle["models"]["v1"]["splits"]["validation"])
    for split, expected in review["patient_sensitivity"].items():
        standard=aggregate_cases(bundle["models"]["Standard"]["splits"][split])
        v3=aggregate_cases(bundle["models"]["v3"]["splits"][split])
        delta={p:float(np.mean(list(v3["patient_small_organ_dice"][p].values()))-
                       np.mean(list(standard["patient_small_organ_dice"][p].values())))
               for p in standard["patients"]}
        assert delta==expected["patient_deltas"]
        assert sum(v>0 for v in delta.values())==expected["patients_improving"]
        if split=="test":
            without={m:aggregate_cases([r for r in bundle["models"][m]["splits"][split] if r["patient"]!="M015"])
                     for m in ("Standard","v3")}
            assert abs(without["v3"]["selection_score"]-without["Standard"]["selection_score"]-
                       expected["delta_without_M015"])<1e-12
            assert sum(r["dice"][5]<.7 for r in v3["patients"].values())==expected["v3_patients_prostate_dice_below_07"]
    for expected in review["selected_four_case_inference_diagnostic"]:
        source=research/expected["research_path"]
        assert sha256_file(source)==expected["sha256"]
        saved=json.loads(source.read_text())
        assert saved["full_confusion_reproduced"]
        for protocol, value in expected["protocols"].items():
            raw=saved["protocols"][protocol]
            assert value["distant_body_fp_voxels"]==raw["background_to_body_at_least_20mm"]
            assert abs(value["volume_mean_small_organ_dice"]-np.mean(raw["overlap"]["dice"][3:6]))<1e-12
    expected=review["auxiliary_output_diagnostic"]
    source=research/expected["research_path"]
    assert sha256_file(source)==expected["source_sha256"]
    saved=json.loads(source.read_text())
    for output, value in expected["outputs"].items():
        rows=[dict(case=r["case"],patient=r["patient"],confusion=r["outputs"][output]["confusion"])
              for r in saved["cases"]]
        assert abs(aggregate_cases(rows)["selection_score"]-value["patient_small_organ_score"])<1e-12
        assert abs(np.median([r["outputs"][output]["body_hd95_mm"] for r in saved["cases"]])-
                   value["median_body_hd95_mm"])<1e-12
    for key in ("raw_audit","preprocessing_audit"):
        value=provenance["pretraining_review"][key]
        source=research/value["research_path"]
        assert sha256_file(source)==value["sha256"]
        saved=json.loads(source.read_text())
        assert len(saved["cases"])==value["cases"] and saved["case_error_count"]==0
        if key=="raw_audit":
            assert all(sum(r["class_counts"][label] for r in saved["cases"])==count
                       for label,count in value["class_voxel_counts"].items())
        else:
            for label, stats in value["small_organ_physical_volume_ratio"].items():
                ratios=[r["physical_volume_ratio"][label] for r in saved["cases"]]
                assert min(ratios)==stats["minimum"] and max(ratios)==stats["maximum"]
                assert np.median(ratios)==stats["median"]
    for value in provenance["pretraining_review"]["optimizer_and_capacity_sources"]:
        assert sha256_file(research/value["research_path"])==value["sha256"]
    figure=provenance["method_figure"]
    assert sha256_file(ROOT/figure["filename"])==figure["sha256"]
    assert sha256_file(ROOT/"modules.py")==figure["architecture_source_sha256"]
    assert sha256_file(research/"code/tools/build_submission_evidence.py")==figure["generator_sha256"]
    print("PASS: pretraining source audits, Body counts, patient sensitivity, fixed-head/window diagnostics and method figure")


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
    spatial=json.loads((ROOT/"evidence/spatial_context_summary.json").read_text())
    assert spatial["source_sha256"]==sha256_file(ROOT/"evidence/experiments.json")
    for name,record in spatial["models"].items():
        grouped={}
        for row in bundle["models"][name]["splits"]["test"]:
            grouped.setdefault(row["patient"],[]).append(row)
        for region in ("apex","central","base"):
            value=float(np.mean([np.mean([r["prostate_regions"][region] for r in rows]) for rows in grouped.values()]))
            assert abs(value-record[region+"_slice_dice"])<1e-12
        gap=float(np.mean([np.mean([r["geometry"]["Prostate"]["prediction"]["internal_empty_z_slices"]>0 for r in rows]) for rows in grouped.values()]))
        transition=float(np.mean([np.mean([abs(r["geometry"]["Prostate"]["area_transition"]["prediction"]-r["geometry"]["Prostate"]["area_transition"]["truth"]) for r in rows]) for rows in grouped.values()]))
        rates=[]
        for rows in grouped.values():
            matrix=sum(np.asarray(r["confusion"]) for r in rows)
            rates.append(float((matrix[3,5]+matrix[4,5])/matrix[[3,4],:].sum()))
        assert abs(gap-record["prediction_internal_gap_visit_fraction"])<1e-12
        assert abs(transition-record["area_transition_absolute_error"])<1e-12
        assert abs(float(np.mean(rates))-record["true_oar_labelled_prostate_fraction"])<1e-12
    print("PASS: 18 end-region, continuity and adjacent-organ results recomputed")
    provenance=json.loads((ROOT/"evidence/provenance.json").read_text())
    verify_report_additions(bundle, provenance)
    additions=provenance["documentation_review_20261022"]
    for key in ("cohort_resource_summary", "saved_record_example", "local_environment"):
        value=additions[key]
        assert sha256_file(ROOT/value["path"])==value["sha256"]
    cohort=json.loads((ROOT/"evidence/cohort_inference_summary.json").read_text())
    assert cohort["source_sha256"]==sha256_file(ROOT/"evidence/experiments.json")
    for row in cohort["rows"]:
        cases=bundle["models"][row["model"]]["splits"][row["split"]]
        times=[c["inference_seconds_single_pass"] for c in cases]
        peaks=[c["inference_peak_allocated_bytes"]/2**30 for c in cases]
        assert row["volumes"]==len(cases) and row["patients"]==len({c["patient"] for c in cases})
        for key,q in (("median",.5),("q1",.25),("q3",.75),("p95",.95)):
            assert abs(row["seconds_"+key]-np.quantile(times,q,method="linear"))<1e-12
        assert row["peak_allocated_gib_median"]==np.median(peaks)
        assert row["peak_allocated_gib_max"]==max(peaks)
    import train
    from types import SimpleNamespace
    from contextlib import redirect_stdout
    from io import StringIO
    with redirect_stdout(StringIO()):
        example=train.inspect_record(SimpleNamespace(records=ROOT/"evidence/experiments.json",
            model="v3",split="validation",case="K008_Week0",output=None))
    assert example==json.loads((ROOT/"evidence/example_K008_v3_record.json").read_text())
    print("PASS: cohort timing/memory quantiles and MRI-free worked example recomputed")
    for filename,digest in provenance["clinical_figures"].items():
        assert sha256_file(ROOT/"figures"/filename)==digest
    replay_path=ARCHIVE/"migration_receipt.json"
    assert sha256_file(replay_path)=="14936ed27162acd63209f8ab34dec9fc05192f30746b80111e520c94abaa38fc"
    replay=json.loads(replay_path.read_text())
    verify_replayed_sources(ROOT, replay)
    for name,record in replay["models"].items():
        assert record["cases"]==38 and record["all_confusions_and_inputs_match"] is True
        assert record["checkpoint_sha256"]==bundle["models"][name]["selected"]["checkpoint_sha256"]
    print("PASS: GPU replay receipt and runtime equivalence (including prose-only edits); 76 Validation matrices")
    print("PASS: 500 epoch entries; Validation-selected checkpoints; 680 case counts; 400 foreground metrics; five original clinical figures")


if __name__=="__main__":main()
