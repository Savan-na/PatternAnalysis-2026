"""Check that the standalone models reproduce the original Validation predictions.

Compare all 76 confusion matrices with the earlier evaluation and check
input and weight hashes. This script does not train, tune, read Test images
or overwrite the recorded results.
"""
import argparse,json
from pathlib import Path
import os
import sys
ROOT = (Path(os.environ["HIPMRI_SUBMISSION_ROOT"]) if "HIPMRI_SUBMISSION_ROOT" in os.environ
        else Path(__file__).resolve().parents[3] / "recognition/HipMRI_UNet_s4906923")
sys.path.insert(0, str(ROOT))
import torch
from dataset import discover
from utils import load_model,evaluate,sha256_file,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--standard-checkpoint",type=Path,required=True)
    p.add_argument("--v3-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    a=p.parse_args()
    if a.output.exists():p.error("Use a new receipt path")
    bundle=json.loads((ROOT/"evidence/experiments.json").read_text())
    cases=discover(a.dataset_root)["validation"]
    receipt={"kind":"standalone_migration_validation_replay","torch":str(torch.__version__),
             "sources":{n:sha256_file(ROOT/n) for n in ("modules.py","dataset.py","utils.py")},
             "driver_sha256":sha256_file(Path(__file__)),"models":{}}
    for name,arm,path in [("standard","Standard",a.standard_checkpoint),("v3","v3",a.v3_checkpoint)]:
        historical=bundle["models"][arm]
        assert sha256_file(path)==historical["selected"]["checkpoint_sha256"],"Weight identity changed"
        model=load_model(name,path,a.device)
        result=evaluate(model,cases,a.device,detailed=True)
        prior={r["case"]:r for r in historical["splits"]["validation"]}
        for row in result["cases"]:
            old=prior[row["case"]]
            assert row["confusion"]==old["confusion"],f"Prediction mismatch: {arm}/{row['case']}"
            for k in ("mri_sha256","mask_sha256"):assert row[k]==old[k],"Input file changed"
            for label in old["surfaces"]:
                for key in ("hd95_mm","assd_mm"):
                    left,right=row["surfaces"][label][key],old["surfaces"][label][key]
                    assert left==right or (left is not None and right is not None and abs(left-right)<1e-6),"Geometry metric changed"
        receipt["models"][arm]={"cases":len(result["cases"]),"all_confusions_and_inputs_match":True,
                                 "selection_score":result["selection_score"],"checkpoint_sha256":sha256_file(path)}
        del model
        if a.device.startswith("cuda"):torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output,receipt)
    print("PASS: Standard/v3 all 76 Validation confusion matrices reproduced;",a.output)


if __name__=="__main__":main()
