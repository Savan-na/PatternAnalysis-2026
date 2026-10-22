"""Summarise existing Test spatial diagnostics for the final research questions.

No inference, training, checkpoint selection or new Test access is performed.
Inputs are the already evaluated and archived prediction measurements.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]/"recognition/HipMRI_UNet_s4906923"


def main():
    path=ROOT/"evidence/experiments.json"
    bundle=json.loads(path.read_text())
    result={"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope":"Previously evaluated Test predictions; descriptive analysis only",
            "aggregation":"Mean across visits within each patient, then equal mean across six patients; OAR counts pooled within patient before calculating the rate",
            "definitions":{
                "apex_central_base_slice_dice":"Mean GT-positive slice Dice in physically ordered inferior 20%, central region, superior 20%; geometric proxies, not clinician-annotated endpoints",
                "visits_with_internal_prostate_gap_fraction":"Fraction of visits with an empty predicted Z slice between nonempty predicted prostate slices; patient-equal mean",
                "area_transition_absolute_error":"Absolute difference between mean normalized adjacent-area changes of prediction and GT, on GT-positive adjacent pairs; mean within patient then across patients; dimensionless proxy",
                "true_oar_labelled_prostate_fraction":"GT Bladder/Rectum voxels predicted as Prostate divided by all GT Bladder/Rectum voxels, pooled within patient then averaged; not a wall-specific or dose metric"},
            "models":{}}
    for name in ("2D","Standard","v3"):
        rows=bundle["models"][name]["splits"]["test"]
        groups=defaultdict(list)
        for row in rows:groups[row["patient"]].append(row)
        assert len(rows)==30 and len(groups)==6
        values={}
        for region in ("apex","central","base"):
            assert all(row["prostate_regions"].get(region) is not None for row in rows)
            values[region+"_slice_dice"]=float(np.mean([np.mean([row["prostate_regions"][region] for row in visits]) for visits in groups.values()]))
        for kind in ("prediction","truth"):
            values[kind+"_internal_gap_visit_fraction"]=float(np.mean([np.mean([row["geometry"]["Prostate"][kind]["internal_empty_z_slices"]>0 for row in visits]) for visits in groups.values()]))
        values["area_transition_absolute_error"]=float(np.mean([np.mean([abs(row["geometry"]["Prostate"]["area_transition"]["prediction"]-row["geometry"]["Prostate"]["area_transition"]["truth"]) for row in visits]) for visits in groups.values()]))
        rates=[]
        for visits in groups.values():
            matrix=sum(np.asarray(row["confusion"]) for row in visits)
            denominator=matrix[[3,4],:].sum()
            assert denominator>0
            rates.append((matrix[3,5]+matrix[4,5])/denominator)
        values["true_oar_labelled_prostate_fraction"]=float(np.mean(rates))
        result["models"][name]=values
    out=ROOT/"evidence/spatial_context_summary.json"
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result["models"],indent=2))


if __name__=="__main__":main()
