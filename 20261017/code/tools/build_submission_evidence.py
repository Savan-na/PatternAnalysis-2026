"""Build a compact, independently checkable submission evidence set; no inference."""
from pathlib import Path
import csv, hashlib, json, shutil, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from tools.build_model_comparison_gallery import collect_evidence, DIRS
OUT=ROOT.parent/"recognition/HipMRI_UNet_s4906923"

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checked=collect_evidence()
    evidence=OUT/"evidence";evidence.mkdir(exist_ok=True)
    figures=OUT/"figures";figures.mkdir(exist_ok=True)
    raw=json.loads((ROOT/"results/experiments/improved_3d_v4/comprehensive_audit_20261021/metrics.json").read_text())
    test=json.loads((ROOT/"results/experiments/improved_3d_v5/comparisons/test_assessment_20261021/metrics.json").read_text())
    records={}
    for model,directory in DIRS.items():
        best=checked["selected"][model]
        record=json.loads((directory/f"epoch_{best:03d}.json").read_text())
        if model in raw["arms"]:
            arm=raw["arms"][model];splits=arm["splits"];benchmark=arm["benchmark"]
            selected=arm["selected"]
        else:
            arm=test["arms"][model[-1]]
            splits={"validation":record["validation"]["cases"],"test":arm["cases"]}
            benchmark=arm["benchmark"];selected=arm["selected"]
        records[model]={"best_epoch":best,"selected":selected,"benchmark":benchmark,
                        "splits":{sp:[dict(c) for c in cases]
                                  for sp,cases in splits.items()}}
    bundle={"kind":"ten_locked_models_descriptive_evidence", "date":"2026-10-21",
            "class_order":["Background","Body","Bones","Bladder","Rectum","Prostate"],
            "limitation":"Previously inspected Test, six patients; no fresh blind evaluation or clinical safety claim",
            "models":records,"training_candidates":checked["curves"],"original_sources":checked["sources"]}
    (evidence/"experiments.json").write_text(json.dumps(bundle,indent=2)+"\n",encoding="utf-8")
    table_source=ROOT/"results/reports/model_performance_comparison/tables"
    for name in ("model_summary.csv","foreground_metrics.csv","surface_metrics.csv",
                 "training_curves.csv","patient_scores.csv","available_resource_benchmarks.csv"):
        shutil.copyfile(table_source/name,evidence/name)
    for relative,name in [
        ("test/M015_Week7_Bladder.png","failure_bladder.png"),
        ("test/W029_Week6_Prostate_base_apex.png","failure_prostate_ends.png"),
        ("validation/K018_Week4_Body_three_planes.png","failure_body.png"),
        ("validation/J026_Week4_Rectum.png","failure_rectum.png"),
        ("validation/B040_Week1_Bones.png","failure_bones.png")]:
        source=ROOT/"results/experiments/improved_3d_v4/comprehensive_audit_20261021/images"/relative
        shutil.copyfile(source,figures/name)
    provenance={"research_root":"20261017", "sources":checked["sources"],
                "clinical_figures":{p.name:sha(p) for p in figures.glob("failure_*.png")},
                "preserved_original_panels":True,
                "panel_note":"Seven historical model columns retained and labelled; no new inference or retouched contours"}
    (evidence/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    # CSV tables are derived from integer confusion counts, not screenshots.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,color in [("Standard","#356ba0"),("v3","#d47a27")]:
        rows=[r for r in checked["curves"] if r["model"]==name]
        axes[0].plot([r["epoch"] for r in rows],[r["train_loss"] for r in rows],label=name,color=color)
        rows=[r for r in rows if r["validation_dice"] is not None]
        axes[1].plot([r["epoch"] for r in rows],[r["validation_dice"] for r in rows],"o-",label=name,color=color)
    axes[0].set(xlabel="Epoch",ylabel="Training objective (not segmentation quality)")
    axes[1].set(xlabel="Epoch",ylabel="Patient-equal small-organ Validation Dice",ylim=(.3,.9))
    for a in axes:a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/"training_and_selection.png",dpi=180);plt.close(fig)
    labels=bundle["class_order"][1:]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axes,["validation","test"]):
        x=np.arange(5)
        for j,(name,color) in enumerate([("Standard","#356ba0"),("v3","#d47a27")]):
            vals=[next(r["dice"] for r in checked["metrics"] if r["model"]==name and r["split"]==split and r["label"]==label) for label in labels]
            ax.bar(x+(j-.5)*.36,vals,width=.36,label=name,color=color)
        ax.set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel="Patient-equal Dice",title=split.title())
        ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(figures/"foreground_comparison.png",dpi=180);plt.close(fig)
    # Explicit architecture diagram: common physical data pipeline + model differences.
    fig,ax=plt.subplots(figsize=(12,3.6));ax.axis("off")
    boxes=[(.02,.48,.23,.37,"Paired MRI / labels\nPatient-disjoint 26 / 6 / 6\n2x XYZ downsample + Z-score"),
           (.33,.61,.29,.28,"Standard 3D U-Net\nGN1, width 8, single output"),
           (.33,.13,.29,.33,"Improved v3 residual U-Net\nGN1, width 8, dropout 0.3\nmain + half + quarter logits"),
           (.72,.37,.26,.38,"6-class full-volume prediction\nRemove high-end padding\nPatient-equal + boundary audit")]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.01",facecolor="#edf2f7",edgecolor="#456"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    for start,end in [((.25,.67),(.33,.75)),((.25,.55),(.33,.30)),((.62,.75),(.72,.62)),((.62,.30),(.72,.48))]:
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#456",lw=1.5))
    fig.tight_layout();fig.savefig(figures/"method.png",dpi=180);plt.close(fig)
    print("Verified 500 epoch records, 680 best-checkpoint case results; wrote compact evidence and eight figures")

if __name__=="__main__":main()
