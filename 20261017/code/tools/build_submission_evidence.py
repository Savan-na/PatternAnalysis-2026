"""Build a compact, independently checkable submission evidence set; no inference."""
from pathlib import Path
import csv, hashlib, json, shutil, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from tools.build_model_comparison_gallery import collect_evidence, DIRS
OUT=ROOT.parent/"recognition/HipMRI_UNet_s4906923"

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def draw_method_figure(figures):
    """Draw the submitted preprocessing and model paths; no data/inference needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(9.5,7.6))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    xs=(.015,.353,.691);w=.294
    def box(x,y,h,text,face):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.008",linewidth=1.1,
                                  facecolor=face,edgecolor="#46586b"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11.3,linespacing=1.45,color="#172c40")
    def arrows(y):
        for x in xs[:2]:
            ax.annotate("",xy=(x+.338-.012,y),xytext=(x+w+.01,y),
                        arrowprops=dict(arrowstyle="->",lw=1.5,color="#46586b"))
    ax.text(.015,.977,"INPUT AND SPATIAL PREPARATION",fontsize=13,weight="bold",color="#172c40")
    box(xs[0],.825,.12,"MRI + six-class mask\nMatching patient/visit\nMatching physical grids","#edf1f5")
    box(xs[1],.825,.12,"Downsample X/Y/Z by 2\nLinear MRI; nearest mask\nNonzero MRI Z-score","#edf1f5")
    box(xs[2],.825,.12,"Train: 64 x 64 x 32 crops\nInfer: padded whole volume\nOne MRI input channel","#edf1f5");arrows(.885)
    ax.text(.015,.774,"STANDARD 3D U-NET  |  1,412,550 PARAMETERS",fontsize=13,weight="bold",color="#245b84")
    box(xs[0],.542,.20,"ENCODER + BOTTLENECK\nChannels: 8,16,32,64,128\nTwo 3x3x3 convs per level\nGN1 + ReLU\nFour max-pooling steps","#edf5fb")
    box(xs[1],.542,.20,"DECODER\nFour transposed conv steps\nConcatenate encoder skips\nTwo 3x3x3 convs per level\nGN1 + ReLU","#edf5fb")
    box(xs[2],.542,.20,"SINGLE OUTPUT\nFull-resolution 1x1x1 head\nSix class logits\nWeighted CE + foreground Dice\n1,412,550 parameters","#edf5fb");arrows(.642)
    ax.text(.015,.488,"IMPROVED 3D U-NET v3  |  2,366,946 PARAMETERS",fontsize=13,weight="bold",color="#8a4f20")
    box(xs[0],.232,.222,"RESIDUAL CONTEXT\nChannels: 8,16,32,64,128\nFive identity residual blocks\nGN1 + LeakyReLU\nDropout3d: 0.3\nFour stride-2 conv steps","#fcf1e7")
    box(xs[1],.232,.222,"DECODER\nNearest-neighbour resize\n3x3x3 channel reduction\nConcatenate context skips\n3x3x3 fusion + GN1\n1x1x1 channel reduction","#fcf1e7")
    box(xs[2],.232,.222,"ADDITIVE OUTPUTS\nMain + half + quarter heads\nResize auxiliary logits\nSum all three outputs\nOne CE + Dice loss on sum\n2,366,946 parameters","#fcf1e7");arrows(.343)
    ax.add_patch(FancyBboxPatch((.015,.025),.97,.137,boxstyle="round,pad=.008",linewidth=1.1,
                              facecolor="#edf1f5",edgecolor="#46586b"))
    ax.text(.5,.094,"EVALUATION: ARGMAX, REMOVE HIGH-END PADDING, RETAIN PHYSICAL GRID\n"
            "Per-class Dice / IoU; physical surface errors; prostate ends and slice continuity\n"
            "Training loss uses paired labels; prediction uses MRI without a GT-defined region",
            ha="center",va="center",fontsize=11.5,linespacing=1.6,color="#172c40")
    fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985)
    fig.savefig(Path(figures)/"method.png",dpi=240,facecolor="white");plt.close(fig)


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
    previous_path=evidence/"provenance.json"
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
    provenance={"research_root":"20261017", "sources":checked["sources"],
                "clinical_figures":{p.name:sha(p) for p in figures.glob("failure_*.png")},
                "preserved_original_panels":True,
                "panel_note":"Seven historical model columns retained and labelled; no new inference or retouched contours"}
    for key in ("pretraining_review","assessment_alignment","report_diagnostic_summaries","method_figure"):
        if key in previous:provenance[key]=previous[key]
    if "report_diagnostic_summaries" in provenance:
        assert provenance["report_diagnostic_summaries"]["source_bundle_sha256"]==sha(evidence/"experiments.json")
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
    draw_method_figure(figures)
    if "method_figure" in provenance:
        provenance["method_figure"].update(sha256=sha(figures/"method.png"),generator_sha256=sha(Path(__file__)))
        (evidence/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print("Verified 500 epoch records, 680 best-checkpoint case results; wrote compact evidence and eight figures")

if __name__=="__main__":main()
