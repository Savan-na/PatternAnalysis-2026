"""Populate required/optional evidence only after a verified remote audit exists."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from tools.summarize_v2_results import metrics


def main():
    source=ROOT/'results/experiments/improved_3d_v3/best_checkpoint_audit_20261020'
    if not (source/'metrics.json').exists():
        raise SystemExit('Remote best-checkpoint audit is not downloaded; do not fabricate missing metrics')
    checks=json.loads((source/'checksums.json').read_text())
    for name,h in checks.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
    result=json.loads((source/'metrics.json').read_text());assert result['complete'] is True
    out=ROOT/'results/model_evaluation/best_checkpoint_comparison'
    required='assessment_required';extra='supplementary_research'
    selected=json.loads((out/required/'tables/selected_checkpoints.json').read_text())
    names=['Background','Body','Bones','Bladder','Rectum','Prostate']
    def table(group,name,rows):
        with (out/group/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def save(group,name,fig):
        for ext in ('png','svg'):fig.savefig(out/group/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    class_rows=[];surface_rows=[];region_rows=[];continuity=[];patient_rows=[];resources=[]
    for arm,data in result['arms'].items():
        assert data['selected']['record_sha256']==selected[arm]['record_sha256']
        assert data['selected']['epoch']==selected[arm]['epoch']
        resources.append(dict(model=arm,gpu=result['gpu'],**data['benchmark']))
        for split,rows in data['splits'].items():
            assert len(rows)==(38 if split=='validation' else 30)
            patients={}
            for r in rows:
                patients.setdefault(r['patient'],np.zeros((6,6),dtype=np.int64));patients[r['patient']]+=np.array(r['confusion'])
                for label,s in r['surfaces'].items():surface_rows.append(dict(model=arm,split=split,case=r['case'],patient=r['patient'],label=label,**s))
                if r['prostate_regions']:
                    for region,value in r['prostate_regions'].items():region_rows.append(dict(model=arm,split=split,case=r['case'],patient=r['patient'],region=region,slice_mean_dice=value,status=r['prostate_region_status']))
                else:region_rows.append(dict(model=arm,split=split,case=r['case'],patient=r['patient'],region='unavailable',slice_mean_dice=None,status=r['prostate_region_status']))
                for label,g in r['geometry'].items():
                    continuity.append(dict(model=arm,split=split,case=r['case'],patient=r['patient'],label=label,
                        truth_components=g['truth']['components_6'],prediction_components=g['prediction']['components_6'],
                        prediction_largest_fraction=g['prediction']['largest_component_fraction'],
                        truth_gap_slices=g['truth']['internal_empty_z_slices'],prediction_gap_slices=g['prediction']['internal_empty_z_slices']))
            assert len(patients)==6
            pooled=metrics(sum(patients.values()));pm=[metrics(m) for m in patients.values()]
            for j,label in enumerate(names):class_rows.append(dict(model=arm,split=split,label=label,
                **{'patient_equal_'+k:float(np.nanmean([m[k][j] for m in pm])) for k in pooled},
                **{'pooled_'+k:float(v[j]) for k,v in pooled.items()}))
            for patient,m in patients.items():patient_rows.append(dict(model=arm,split=split,patient=patient,small_organ_dice=float(metrics(m)['dice'][3:].mean())))
    table(required,'audited_all_splits_six_class_metrics.csv',class_rows)
    table(required,'audited_resource_profile.csv',resources)
    table(required,'audited_foreground_surfaces_with_empty_status.csv',surface_rows)
    table(required,'audited_prostate_base_apex.csv',region_rows)
    table(extra,'audited_continuity.csv',continuity)
    table(extra,'audited_patient_scores.csv',patient_rows)
    for split in ('validation','test'):
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for ax,metric in zip(axes,('dice','iou')):
            for i,arm in enumerate(result['arms']):
                values=[next(r['patient_equal_'+metric] for r in class_rows if r['model']==arm and r['split']==split and r['label']==label) for label in names]
                ax.bar(np.arange(6)+(i-2)*.155,values,.155,label=f'{arm} e{selected[arm]["epoch"]}')
            ax.set_xticks(range(6),names,rotation=20);ax.set(ylim=(0,1.05),title=split+' patient-equal '+metric)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=5)
        if split=='test':fig.suptitle('Previously inspected Test: descriptive, not new independent confirmation')
        save(required,split+'_best_six_class_audited',fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,split in zip(axes,('validation','test')):
        for arm in result['arms']:
            values=[]
            for region in ('apex','central','base'):
                rows=[r for r in region_rows if r['model']==arm and r['split']==split and r['region']==region]
                ps=sorted({r['patient'] for r in rows})
                values.append(float(np.mean([np.mean([r['slice_mean_dice'] for r in rows if r['patient']==p]) for p in ps])) if ps else np.nan)
            ax.plot(['apex','central','base'],values,'-o',label=arm)
        ax.set(title=split+' prostate slice-region Dice',ylim=(0,1),ylabel='Equal patients; mean visits then slices')
    axes[0].legend(fontsize=8);save(required,'prostate_regions_audited',fig)
    write={'source_metrics_sha256':hashlib.sha256((source/'metrics.json').read_bytes()).hexdigest(),
        'remote_files_checksum_verified':len(checks),'models':list(result['arms']),
        'selection':'recorded Validation best only','test_status':'already inspected; descriptive only',
        'surface_failures':sum(r['status']=='one_empty' for r in surface_rows),
        'region_unavailable':sum(r['region']=='unavailable' for r in region_rows)}
    (out/'audited_coverage.json').write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps(write,indent=2))


if __name__=='__main__':main()
