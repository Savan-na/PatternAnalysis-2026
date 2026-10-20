"""Best recorded Validation checkpoints; assessment vs supplemental evidence."""
from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from tools.summarize_v2_results import metrics, patient_matrices, score

OUT=ROOT/'results/model_evaluation/best_checkpoint_comparison'
NAMES=['Background','Body','Bones','Bladder','Rectum','Prostate']
DIRS={'2D':ROOT/'logs/training/baseline_2d','Standard':ROOT/'logs/training/baseline_3d',
      'v1':ROOT/'logs/training/improved_3d',
      'v2':ROOT/'results/experiments/improved_3d_v2/training_records',
      'v3':ROOT/'results/experiments/improved_3d_v3/training_records'}
SOURCES={}


def read(path):
    SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def table(group,name,rows):
    with (OUT/group/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def save(group,name,fig):
    for ext in ('png','svg'):fig.savefig(OUT/group/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    required='assessment_required';extra='supplementary_research'
    for group in (required,extra):
        for d in ('figures','tables'):(OUT/group/d).mkdir(parents=True,exist_ok=True)
    runs={n:[read(p/f'epoch_{e:03d}.json') for e in range(1,51)] for n,p in DIRS.items()}
    best={};selection={}
    for n,records in runs.items():
        available=[r for r in records if r['validation'] is not None]
        assert [r['epoch'] for r in available]==list(range(5,51,5))
        for r in available:
            assert len(r['validation']['cases'])==38 and len(patient_matrices(r))==6
            assert abs(score(r)-r['validation']['selection_score'])<1e-12
        b=max(available,key=lambda r:(score(r),-r['epoch']))
        assert records[-1]['best_epoch']==b['epoch']
        best[n]=b
        lrs=sorted({r['train']['learning_rate'] for r in records})
        selection[n]={'epoch':b['epoch'],'validation_score':score(b),'candidate_epochs':[r['epoch'] for r in available],
                      'record_sha256':SOURCES[str((DIRS[n]/f"epoch_{b['epoch']:03d}.json").relative_to(ROOT))],
                      'learning_rates_observed':lrs,'policy':'highest patient-equal small-organ Validation Dice; earliest tie; no Test selection'}
    (OUT/required/'tables/selected_checkpoints.json').write_text(json.dumps(selection,indent=2)+'\n')
    classes=[];patients=[];body=[];confusions=[];bias=[]
    old=read(ROOT/'results/tables/comparisons/g5_validation_body_audit.json')
    oldbody={r['case']:r for r in old['cases']}
    for n,r in best.items():
        grouped=patient_matrices(r);pm=[metrics(m) for m in grouped.values()];pooled=metrics(r['validation']['pooled_confusion'])
        for j,label in enumerate(NAMES):
            classes.append(dict(model=n,selected_epoch=r['epoch'],label=label,
                **{'patient_equal_'+k:float(np.mean([a[k][j] for a in pm])) for k in pooled},
                **{'pooled_'+k:float(v[j]) for k,v in pooled.items()}))
        for p,m in sorted(grouped.items()):
            v=metrics(m)['dice'];patients.append(dict(model=n,patient=p,small_organ_dice=float(v[3:].mean()),**dict(zip(NAMES,v.tolist()))))
            for j,label in enumerate(NAMES):
                bias.append(dict(model=n,patient=p,label=label,relative_volume_bias=float((m[:,j].sum()-m[j].sum())/m[j].sum())))
        matrix=np.array(r['validation']['pooled_confusion']);normalized=matrix/matrix.sum(1,keepdims=True)
        confusions.append(normalized)
        for c in r['validation']['cases']:
            hd=None;far=None
            if n in ('Standard','v1'):
                b=oldbody[c['case']]['models']['standard_3d' if n=='Standard' else 'improved_3d']
                hd=b['boundary']['hd95_mm'];far=b['background_to_body_at_least_20mm']
            elif n in ('v2','v3'):
                hd=c['surfaces']['Body']['hd95_mm'];far=c['far_background_to_body_voxels']
            body.append(dict(model=n,case=c['case'],patient=c['patient'],body_hd95_mm=hd,far_fp=far,
                             availability='measured' if hd is not None else 'pending matched audit'))
    table(required,'six_class_best_metrics.csv',classes)
    table(required,'body_boundary_available.csv',body)
    table(extra,'per_patient_best.csv',patients);table(extra,'relative_volume_bias.csv',bias)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,k in zip(axes,('dice','iou')):
        for i,n in enumerate(best):
            ax.bar(np.arange(6)+(i-2)*.155,[r['patient_equal_'+k] for r in classes if r['model']==n],.155,label=f'{n} e{best[n]["epoch"]}')
        ax.set_xticks(range(6),NAMES,rotation=20);ax.set(ylim=(0,1.05),title='Patient-equal '+k)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=5,fontsize=9)
    save(required,'six_class_dice_iou',fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,k in zip(axes,('precision','recall')):
        for i,n in enumerate(best):ax.bar(np.arange(6)+(i-2)*.155,[r['patient_equal_'+k] for r in classes if r['model']==n],.155,label=f'{n} e{best[n]["epoch"]}')
        ax.set_xticks(range(6),NAMES,rotation=20);ax.set(ylim=(0,1.05),title='Patient-equal '+k)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=5,fontsize=9)
    save(extra,'precision_recall',fig)
    fig,axes=plt.subplots(1,4,figsize=(21,4.5),layout='constrained')
    for n,rs in runs.items():
        vv=[r for r in rs if r['validation']]
        axes[0].plot([r['epoch'] for r in vv],[score(r) for r in vv],'-o',label=n)
        axes[2].plot(range(1,51),[r['train']['learning_rate'] for r in rs],label=n)
        if n in ('Standard','v1','v3'):axes[1].plot(range(1,51),[r['train']['mean_loss'] for r in rs],label=n)
    axes[0].set(title='All recorded Validation checkpoints',xlabel='Epoch',ylabel='Small-organ Dice',ylim=(0,1));axes[0].legend(fontsize=8)
    axes[1].set(title='Shared-objective 3D training loss',xlabel='Epoch',ylabel='Loss; sampled visits vary');axes[1].legend(fontsize=8)
    axes[2].set(title='Logged learning rate: all runs fixed',xlabel='Epoch',ylabel='Adam base learning rate',ylim=(0,.0015));axes[2].legend(fontsize=8)
    for n in ('v2','v3'):
        vv=[r for r in runs[n] if r['validation']]
        axes[3].plot([r['epoch'] for r in vv],
                     [np.median([c['surfaces']['Body']['hd95_mm'] for c in r['validation']['cases']]) for r in vv],'-o',label=n)
    axes[3].set(title='Body boundary trajectory',xlabel='Epoch',ylabel='Volume-median HD95 (mm)');axes[3].legend(fontsize=8)
    save(required,'learning_curves_and_lr',fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    ids=sorted({r['case'] for r in body})
    for n in best:
        rows={r['case']:r for r in body if r['model']==n}
        if any(r['body_hd95_mm'] is None for r in rows.values()):continue
        axes[0].plot(range(38),[rows[c]['body_hd95_mm'] for c in ids],'.-',label=n)
        pp=sorted(grouped)
        axes[1].plot(pp,[np.median([r['body_hd95_mm'] for r in rows.values() if r['patient']==p]) for p in pp],'-o',label=n)
    axes[0].set(title='Body HD95: each volume (2D pending)',xlabel='Sorted case index',ylabel='mm')
    axes[1].set(title='Body HD95: within-patient medians',ylabel='mm');axes[0].legend(fontsize=8)
    save(required,'body_boundary_by_case',fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    pp=sorted(grouped)
    for n in best:ax.plot(pp,[next(r['small_organ_dice'] for r in patients if r['model']==n and r['patient']==p) for p in pp],'-o',label=n)
    ax.set(ylim=(0,1),ylabel='Patient small-organ Dice',title='Best-checkpoint patient heterogeneity');ax.legend();save(extra,'patient_small_organ_dice',fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for ax,n,m in zip(axes.flat,best,confusions):
        ax.imshow(m,vmin=0,vmax=1,cmap='Blues')
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{100*m[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if m[i,j]>.5 else 'black')
        ax.set_xticks(range(6),NAMES,rotation=50,ha='right',fontsize=8);ax.set_yticks(range(6),NAMES,fontsize=8)
        ax.set(title=n+' | GT-row normalized (%)',xlabel='Predicted',ylabel='Ground truth')
    axes.flat[-1].axis('off');save(extra,'normalized_confusion',fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for i,n in enumerate(best):ax.bar(np.arange(6)+(i-2)*.155,[np.mean([r['relative_volume_bias'] for r in bias if r['model']==n and r['label']==label]) for label in NAMES],.155,label=n)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(6),NAMES,rotation=20);ax.set(ylabel='Mean patient (predicted-GT)/GT volume',title='Signed volume bias; location errors can cancel');ax.legend(fontsize=8)
    save(extra,'volume_bias',fig)
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    for n in best:
        rr={r['case']:r for r in body if r['model']==n}
        if any(r['far_fp'] is None for r in rr.values()):continue
        ax.plot(range(38),[rr[c]['far_fp'] for c in ids],'.-',label=n)
    ax.set_yscale('symlog',linthresh=10);ax.set(title='Distant Body FP (2D pending)',xlabel='Sorted case index',ylabel='Voxels');ax.legend(fontsize=8)
    save(extra,'distant_body_fp',fig)
    # Resource columns are measured but are not a matched inference benchmark.
    from hipmri_rewrite.models import UNet2D,UNet3D,ImprovedUNet3D
    from hipmri_rewrite.improved_v2 import ImprovedUNet3DV2
    from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3
    ctors=dict(zip(DIRS,[UNet2D,UNet3D,ImprovedUNet3D,ImprovedUNet3DV2,ImprovedUNet3DV3]))
    resources=[]
    for n,rs in runs.items():
        resources.append(dict(model=n,parameters=sum(p.numel() for p in ctors[n]().parameters()),
            measured_train_peak_allocated_GiB=max(r['train']['peak_allocated_bytes'] for r in rs)/2**30,
            summed_epoch_train_and_checkpoint_seconds=sum(r['train']['epoch_seconds'] for r in rs),
            matched_inference_peak_vram='pending fixed-checkpoint audit',matched_inference_latency='pending fixed-checkpoint audit',
            warning='Training/IO measurements are not pure inference performance; 2D recipe differs'))
    table(required,'resources_measured_and_pending.csv',resources)
    (OUT/'source_checksums.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
