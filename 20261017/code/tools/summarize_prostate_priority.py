"""Four core figures: foreground-only metrics, explicit prostate priority, fixed checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/reports/model_performance_comparison'
MODELS=['2D','Standard','v1','v2','v3','A0','A1','v5-C','v5-L','v5-S']
LABELS=['Body','Bones','Bladder','Rectum','Prostate']
WEIGHTS=np.array([.125,.125,.125,.125,.5])


def write_table(name, rows):
    with (OUT/'tables'/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save(name, fig):
    fig.tight_layout()
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=190,bbox_inches='tight')
    plt.close(fig)


def render(data):
    metrics=data['metrics']; boundaries=data['surfaces']; chosen=data['selected']
    assert abs(WEIGHTS.sum()-1)<1e-12 and WEIGHTS[-1]>max(WEIGHTS[:-1])
    get=lambda m,s,c:next(r for r in metrics if r['model']==m and r['split']==s and r['label']==c)
    surface=lambda m,s:next(r for r in boundaries if r['model']==m and r['split']==s and r['label']=='Prostate')
    scores=[]
    for model in MODELS:
        for split in ('validation','test'):
            rows=[get(model,split,c) for c in LABELS]
            values={k:np.array([r[k] for r in rows]) for k in ('dice','iou','precision','recall')}
            score=dict(model=model,split=split,best_epoch=chosen[model],
                       **{f'foreground_{k}':float(v.mean()) for k,v in values.items()},
                       prostate_priority_dice=float(values['dice']@WEIGHTS),
                       **{f'prostate_{k}':float(v[-1]) for k,v in values.items()},
                       prostate_hd95_median_mm=surface(model,split)['hd95_median_mm'],
                       prostate_hd95_max_mm=surface(model,split)['hd95_max_mm'])
            assert abs(score['foreground_dice']-next(r['foreground_macro_dice'] for r in data['summary'] if r['model']==model and r['split']==split))<1e-12
            scores.append(score)
    write_table('model_summary',scores)
    write_table('foreground_metrics',[r for r in metrics if r['label']!='Background'])
    write_table('surface_metrics',boundaries)
    write_table('training_curves',data['curves'])
    write_table('patient_scores',data['patients'])
    write_table('available_resource_benchmarks',data['resources'])
    score=lambda m,s:next(r for r in scores if r['model']==m and r['split']==s)
    labels=[f'{m} (e{chosen[m]})' for m in MODELS]
    y=np.arange(10)
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    for ax,key,title in zip(axes,['foreground_dice','prostate_priority_dice'],['Five foreground classes: equal-weight Dice','Prostate-priority Dice: 50% + 4 x 12.5%']):
        vals=[score(m,'test')[key] for m in MODELS]
        ax.barh(y,vals,color=['#2981a3' if m not in ('v3','v5-C') else '#288451' for m in MODELS])
        for i,v in enumerate(vals):ax.text(v+.008,i,f'{v:.4f}',va='center',fontsize=10)
        ax.set(xlim=(0,1),title=title,xlabel='Existing Test; higher is better')
        ax.set_yticks(y,labels)
        if ax is axes[0]: ax.invert_yaxis()
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('All models: exclude Background; checkpoints remain fixed by original Validation selection',fontsize=12)
    save('01_foreground_and_prostate_priority',fig)
    fig,ax=plt.subplots(figsize=(11,7))
    for split,offset,color in [('validation',-.18,'#2878aa'),('test',.18,'#d58a32')]:
        vals=[score(m,split)['prostate_dice'] for m in MODELS]
        ax.barh(y+offset,vals,height=.34,label=split,color=color)
        for i,v in enumerate(vals):ax.text(v+.006,i+offset,f'{v:.4f}',va='center',fontsize=8)
    ax.axvline(.7,color='#a52a2a',linestyle='--',linewidth=1,label='Draft Hard threshold 0.7; aggregation needs confirmation')
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set(xlim=(0,1.08),xlabel='Patient-equal Prostate Dice',title='Prostate first: Validation and previously inspected Test')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=3,fontsize=8)
    save('02_prostate_validation_test',fig)
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    for ax,key in zip(axes,['dice','iou']):
        matrix=np.array([[get(m,'test',c)[key] for c in LABELS] for m in MODELS])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(5),LABELS,rotation=20); ax.set_yticks(y,labels)
        ax.set_title('Test '+key.upper()+' — Background excluded')
        for i in range(10):
            for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.68 else 'black',fontsize=9)
    save('03_foreground_dice_iou',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    matrices=[np.array([[score(m,'test')[k] for k in ['prostate_precision','prostate_recall']] for m in MODELS]),
              np.array([[score(m,'test')[k] for k in ['prostate_hd95_median_mm','prostate_hd95_max_mm']] for m in MODELS])]
    for ax,matrix,cols,title,dist in zip(axes,matrices,[['Precision','Recall'],['HD95 median','HD95 maximum']],['Prostate overlap trade-off','Prostate boundary risk (mm; lower is better)'],[False,True]):
        im=ax.imshow(matrix,aspect='auto',cmap='YlOrRd' if dist else 'YlGnBu',**(dict(norm=LogNorm(vmin=1,vmax=matrix.max())) if dist else dict(vmin=0,vmax=1)))
        ax.set_xticks(range(2),cols);ax.set_yticks(y,labels);ax.set_title(title,fontsize=11)
        for i in range(10):
            for j in range(2):ax.text(j,i,f'{matrix[i,j]:.2f}' if dist else f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if im.norm(matrix[i,j])>.68 else 'black')
    fig.suptitle('Existing Test: good average Dice does not establish reliable boundaries\nHD95 statistics across 30 volumes; distance colors use a log scale',fontsize=12)
    save('04_prostate_precision_recall_boundaries',fig)
    (OUT/'source_checksums.json').write_text(json.dumps(dict(input_sha256=data['sources'],selected_epochs=chosen,
        score_definition=dict(class_order=LABELS,weights=WEIGHTS.tolist(),scope='Post-hoc descriptive analysis, not official criterion or training loss'),
        reader_sha256=hashlib.sha256((ROOT/'code/tools/build_model_comparison_gallery.py').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('Recomputed foreground metrics and 50%-Prostate scores; generated four PNG figures')
