"""Compare seven locked Validation-best records without running inference."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/experiments/improved_3d_v4/paired_analysis/all_models'
DIRS = {
    '2D': ROOT/'logs/training/baseline_2d',
    'Standard': ROOT/'logs/training/baseline_3d',
    'v1': ROOT/'logs/training/improved_3d',
    'v2': ROOT/'results/experiments/improved_3d_v2/training_records',
    'v3': ROOT/'results/experiments/improved_3d_v3/training_records',
    'A0': ROOT/'results/experiments/improved_3d_v4/a0/training_records',
    'A1': ROOT/'results/experiments/improved_3d_v4/a1/training_records',
}
NAMES = ['Background','Body','Bones','Bladder','Rectum','Prostate']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, selection, inputs = [], [], {}
    for name, directory in DIRS.items():
        records=[]
        for e in range(1,51):
            p=directory/f'epoch_{e:03}.json'
            inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            r=json.loads(p.read_text(encoding='utf-8'))
            assert r['epoch']==e
            records.append(r)
        best=max((r for r in records if r.get('validation')), key=lambda r:(r['validation']['selection_score'],-r['epoch']))
        assert records[-1]['best_epoch']==best['epoch']
        patients={}
        for case in best['validation']['cases']:
            patients.setdefault(case['patient'], np.zeros((6,6),dtype=np.int64))
            patients[case['patient']]+=np.asarray(case['confusion'])
        assert len(patients)==6 and len(best['validation']['cases'])==38
        scores=[]
        for matrix in patients.values():
            tp=np.diag(matrix);true=matrix.sum(1);pred=matrix.sum(0)
            scores.append({'dice':2*tp/(true+pred),'iou':tp/(true+pred-tp),'precision':tp/pred,'recall':tp/true})
        for i,label in enumerate(NAMES):
            rows.append({'model':name,'epoch':best['epoch'],'class':label,
                         **{k:float(np.mean([s[k][i] for s in scores])) for k in scores[0]}})
        score=float(np.mean([s['dice'][3:].mean() for s in scores]))
        assert abs(score-best['validation']['selection_score'])<1e-12
        selection.append({'model':name,'epoch':best['epoch'],'patient_equal_small_organ_dice':score})
    for filename,data in [('six_class_validation.csv',rows),('selection.csv',selection)]:
        with (OUT/filename).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (OUT/'source_checksums.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,5))
    vals=[s['patient_equal_small_organ_dice'] for s in selection]
    bars=ax.bar([s['model'] for s in selection],vals,color=['#698da7']*4+['#268153','#698da7','#cf8635'])
    for b,s in zip(bars,selection):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.008,f"{b.get_height():.4f}\ne{s['epoch']}",ha='center',fontsize=9)
    ax.set(ylim=(0,1),ylabel='Patient-equal small-organ Dice',title='Best measured Validation checkpoint per model; 6 patients, one seed')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'seven_model_selection.{ext}',dpi=180)
    plt.close(fig)
    data=np.array([[r['dice'] for r in rows if r['model']==s['model']] for s in selection])
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(data,vmin=.65,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(6),NAMES);ax.set_yticks(range(7),[s['model'] for s in selection])
    for i in range(7):
        for j in range(6):ax.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',color='white' if data[i,j]>.88 else 'black')
    ax.set_title('Patient-equal six-class Validation Dice; each model at its own selected epoch')
    fig.colorbar(im,ax=ax,label='Dice');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(OUT/f'seven_model_six_class_dice.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
