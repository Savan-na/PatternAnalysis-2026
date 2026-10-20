"""Recompute v2 Validation comparisons from immutable JSON; no MRI/Test access."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/experiments/improved_3d_v2'
NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(matrix):
    a = np.array(matrix, dtype=float)
    tp = a.diagonal()
    def divide(num, den):
        return np.divide(num, den, out=np.full(6, np.nan), where=den != 0)
    return dict(dice=divide(2*tp, a.sum(0)+a.sum(1)),
                iou=divide(tp, a.sum(0)+a.sum(1)-tp),
                precision=divide(tp, a.sum(0)), recall=divide(tp, a.sum(1)))


def patient_matrices(record):
    groups = {}
    for row in record['validation']['cases']:
        groups.setdefault(row['patient'], np.zeros((6, 6), dtype=np.int64))
        groups[row['patient']] += np.array(row['confusion'])
    return groups


def score(record):
    return float(np.mean([metrics(m)['dice'][3:].mean() for m in patient_matrices(record).values()]))


def table(name, rows):
    with (OUT/'tables'/name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    dirs = {'Standard 3D': ROOT/'logs/training/baseline_3d',
            'Improved v1': ROOT/'logs/training/improved_3d',
            'Improved v2': OUT/'training_records'}
    runs = {k: [read(p/f'epoch_{e:03d}.json') for e in range(1, 51)] for k,p in dirs.items()}
    manifest = read(dirs['Improved v2']/'manifest.json')
    config = read(dirs['Improved v2']/'config_frozen.json')
    assert manifest['config_sha256'] == SOURCES[str((dirs['Improved v2']/'config_frozen.json').relative_to(ROOT))]
    assert manifest['model'] == 'improved_3d_v2' and manifest['max_epochs'] == 50
    val_epochs = []
    for e, record in enumerate(runs['Improved v2'], 1):
        assert record['epoch'] == e and len(record['train']['batch_history']) == 130
        assert len(record['selected_cases']) == 26
        assert [b['batch'] for b in record['train']['batch_history']] == list(range(1,131))
        assert np.isfinite([b['loss'] for b in record['train']['batch_history']]).all()
        baseline = runs['Improved v1'][e-1]
        assert record['selected_cases'] == baseline['selected_cases']
        assert record['train']['exposure'] == baseline['train']['exposure']
        if record['validation'] is not None:
            val_epochs.append(e)
            assert len(record['validation']['cases']) == 38
            assert len(patient_matrices(record)) == 6
            pooled = sum(patient_matrices(record).values())
            assert np.array_equal(pooled, record['validation']['pooled_confusion'])
            assert np.isclose(score(record), record['validation']['selection_score'], atol=1e-12)
    assert val_epochs == list(range(5,51,5))
    selected = {k: max((r for r in records if r['validation']), key=score) for k,records in runs.items()}
    arms = {**selected, 'v2 epoch50': runs['Improved v2'][-1]}
    class_rows, patient_rows = [], []
    for arm, record in arms.items():
        groups = patient_matrices(record)
        pm = [metrics(m) for m in groups.values()]
        pooled = metrics(record['validation']['pooled_confusion'])
        for j,n in enumerate(NAMES):
            class_rows.append(dict(model=arm, epoch=record['epoch'], label=n,
                **{f'pooled_{m}':float(v[j]) for m,v in pooled.items()},
                **{f'patient_equal_{m}':float(np.nanmean([v[m][j] for v in pm])) for m in pooled}))
        for patient,m in sorted(groups.items()):
            values = metrics(m)['dice']
            patient_rows.append(dict(model=arm, epoch=record['epoch'], patient=patient,
                                     small_organ_dice=float(values[3:].mean()),
                                     **{n:float(values[j]) for j,n in enumerate(NAMES)}))
    table('six_class_comparison.csv',class_rows); table('patient_comparison.csv',patient_rows)
    original_body = read(ROOT/'results/tables/comparisons/g5_validation_body_audit.json')
    original_cases = {r['case']:r for r in original_body['cases']}
    body_rows=[]
    for arm, record in arms.items():
        for row in record['validation']['cases']:
            if arm in ('Standard 3D','Improved v1'):
                b=original_cases[row['case']]['models']['standard_3d' if arm=='Standard 3D' else 'improved_3d']
                hd=b['boundary']['hd95_mm']; far=b['background_to_body_at_least_20mm']
            else:
                hd=row['surfaces']['Body']['hd95_mm']; far=row['far_background_to_body_voxels']
            body_rows.append(dict(model=arm,case=row['case'],patient=row['patient'],hd95_mm=hd,
                far_fp=far,background_voxels=int(np.array(row['confusion'])[0].sum())))
    table('body_case_comparison.csv',body_rows)
    summary={}
    for arm, record in arms.items():
        rows=[r for r in body_rows if r['model']==arm]
        ps={p:[r for r in rows if r['patient']==p] for p in sorted({r['patient'] for r in rows})}
        summary[arm]={'epoch':record['epoch'],'small_organ_score':score(record),
            'body_hd95_volume_median':float(np.median([r['hd95_mm'] for r in rows])),
            'far_fp_total':sum(r['far_fp'] for r in rows),
            'patient_equal_far_fp_rate':float(np.mean([sum(r['far_fp'] for r in rr)/sum(r['background_voxels'] for r in rr) for rr in ps.values()])),
            'patient_body_hd95_median':{p:float(np.median([r['hd95_mm'] for r in rr])) for p,rr in ps.items()}}
    table('validation_trajectory.csv',[dict(epoch=r['epoch'],score=score(r),
        body_hd95_median=float(np.median([c['surfaces']['Body']['hd95_mm'] for c in r['validation']['cases']])),
        far_fp=sum(c['far_background_to_body_voxels'] for c in r['validation']['cases']))
        for r in runs['Improved v2'] if r['validation']])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for arm,records in runs.items():
        vv=[r for r in records if r['validation']]
        axes[0].plot([r['epoch'] for r in vv],[score(r) for r in vv],'-o',label=arm)
    axes[0].set(title='Patient-equal small-organ Dice',xlabel='Epoch',ylim=(0,1)); axes[0].legend(fontsize=8)
    vr=[r for r in runs['Improved v2'] if r['validation']]
    axes[1].plot([r['epoch'] for r in vr],[np.median([c['surfaces']['Body']['hd95_mm'] for c in r['validation']['cases']]) for r in vr],'-o')
    axes[1].set(title='v2 Body HD95 (volume median)',xlabel='Epoch',ylabel='mm')
    for key in ('main','half','quarter'):
        axes[2].plot(range(1,51),[np.mean([b['head_losses'][key] for b in r['train']['batch_history']]) for r in runs['Improved v2']],label=key)
    axes[2].set(title='v2 independent head training losses',xlabel='Epoch',ylabel='CE/Dice term'); axes[2].legend()
    save(fig,'training_and_boundary_trajectory')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,m in zip(axes.flat,['dice','iou','precision','recall']):
        for i,arm in enumerate(arms):
            vals=[r[f'patient_equal_{m}'] for r in class_rows if r['model']==arm]
            ax.bar(np.arange(6)+(i-1.5)*.19, vals,.19,label=arm)
        ax.set_xticks(range(6),NAMES,rotation=20); ax.set_ylim(0,1.05); ax.set_title('Patient-equal '+m)
    axes[0,0].legend(fontsize=8); save(fig,'all_six_classes')
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    patients=sorted(patient_matrices(selected['Improved v2']))
    for arm in arms:
        vals=[next(r['small_organ_dice'] for r in patient_rows if r['model']==arm and r['patient']==p) for p in patients]
        ax.plot(patients,vals,'-o',label=arm)
    ax.set(ylim=(0,1),ylabel='Small-organ Dice',title='Six independent Validation patients; visits pooled within patient'); ax.legend()
    save(fig,'patient_comparison')
    before={r['patient']:r for r in patient_rows if r['model']=='Improved v2'}
    after={r['patient']:r for r in patient_rows if r['model']=='v2 epoch50'}
    delta={p:{n:after[p][n]-before[p][n] for n in ['small_organ_dice',*NAMES]} for p in before}
    table('epoch45_to_50_patient_delta.csv',[dict(patient=p,**d) for p,d in delta.items()])
    flows=[]
    for epoch in (45,50):
        record=runs['Improved v2'][epoch-1]
        case=next(c for c in record['validation']['cases'] if c['case']=='S022_Week2')
        matrix=np.array(case['confusion'])
        for k in (3,4,5):
            flows.append(dict(epoch=epoch,case=case['case'],target=NAMES[k],
                true_positive=int(matrix[k,k]),false_positive=int(matrix[:,k].sum()-matrix[k,k]),
                false_negative=int(matrix[k].sum()-matrix[k,k]),
                background_to_target=int(matrix[0,k]),body_to_target=int(matrix[1,k])))
    table('S022_Week2_error_counts.csv',flows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,e in zip(axes,(45,50)):
        rr=[r for r in flows if r['epoch']==e]
        for j,key in enumerate(('true_positive','false_positive','false_negative')):
            ax.bar(np.arange(3)+(j-1)*.25,[r[key] for r in rr],.25,label=key.replace('_',' '))
        ax.set_xticks(range(3),['Bladder','Rectum','Prostate'])
        ax.set(title=f'S022 Week2 | v2 epoch {e}',ylabel='Whole-volume voxel count',ylim=(0,9500))
        ax.legend(fontsize=8)
    save(fig,'S022_Week2_error_counts')
    result={'summary':summary,'epoch45_to_50_patient_delta':delta,
            'verification':{'epochs':50,'batches':6500,'validation_epochs':val_epochs,
                'identical_v1_patient_schedule_and_class_exposure':True,
                'local_hashes_recorded_not_yet_remote_verified':True}}
    (OUT/'tables/analysis_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'provenance/local_source_checksums.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
