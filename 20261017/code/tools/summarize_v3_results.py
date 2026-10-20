"""Recompute v3 Validation comparisons from immutable JSON; no MRI/Test access."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/experiments/improved_3d_v3'
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


def main():
    for d in ('tables','provenance','reports'):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    dirs = {'Standard 3D': ROOT/'logs/training/baseline_3d',
            'Improved v1': ROOT/'logs/training/improved_3d',
            'Improved v2': ROOT/'results/experiments/improved_3d_v2/training_records',
            'Improved v3': OUT/'training_records'}
    runs = {k: [read(p/f'epoch_{e:03d}.json') for e in range(1, 51)] for k,p in dirs.items()}
    manifest = read(dirs['Improved v3']/'manifest.json')
    config = read(dirs['Improved v3']/'config_frozen.json')
    assert manifest['config_sha256'] == SOURCES[str((dirs['Improved v3']/'config_frozen.json').relative_to(ROOT))]
    assert manifest['model'] == 'improved_3d_v3' and manifest['max_epochs'] == 50
    val_epochs = []
    for e, record in enumerate(runs['Improved v3'], 1):
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
    assert selected['Improved v3']['epoch']==50
    assert manifest['config_sha256']==hashlib.sha256((ROOT/'code/config/v3_gn1_frozen.json').read_bytes()).hexdigest()
    arms = selected
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
        for r in runs['Improved v3'] if r['validation']])
    before={r['patient']:r for r in patient_rows if r['model']=='Improved v1'}
    after={r['patient']:r for r in patient_rows if r['model']=='Improved v3'}
    delta={p:{n:after[p][n]-before[p][n] for n in ['small_organ_dice',*NAMES]} for p in before}
    table('v3_minus_v1_patient_delta.csv',[dict(patient=p,**d) for p,d in delta.items()])
    table('foreground_surfaces_v3.csv',[dict(case=c['case'],patient=c['patient'],label=n,**s)
          for c in selected['Improved v3']['validation']['cases'] for n,s in c['surfaces'].items()])
    std={r['patient']:r for r in patient_rows if r['model']=='Standard 3D'}
    table('v3_minus_standard_patient_delta.csv',[dict(patient=p,**{n:after[p][n]-std[p][n] for n in ['small_organ_dice',*NAMES]}) for p in after])
    result={'summary':summary,'v3_minus_v1_patient_delta':delta,
            'verification':{'epochs':50,'batches':6500,'validation_epochs':val_epochs,
                'identical_v1_patient_schedule_and_class_exposure':True,
                'local_hashes_recorded_not_yet_remote_verified':True}}
    v1=summary['Improved v1']; v3=summary['Improved v3']
    result['decision_gates']={
        'body_hd95_reduction_at_least_50pct':v3['body_hd95_volume_median']<=.5*v1['body_hd95_volume_median'],
        'at_least_five_patients_body_improved':sum(v3['patient_body_hd95_median'][p]<h for p,h in v1['patient_body_hd95_median'].items())>=5,
        'far_fp_rate_reduction_at_least_50pct':v3['patient_equal_far_fp_rate']<=.5*v1['patient_equal_far_fp_rate'],
        'small_organ_drop_at_most_001':v3['small_organ_score']>=v1['small_organ_score']-.01,
        'each_foreground_drop_at_most_001':all(next(r['patient_equal_dice'] for r in class_rows if r['model']=='Improved v3' and r['label']==n)>=next(r['patient_equal_dice'] for r in class_rows if r['model']=='Improved v1' and r['label']==n)-.01 for n in NAMES[1:])}
    (OUT/'tables/analysis_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'provenance/local_source_checksums.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print("Canonical charts: python code/tools/build_best_checkpoint_review.py")


if __name__=='__main__': main()
