"""Verify ten-model evidence and build the compact foreground/prostate report."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from tools.summarize_all_model_validation import DIRS as OLD_DIRS
from hipmri_rewrite.metrics import overlap
from hipmri_rewrite.training import patient_small_organ_score

OUT = ROOT/'results/reports/model_performance_comparison'
DIRS = dict(OLD_DIRS, **{f'v5-{a.upper()}': ROOT/f'results/experiments/improved_3d_v5/{a}/training_records' for a in 'cls'})
NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']
MODELS = list(DIRS)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect_evidence():
    for name in ('figures', 'tables'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    sources, selected, curves, metric_rows, surface_rows, patient_rows, summary, resource_rows = {}, {}, [], [], [], [], [], []
    raw_path = ROOT/'results/experiments/improved_3d_v4/comprehensive_audit_20261021/metrics.json'
    assert sha(raw_path) == '74c298b956b684be3c377e495f70d6deb03445a9f5782663b189c33cdc6dbbc2'
    raw = json.loads(raw_path.read_text())
    assert raw['complete']
    sources[str(raw_path.relative_to(ROOT))] = sha(raw_path)
    v5_test_path = ROOT/'results/experiments/improved_3d_v5/comparisons/test_assessment_20261021/metrics.json'
    assert sha(v5_test_path) == '454a8080d1ecff3fb0bfe0d675af420cad5a26f69f236458e0036ed56e511986'
    v5_test = json.loads(v5_test_path.read_text())
    assert v5_test['complete'] and set(v5_test['arms']) == set('CLS')
    assert v5_test['script_sha256'] == sha(ROOT/'code/tools/evaluate_v5_missing.py')
    sources[str(v5_test_path.relative_to(ROOT))] = sha(v5_test_path)
    sources['code/tools/evaluate_v5_missing.py'] = sha(ROOT/'code/tools/evaluate_v5_missing.py')
    image_receipt_path = ROOT/'results/experiments/improved_3d_v5/case_studies/validation/best_cls_review/provenance.json'
    image_receipt = json.loads(image_receipt_path.read_text())
    sources[str(image_receipt_path.relative_to(ROOT))] = sha(image_receipt_path)
    v5_receipt_path = ROOT/'results/experiments/improved_3d_v5/comparisons/validation_analysis/remote_record_hashes.json'
    v5_receipt = json.loads(v5_receipt_path.read_text())
    sources[str(v5_receipt_path.relative_to(ROOT))] = sha(v5_receipt_path)
    for model, directory in DIRS.items():
        records = []
        for epoch in range(1, 51):
            path = directory/f'epoch_{epoch:03d}.json'
            row = json.loads(path.read_text())
            assert row['epoch'] == epoch
            sources[str(path.relative_to(ROOT))] = sha(path)
            records.append(row)
            curves.append(dict(model=model, epoch=epoch, train_loss=row['train']['mean_loss'],
                               validation_dice=(row.get('validation') or {}).get('selection_score')))
        best = max((r for r in records if r.get('validation')), key=lambda r: (r['validation']['selection_score'], -r['epoch']))
        assert best['epoch'] == records[-1]['best_epoch']
        selected[model] = best['epoch']
        if model.startswith('v5-'):
            aggregate = hashlib.sha256(''.join(p.name+' '+sha(p)+'\n' for p in sorted(directory.glob('*.json'))).encode()).hexdigest()
            assert aggregate == v5_receipt['hashes'][model[-1]]
        splits = {'validation': best['validation']['cases']}
        if model.startswith('v5-'):
            arm = model[-1]
            test_arm = v5_test['arms'][arm]
            assert test_arm['selected']['epoch'] == best['epoch']
            assert test_arm['selected']['record_sha256'] == aggregate
            assert test_arm['selected']['checkpoint_sha256'] == image_receipt['checkpoints'][arm]['checkpoint_sha256']
            assert test_arm['selected']['validation_replay_case'] == 'K008_Week0'
            assert abs(test_arm['selected']['validation_score']-best['validation']['selection_score']) < 1e-12
            splits['test'] = test_arm['cases']
            old_cases = {c['case']: c for c in raw['arms']['Standard']['splits']['test']}
            assert {c['case'] for c in splits['test']} == set(old_cases)
            for c in splits['test']:
                assert all(c[key] == old_cases[c['case']][key] for key in ('mri_sha256', 'mask_sha256', 'patient'))
            benchmark = test_arm['benchmark']
            resource_rows.append(dict(model=model, parameters=test_arm['selected']['parameters'],
                                      peak_allocated_gib=benchmark['peak_allocated_bytes']/1024**3,
                                      median_reference_case_seconds=benchmark['median_seconds'], reference_case=benchmark['case']))
        if model in raw['arms']:
            assert raw['arms'][model]['selected']['epoch'] == best['epoch']
            audited = {r['case']: r for r in raw['arms'][model]['splits']['validation']}
            assert all(r['confusion'] == audited[r['case']]['confusion'] for r in splits['validation'])
            # Early training JSON did not contain surfaces; the fixed audit replayed
            # these exact selected predictions and measured their physical surfaces.
            splits['validation'] = raw['arms'][model]['splits']['validation']
            splits['test'] = raw['arms'][model]['splits']['test']
            benchmark = raw['arms'][model]['benchmark']
            resource_rows.append(dict(model=model, parameters=benchmark['parameters'],
                                      peak_allocated_gib=benchmark['peak_allocated_bytes']/1024**3,
                                      median_reference_case_seconds=benchmark['median_seconds'],
                                      reference_case=benchmark['case']))
        for split, cases in splits.items():
            assert len(cases) == (38 if split == 'validation' else 30)
            matrices = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
            for case in cases:
                matrices[case['patient']] += np.asarray(case['confusion'])
            assert len(matrices) == 6
            score, _ = patient_small_organ_score(matrices)
            if split == 'validation':
                assert abs(score-best['validation']['selection_score']) < 1e-12
            elif model.startswith('v5-'):
                assert abs(score-test_arm['summary']['patient_equal_small_organ_dice']) < 1e-12
            for patient, matrix in sorted(matrices.items()):
                metrics = overlap(matrix)
                patient_rows.append(dict(model=model, split=split, patient=patient,
                                         small_organ_dice=np.mean(metrics['dice'][3:])))
            metrics = [overlap(m) for m in matrices.values()]
            for k, name in enumerate(NAMES):
                metric_rows.append(dict(model=model, split=split, epoch=best['epoch'], label=name,
                                        **{m:float(np.mean([p[m][k] for p in metrics])) for m in metrics[0]}))
            summary.append(dict(model=model, split=split, best_epoch=best['epoch'], small_organ_dice=score,
                                foreground_macro_dice=float(np.mean([p['dice'][1:] for p in metrics])),
                                six_class_macro_dice=float(np.mean([p['dice'] for p in metrics]))))
            for name in NAMES[1:]:
                values = [c['surfaces'][name] for c in cases]
                assert all(s['status'] == 'both_present' for s in values)
                hd = [s['hd95_mm'] for s in values]
                surface_rows.append(dict(model=model, split=split, label=name,
                                         hd95_median_mm=float(np.median(hd)), hd95_p95_mm=float(np.percentile(hd, 95)),
                                         hd95_max_mm=float(np.max(hd)), assd_median_mm=float(np.median([s['assd_mm'] for s in values]))))
    return dict(sources=sources, selected=selected, curves=curves, metrics=metric_rows,
                surfaces=surface_rows, patients=patient_rows, summary=summary, resources=resource_rows)


def main():
    from tools.summarize_prostate_priority import render
    render(collect_evidence())


if __name__ == '__main__':
    main()
