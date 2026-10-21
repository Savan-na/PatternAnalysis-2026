"""Build an isolated ten-model gallery from Validation-best records and fixed Test audit."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from tools.summarize_all_model_validation import DIRS as OLD_DIRS
from hipmri_rewrite.metrics import overlap
from hipmri_rewrite.training import patient_small_organ_score

OUT = ROOT/'results/reports/model_performance_comparison'
DIRS = dict(OLD_DIRS, **{f'v5-{a.upper()}': ROOT/f'results/experiments/improved_3d_v5/{a}/training_records' for a in 'cls'})
NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']
MODELS = list(DIRS)
COLORS = plt.get_cmap('tab10').colors


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(name, rows):
    with (OUT/'tables'/f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(name, fig):
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def heat(name, data, labels, columns, title, boundary=False):
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(labels)*.48+1.5)))
    kwargs = dict(norm=LogNorm(vmin=1, vmax=max(10, np.max(data))), cmap='YlOrRd') if boundary else dict(vmin=0, vmax=1, cmap='YlGnBu')
    im = ax.imshow(data, aspect='auto', **kwargs)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(columns)):
            value = data[i, j]
            ax.text(j, i, f'{value:.1f}' if boundary else f'{value:.3f}', ha='center', va='center',
                    color='white' if im.norm(value) > .68 else 'black', fontsize=10)
    ax.set_title(title, pad=15)
    fig.colorbar(im, ax=ax, label='mm (log color scale); lower is better' if boundary else 'Score; higher is better')
    save(name, fig)


def main():
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
    for name, rows in [('six_class_metrics', metric_rows), ('surface_metrics', surface_rows), ('model_summary', summary),
                       ('patient_scores', patient_rows), ('training_curves', curves), ('available_resource_benchmarks', resource_rows)]:
        table(name, rows)
    for split in ('validation', 'test'):
        models = MODELS
        labels = [f'{m} (e{selected[m]})' for m in models]
        scope = 'Validation: 10 models, 6 patients / 38 volumes' if split == 'validation' else 'Existing Test: 10 models, 6 patients / 30 volumes; descriptive only'
        for metric in ('dice', 'iou', 'precision', 'recall'):
            data = np.array([[next(r[metric] for r in metric_rows if r['model']==m and r['split']==split and r['label']==c) for c in NAMES] for m in models])
            heat(f'{split}_{metric}_six_classes', data, labels, NAMES, scope+'\nPatient-equal '+metric+'; each model at its own Validation-best checkpoint')
        for metric in ('hd95_median_mm', 'hd95_p95_mm', 'hd95_max_mm', 'assd_median_mm'):
            data = np.array([[next(r[metric] for r in surface_rows if r['model']==m and r['split']==split and r['label']==c) for c in NAMES[1:]] for m in models])
            heat(f'{split}_{metric}', data, labels, NAMES[1:], scope+'\n'+metric+' across volumes; lower is better', boundary=True)
        patients = sorted({r['patient'] for r in patient_rows if r['split']==split})
        data = np.array([[next(r['small_organ_dice'] for r in patient_rows if r['model']==m and r['split']==split and r['patient']==p) for p in patients] for m in models])
        heat(f'{split}_patient_small_organ_dice', data, labels, patients, scope+'\nSmall-organ Dice per patient; repeated visits pooled within patient')
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(MODELS))
    for split, offset, color in [('validation', -.18, '#2878aa'), ('test', .18, '#d58a32')]:
        vals = [next((r['small_organ_dice'] for r in summary if r['model']==m and r['split']==split), np.nan) for m in MODELS]
        bars = ax.bar(x+offset, vals, width=.34, color=color, label=split)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x()+bar.get_width()/2, val+.008, f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=30)
    ax.set_xticks(x, [f'{m}\ne{selected[m]}' for m in MODELS])
    ax.set(ylim=(0, 1.06), ylabel='Patient-equal small-organ Dice', title='All 10 selected checkpoints: Validation and existing descriptive Test\nEach checkpoint fixed by Validation; Test is not a new blind evaluation')
    ax.legend(loc='upper left')
    save('all_models_primary_dice', fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for color, model in zip(COLORS, MODELS):
        rows = [r for r in curves if r['model']==model]
        vr = [r for r in rows if r['validation_dice'] is not None]
        axes[0].plot([r['epoch'] for r in vr], [r['validation_dice'] for r in vr], marker='.', label=model, color=color)
        axes[1].plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows], label=model, color=color)
    axes[0].set(title='Validation selection trajectories; same primary aggregation', ylabel='Small-organ Dice')
    axes[1].set(title='Train loss trajectories: objectives/sampling differ; not a quality ranking', ylabel='Reported total loss', xlabel='Epoch')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(ncol=5, fontsize=8)
    save('all_models_training_trajectories', fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, title in zip(axes, ['parameters','peak_allocated_gib','median_reference_case_seconds'],
                              ['Parameter count', 'Peak allocated GPU memory (GiB)', 'Reference-case inference median (s)']):
        ax.bar([r['model'] for r in resource_rows], [r[key] for r in resource_rows])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', labelrotation=40)
    fig.suptitle('Fixed-case benchmarks: 7 historical models plus v5 completion\nSame prepared case, warmup/repeats; separate jobs/GPU load can affect timing', fontsize=11)
    save('available_resource_benchmarks', fig)
    (OUT/'source_checksums.json').write_text(json.dumps(dict(selected_epochs=selected, input_sha256=sources,
                                                           script_sha256=sha(Path(__file__)), test_scope='All 10 models; existing previously inspected cohort; descriptive only'), indent=2)+'\n')
    print(f'Generated {len(list((OUT/"figures").glob("*.png")))} figures (PNG + SVG), six CSV tables; selected={selected}')


if __name__ == '__main__':
    main()
