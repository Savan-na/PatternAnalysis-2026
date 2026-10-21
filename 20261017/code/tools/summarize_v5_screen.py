"""Recompute Validation-only C/L/S evidence from immutable training JSON."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from hipmri_rewrite.metrics import overlap
from hipmri_rewrite.training import patient_small_organ_score
from hipmri_rewrite.v5_branches import validate_config, learning_rate

BASE = ROOT/'results/experiments/improved_3d_v5'
OUT = BASE/'comparisons/validation_analysis'
NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_table(name, rows):
    with (OUT/'tables'/f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(name, fig):
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT/'figures'/f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    for name in ('tables', 'figures'):
        (OUT/name).mkdir(parents=True, exist_ok=True)
    summary, classes, surfaces, patients, case_metrics, curves, provenance = [], [], [], [], [], [], {}
    reference = None
    selected = {}
    for arm in 'CLS':
        directory = BASE/arm.lower()/'training_records'
        files = sorted(directory.glob('*.json'))
        assert len(files) == 52, f'{arm}: expected exactly 52 records'
        config = json.loads((directory/'config_frozen.json').read_text())
        validate_config(config)
        manifest = json.loads((directory/'manifest.json').read_text())
        assert manifest['config_sha256'] == sha(directory/'config_frozen.json')
        assert manifest['config_sha256'] == sha(ROOT/f'code/config/v5_{arm.lower()}_frozen.json')
        assert manifest['trainer_sha256'] == sha(ROOT/'code/tools/train_v5_branch.py')
        assert manifest['branch_helper_sha256'] == sha(ROOT/'code/hipmri_rewrite/v5_branches.py')
        rows = [json.loads((directory/f'epoch_{i:03d}.json').read_text()) for i in range(1, 51)]
        schedule = []
        for i, row in enumerate(rows, 1):
            assert row['epoch'] == i and len(row['train']['batch_history']) == 130
            assert row['train']['ce_class_weights'] == config['objective']['ce_class_weights']
            for b, batch in enumerate(row['train']['batch_history']):
                # Windows/Linux libm cosine can differ by a few floating-point ULPs.
                assert np.isclose(batch['learning_rate_used'], learning_rate(arm, (i-1)*130+b), rtol=1e-13, atol=1e-18)
                assert np.isclose(batch['learning_rate_next'], learning_rate(arm, (i-1)*130+b+1), rtol=1e-13, atol=1e-18)
            schedule.append((row['selected_cases'], row['train']['exposure']))
            curves.append(dict(arm=arm, epoch=i, train_loss=row['train']['mean_loss'],
                               ce=row['train']['mean_ce_component'], dice_loss=row['train']['mean_dice_component'],
                               learning_rate=row['train']['learning_rate'],
                               validation_score=(row['validation'] or {}).get('selection_score')))
        if reference is None:
            reference = schedule
        else:
            assert schedule == reference, 'Actual sampling/exposure differs between arms'
        candidates = [r for r in rows if r['validation'] is not None]
        assert [r['epoch'] for r in candidates] == list(range(5, 51, 5))
        best = max(candidates, key=lambda r: r['validation']['selection_score'])
        assert rows[-1]['best_epoch'] == best['epoch']
        assert rows[-1]['best_score'] == best['validation']['selection_score']
        for row in candidates:
            matrices = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
            cases = row['validation']['cases']
            assert len(cases) == 38 and len({c['case'] for c in cases}) == 38
            for case in cases:
                matrix = np.asarray(case['confusion'], dtype=np.int64)
                assert matrix.shape == (6, 6) and (matrix >= 0).all()
                matrices[case['patient']] += matrix
            assert len(matrices) == 6
            score, details = patient_small_organ_score(matrices)
            assert abs(score-row['validation']['selection_score']) < 1e-12
            assert details == row['validation']['patient_small_organ_dice']
            assert np.array_equal(sum(matrices.values()), row['validation']['pooled_confusion'])
        v = best['validation']
        selected[arm] = best
        matrices = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
        for case in v['cases']:
            matrices[case['patient']] += np.asarray(case['confusion'])
            metrics = overlap(np.asarray(case['confusion']))
            for k, name in enumerate(NAMES):
                case_metrics.append(dict(arm=arm, epoch=best['epoch'], case=case['case'], patient=case['patient'],
                                         label=name, **{m: metrics[m][k] for m in metrics}))
        for patient, matrix in sorted(matrices.items()):
            metrics = overlap(matrix)
            patients.append(dict(arm=arm, patient=patient,
                                 small_organ_dice=np.mean(metrics['dice'][3:])))
        patient_overlap = [overlap(m) for m in matrices.values()]
        for k, name in enumerate(NAMES):
            classes.append(dict(arm=arm, epoch=best['epoch'], label=name,
                                **{m: float(np.mean([p[m][k] for p in patient_overlap])) for m in patient_overlap[0]}))
        for name in NAMES[1:]:
            values = [c['surfaces'][name] for c in v['cases']]
            assert all(s['status'] == 'both_present' for s in values)
            hd = np.array([s['hd95_mm'] for s in values])
            surfaces.append(dict(arm=arm, label=name, hd95_median_mm=float(np.median(hd)),
                                 hd95_p95_mm=float(np.percentile(hd, 95)), hd95_max_mm=float(hd.max()),
                                 assd_median_mm=float(np.median([s['assd_mm'] for s in values]))))
        summary.append(dict(arm=arm, best_epoch=best['epoch'], selection=v['selection_score'],
                            final_selection=rows[-1]['validation']['selection_score'],
                            far_body_fp=sum(c['far_background_to_body_voxels'] for c in v['cases']),
                            training_checkpoint_seconds=sum(r['train']['epoch_seconds'] for r in rows),
                            validation_seconds=sum(r['validation']['seconds'] for r in candidates),
                            peak_train_allocated_gib=max(r['train']['peak_allocated_bytes'] for r in rows)/1024**3))
        file_hashes = {p.name: sha(p) for p in files}
        aggregate = hashlib.sha256(''.join(f'{n} {h}\n' for n, h in sorted(file_hashes.items())).encode()).hexdigest()
        provenance[arm] = dict(files=file_hashes, aggregate_sha256=aggregate, manifest=manifest)
    for name, rows in [('summary', summary), ('six_class_patient_equal', classes), ('surface_metrics', surfaces),
                       ('patient_scores', patients), ('case_metrics', case_metrics), ('training_curves', curves)]:
        save_table(name, rows)
    (OUT/'integrity.json').write_text(json.dumps(dict(status='local structural/source/numerical checks passed; remote aggregate hash pending',
                                                     provenance=provenance), indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for arm in 'CLS':
        r = [x for x in curves if x['arm'] == arm]
        epochs = [x['epoch'] for x in r]
        axes[0, 0].plot(epochs, [x['train_loss'] for x in r], label=arm)
        vr = [x for x in r if x['validation_score'] is not None]
        axes[0, 1].plot([x['epoch'] for x in vr], [x['validation_score'] for x in vr], marker='o', label=arm)
        axes[1, 0].plot(epochs, [x['learning_rate'] for x in r], label=arm)
        axes[1, 1].plot(epochs, [x['dice_loss'] for x in r], label=arm)
    for ax, title in zip(axes.flat, ['Train total loss (L objective differs)', 'Validation patient-equal small-organ Dice',
                                    'Learning rate after epoch', 'Train foreground Dice loss']):
        ax.set(title=title, xlabel='Epoch')
        ax.legend()
        ax.grid(alpha=.2)
    save_figure('training_and_selection', fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric in zip(axes.flat, ('dice', 'iou', 'precision', 'recall')):
        for i, arm in enumerate('CLS'):
            r = [x for x in classes if x['arm'] == arm]
            ax.bar(np.arange(6)+(i-1)*.25, [x[metric] for x in r], width=.25, label=arm)
        ax.set_xticks(range(6), NAMES, rotation=20)
        ax.set(title=metric.title()+' — mean of six patient metrics', ylim=(0, 1.04))
        ax.legend(loc='lower left')
    save_figure('six_class_metrics', fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in 'CLS':
        r = [x for x in patients if x['arm'] == arm]
        axes[0].plot([x['patient'] for x in r], [x['small_organ_dice'] for x in r], marker='o', label=arm)
        s = [x for x in surfaces if x['arm'] == arm]
        axes[1].plot([x['label'] for x in s], [x['hd95_p95_mm'] for x in s], marker='o', label=arm)
    axes[0].set(title='Patient-equal small-organ Dice', ylabel='Dice')
    axes[1].set(title='HD95 tail across 38 volumes', ylabel='95th percentile of volume HD95 (mm)')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    save_figure('patients_and_boundary_tails', fig)
    print(json.dumps(dict(summary=summary, classes=classes, surfaces=surfaces, patients=patients), indent=2))


if __name__ == '__main__':
    main()
