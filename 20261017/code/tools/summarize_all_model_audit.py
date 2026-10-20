"""Verify the seven-model remote receipt, then export assessment and research evidence.

Raw audit outputs are never modified. Test is previously inspected/descriptive;
model choice is locked from recorded Validation, not from these Test results.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from hipmri_rewrite.metrics import overlap
from tools.summarize_all_model_validation import DIRS

SOURCE = ROOT/'results/experiments/improved_3d_v4/comprehensive_audit_20261021'
OUT = ROOT/'results/experiments/improved_3d_v4/assessment_analysis'
MODELS = ['2D', 'Standard', 'v1', 'v2', 'v3', 'A0', 'A1']
NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']
COLORS = ['#999999', '#326ba8', '#bd6651', '#9c7aba', '#218654', '#779da1', '#d28b2e']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify():
    checks = read(SOURCE/'checksums.json')
    assert checks and all(sha(SOURCE/n) == h for n, h in checks.items()), 'Download checksum mismatch'
    data = read(SOURCE/'metrics.json')
    assert data['complete'] is True and list(data['arms']) == MODELS
    assert data['script_sha256'] == sha(ROOT/'code/tools/audit_all_models.py')
    lock_path = ROOT/'code/config/all_model_assessment_audit.json'
    assert data['lock_sha256'] == sha(lock_path)
    lock = read(lock_path)
    assert sha(ROOT/lock['v4_pretest_plan_path']) == data['v4_pretest_plan_sha256']
    historical_path = ROOT/'results/tables/comparisons/g6_test_3d.json'
    assert sha(historical_path) == lock['historical_test_sha256']
    historical = read(historical_path)
    inputs = read(SOURCE/'input_provenance.json')
    assert inputs['exact_cross_split_mri_duplicates'] == []
    case_sets = {}
    reproduced = 0
    for model, arm in data['arms'].items():
        expected = lock['selected'][model]
        assert all(arm['selected'][k] == v for k, v in expected.items())
        directory = DIRS[model]
        assert sha(directory/'manifest.json') == expected['manifest_sha256']
        records = [read(directory/f'epoch_{e:03}.json') for e in range(1, 51)]
        best = max((r for r in records if r.get('validation')),
                   key=lambda r: (r['validation']['selection_score'], -r['epoch']))
        assert best['epoch'] == expected['epoch'] == records[-1]['best_epoch']
        assert sha(directory/f"epoch_{expected['epoch']:03}.json") == expected['record_sha256']
        saved = {r['case']: r for r in best['validation']['cases']}
        for split, rows in arm['splits'].items():
            assert len(rows) == (38 if split == 'validation' else 30)
            assert len({r['case'] for r in rows}) == len(rows)
            assert len({r['patient'] for r in rows}) == 6
            keys = {r['case'] for r in rows}
            assert keys == case_sets.setdefault(split, keys)
            for row in rows:
                m = np.array(row['confusion'])
                assert m.shape == (6, 6) and (m >= 0).all()
                assert m.sum() == np.prod(row['shape_xyz'])
                assert all(row[k] == inputs['hashes'][row['case']][k] for k in ('mri_sha256', 'mask_sha256'))
                if split == 'validation':
                    assert row['confusion'] == saved[row['case']]['confusion']
                    reproduced += 1
                elif model in ('Standard', 'v1'):
                    identity = 'standard_3d' if model == 'Standard' else 'improved_3d'
                    old = next(r for r in historical['arms'][identity]['cases'] if r['case'] == row['case'])
                    assert row['confusion'] == old['confusion']
    assert not case_sets['validation'] & case_sets['test']
    receipt = {'job': data['job'], 'complete': True, 'verified_files': len(checks),
               'metrics_sha256': sha(SOURCE/'metrics.json'), 'script_sha256': data['script_sha256'],
               'lock_sha256': data['lock_sha256'], 'models': MODELS,
               'model_volume_comparisons': 476, 'validation_matrices_reproduced_locally': reproduced,
               'historical_test_matrices_reproduced_locally': 60,
               'exact_cross_split_file_duplicates': 0,
               'test_status': 'Previously inspected; descriptive only; not an external or new blind test'}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/'integrity_verification.json', receipt)
    return data


def table(group, name, rows):
    dest = OUT/group/'tables'
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def save(group, name, fig):
    dest = OUT/group/'figures'
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(dest/f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    data = verify()
    classes, patients, cases, surfaces, surface_summary, regions, region_summary = [], [], [], [], [], [], []
    errors, resources, times, continuity, summaries = [], [], [], [], []
    for model, arm in data['arms'].items():
        b = arm['benchmark']
        resources.append({'model': model, 'epoch': arm['selected']['epoch'], 'gpu': data['gpu'],
                          'parameters': b['parameters'], 'reference_median_seconds': b['median_seconds'],
                          'reference_peak_allocated_GiB': b['peak_allocated_bytes']/2**30,
                          'reference_peak_reserved_GiB': b['peak_reserved_bytes']/2**30,
                          **{f'test_{key}': float(fn([r['inference_seconds_single_pass'] for r in arm['splits']['test']]))
                             for key, fn in [('median_seconds', np.median), ('p95_seconds', lambda x: np.percentile(x, 95))]},
                          'test_max_peak_allocated_GiB': max(r['inference_peak_allocated_bytes'] for r in arm['splits']['test'])/2**30})
        for split, rows in arm['splits'].items():
            grouped = {}
            for r in rows:
                m = np.asarray(r['confusion'])
                grouped.setdefault(r['patient'], np.zeros((6, 6), dtype=np.int64))
                grouped[r['patient']] += m
                ov = overlap(m)
                for j, label in enumerate(NAMES):
                    cases.append({'model': model, 'split': split, 'case': r['case'], 'patient': r['patient'], 'class': label,
                                  **{k: v[j] for k, v in ov.items()},
                                  'truth_voxels': int(m[j].sum()), 'prediction_voxels': int(m[:, j].sum())})
                for label, s in r['surfaces'].items():
                    surfaces.append({'model': model, 'split': split, 'case': r['case'], 'patient': r['patient'], 'class': label, **s})
                for region, value in (r['prostate_regions'] or {}).items():
                    regions.append({'model': model, 'split': split, 'case': r['case'], 'patient': r['patient'], 'region': region, 'dice': value})
                times.append({'model': model, 'split': split, 'case': r['case'],
                              'seconds_single_pass': r['inference_seconds_single_pass'],
                              'peak_allocated_GiB': r['inference_peak_allocated_bytes']/2**30})
                for label, g in r['geometry'].items():
                    continuity.append({'model': model, 'split': split, 'case': r['case'], 'patient': r['patient'], 'class': label,
                                       'truth_components': g['truth']['components_6'], 'prediction_components': g['prediction']['components_6'],
                                       'truth_internal_gap_slices': g['truth']['internal_empty_z_slices'],
                                       'prediction_internal_gap_slices': g['prediction']['internal_empty_z_slices'],
                                       'largest_prediction_component_fraction': g['prediction']['largest_component_fraction'],
                                       'truth_area_transition': g.get('area_transition', {}).get('truth'),
                                       'prediction_area_transition': g.get('area_transition', {}).get('prediction')})
            for patient, m in grouped.items():
                ov = overlap(m)
                patients.append({'model': model, 'split': split, 'patient': patient,
                                 **{label+'_dice': ov['dice'][j] for j, label in enumerate(NAMES)},
                                 'small_organ_dice': float(np.mean(ov['dice'][3:]))})
            pm = [overlap(m) for m in grouped.values()]
            pooled = overlap(sum(grouped.values()))
            for j, label in enumerate(NAMES):
                classes.append({'model': model, 'split': split, 'class': label,
                                **{'patient_equal_'+k: float(np.mean([s[k][j] for s in pm])) for k in pooled},
                                **{'voxel_pooled_'+k: v[j] for k, v in pooled.items()}})
            summaries.append({'model': model, 'split': split, 'epoch': arm['selected']['epoch'],
                              'small_organ_dice': float(np.mean([np.mean(s['dice'][3:]) for s in pm])),
                              'prostate_dice': float(np.mean([s['dice'][5] for s in pm])),
                              'foreground_macro_dice': float(np.mean([np.mean(s['dice'][1:]) for s in pm])),
                              'six_class_macro_dice': float(np.mean([np.mean(s['dice']) for s in pm]))})
            for label in NAMES[1:]:
                ss = [s for s in surfaces if s['model'] == model and s['split'] == split and s['class'] == label]
                valid = [s for s in ss if s['status'] == 'both_present']
                means = {p: np.mean([s['hd95_mm'] for s in valid if s['patient'] == p]) for p in grouped}
                surface_summary.append({'model': model, 'split': split, 'class': label,
                                        'volume_median_hd95_mm': float(np.median([s['hd95_mm'] for s in valid])),
                                        'volume_p95_hd95_mm': float(np.percentile([s['hd95_mm'] for s in valid], 95)),
                                        'volume_max_hd95_mm': max(s['hd95_mm'] for s in valid),
                                        'patient_equal_mean_hd95_mm': float(np.mean(list(means.values()))),
                                        'patient_equal_mean_assd_mm': float(np.mean([np.mean([s['assd_mm'] for s in valid if s['patient'] == p]) for p in grouped])),
                                        'both_present': len(valid), 'one_empty': sum(s['status'] == 'one_empty' for s in ss),
                                        'both_empty': sum(s['status'] == 'both_empty' for s in ss)})
            for region in ('apex', 'central', 'base'):
                rr = [r for r in regions if r['model'] == model and r['split'] == split and r['region'] == region]
                region_summary.append({'model': model, 'split': split, 'region': region,
                                       'patient_equal_slice_mean_dice': float(np.mean([np.mean([r['dice'] for r in rr if r['patient'] == p]) for p in grouped])),
                                       'available_volumes': len(rr)})
            matrix = sum(grouped.values())
            errors.append({'model': model, 'split': split,
                           'far_background_to_body_voxels': sum(r['far_background_to_body_voxels'] for r in rows),
                           'far_body_fp_per_100000_background_voxels': 100000*sum(r['far_background_to_body_voxels'] for r in rows)/sum(r['background_voxels'] for r in rows),
                           'prostate_predictions_in_true_bladder': int(matrix[3, 5]),
                           'prostate_predictions_in_true_rectum': int(matrix[4, 5]),
                           'prostate_in_OAR_per_100000_predicted_prostate': 100000*int(matrix[3, 5]+matrix[4, 5])/int(matrix[:, 5].sum()),
                           'missed_true_prostate_fraction': float(1-matrix[5, 5]/matrix[5].sum()),
                           'mean_missed_GT_positive_slices_per_volume': float(np.mean([r['prostate_extent']['GT_positive_slices_without_any_predicted_prostate'] for r in rows])),
                           'mean_predicted_slices_outside_GT_extent': float(np.mean([r['prostate_extent']['predicted_slices_outside_GT_extent'] for r in rows]))})
    paired = []
    for split in ('validation', 'test'):
        by = {m: {r['patient']: r['small_organ_dice'] for r in patients if r['model'] == m and r['split'] == split} for m in MODELS}
        for model, ref in [(m, 'Standard') for m in MODELS if m != 'Standard'] + [('A1', 'A0'), ('A0', 'v3'), ('A1', 'v3')]:
            pp = sorted(by[ref]); diffs = np.array([by[model][p]-by[ref][p] for p in pp])
            loo = [(diffs.sum()-v)/(len(diffs)-1) for v in diffs]
            paired.append({'split': split, 'model': model, 'reference': ref, 'mean_difference': float(diffs.mean()),
                           'patients_improved': int((diffs > 0).sum()), 'patients_worse': int((diffs < 0).sum()),
                           'leave_one_patient_out_min_difference': float(min(loo)), 'leave_one_patient_out_max_difference': float(max(loo)),
                           'difference_without_M015_descriptive_only': float(np.mean([by[model][p]-by[ref][p] for p in pp if p != 'M015'])) if split == 'test' else ''})
    required = 'assessment_required'; extra = 'supplementary_research'
    for name, rows in [('six_class_metrics', classes), ('model_summary', summaries), ('surface_summary', surface_summary),
                       ('prostate_regions', region_summary), ('resources', resources), ('per_case_inference', times), ('boundary_errors', errors)]:
        table(required, name+'.csv', rows)
    for name, rows in [('per_patient', patients), ('per_case_class_metrics', cases), ('surface_distances', surfaces),
                       ('per_case_regions', regions), ('continuity', continuity), ('paired_patient_sensitivity', paired)]:
        table(extra, name+'.csv', rows)
    # Heatmaps keep seven models legible without overlapping grouped bars.
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout='constrained')
    for row, split in enumerate(('validation', 'test')):
        for col, metric in enumerate(('dice', 'iou')):
            ax = axes[row, col]
            values = np.array([[next(r['patient_equal_'+metric] for r in classes if r['model'] == m and r['split'] == split and r['class'] == c) for c in NAMES] for m in MODELS])
            ax.imshow(values, vmin=.5, vmax=1, cmap='YlGnBu')
            for i in range(7):
                for j in range(6):ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', color='white' if values[i,j] > .85 else 'black', fontsize=9)
            ax.set_xticks(range(6), NAMES, rotation=25); ax.set_yticks(range(7), MODELS)
            ax.set_title(split.title()+' patient-equal '+metric)
    fig.suptitle('Each model at its Validation-selected best | Test is previously inspected, descriptive')
    save(required, 'six_class_validation_test', fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, key, title in zip(axes, ('small_organ_dice', 'prostate_dice'), ('Three small organs', 'Prostate only')):
        for i, split in enumerate(('validation', 'test')):
            values = [next(r[key] for r in summaries if r['model'] == m and r['split'] == split) for m in MODELS]
            ax.bar(np.arange(7)+(i-.5)*.36, values, .36, label=split)
        ax.set_xticks(range(7), MODELS); ax.set(ylim=(0,1), title=title, ylabel='Patient-equal Dice'); ax.legend()
    save(required, 'validation_test_generalization', fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, split in zip(axes, ('validation', 'test')):
        for m, color in zip(MODELS, COLORS):
            ax.plot(['apex', 'central', 'base'], [next(r['patient_equal_slice_mean_dice'] for r in region_summary if r['model'] == m and r['split'] == split and r['region'] == z) for z in ('apex','central','base')], 'o-', color=color, label=m)
        ax.set(title=split.title()+' prostate regions', ylim=(0,1), ylabel='Equal patients; mean visits and selected slices')
    axes[1].legend(ncol=2, fontsize=9); save(required, 'prostate_base_apex', fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, split in zip(axes, ('validation', 'test')):
        for i, m in enumerate(MODELS):
            ss = [s['hd95_mm'] for s in surfaces if s['model']==m and s['split']==split and s['class']=='Body' and s['status']=='both_present']
            ax.boxplot(ss, positions=[i], widths=.5, whis=(0,100), showfliers=False)
        ax.set_xticks(range(7),MODELS);ax.set(title=split.title()+' Body HD95 (full min–max whiskers)', ylabel='mm')
    save(required, 'body_boundary_distribution', fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for ax, key, title in zip(axes, ('parameters','reference_peak_allocated_GiB','test_median_seconds'), ('Parameters','Reference peak allocated VRAM / GiB','Test median single-pass inference / s')):
        ax.bar(MODELS,[r[key] for r in resources],color=COLORS);ax.set_title(title);ax.tick_params(axis='x',rotation=25)
    save(required, 'resource_profile', fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, split in zip(axes, ('validation', 'test')):
        pp=sorted({r['patient'] for r in patients if r['split']==split})
        for m,color in zip(MODELS,COLORS):
            ax.plot(pp,[next(r['small_organ_dice'] for r in patients if r['model']==m and r['split']==split and r['patient']==p) for p in pp],'o-',label=m,color=color)
        ax.set(title=split.title()+' patient-level small-organ Dice',ylim=(0,1));ax.tick_params(axis='x',rotation=25)
    axes[1].legend(ncol=2,fontsize=9);save(extra,'patient_variability',fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax,split in zip(axes,('validation','test')):
        values=[next(r['far_background_to_body_voxels'] for r in errors if r['model']==m and r['split']==split) for m in MODELS]
        ax.bar(MODELS,values,color=COLORS);ax.set_yscale('log');ax.set(title=split.title()+' distant Body FP',ylabel='Voxel count (log scale; >=20 mm)')
        for i,v in enumerate(values):ax.text(i,v*1.1,f'{v:,}',ha='center',fontsize=8)
    save(extra,'distant_body_false_positives',fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, split in zip(axes, ('validation', 'test')):
        for i, m in enumerate(MODELS):
            values = [s['hd95_mm'] for s in surfaces if s['model']==m and s['split']==split and s['class']=='Prostate' and s['status']=='both_present']
            ax.boxplot(values, positions=[i], widths=.5, whis=(0,100), showfliers=False)
        ax.set_xticks(range(7),MODELS)
        ax.set(title=split.title()+' Prostate HD95 (full min–max)',ylabel='mm; includes distant tails')
    save(required,'prostate_boundary_tails',fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, split in zip(axes, ('validation', 'test')):
        for m, color in zip(MODELS, COLORS):
            r = next(r for r in errors if r['model']==m and r['split']==split)
            x = 100*r['missed_true_prostate_fraction']
            y = r['prostate_in_OAR_per_100000_predicted_prostate']/1000
            ax.scatter(x,y,color=color,s=55)
            ax.annotate(m,(x,y),xytext=(4,4),textcoords='offset points',fontsize=9)
        ax.set(title=split.title()+' Prostate miss/OAR-confusion trade-off',
               xlabel='Missed true prostate voxels / % (pooled)',
               ylabel='Prostate predictions inside true Bladder/Rectum / %')
        ax.grid(alpha=.2)
    save(required,'prostate_oar_tradeoff',fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    for col, split in enumerate(('validation','test')):
        for row, field, title in [(0,'prediction_components','Volumes with >1 prostate component'),
                                  (1,'prediction_internal_gap_slices','Internal predicted prostate gap slices (sum)')]:
            values=[]
            for m in MODELS:
                rs=[r for r in continuity if r['model']==m and r['split']==split and r['class']=='Prostate']
                values.append(sum(r[field]>1 for r in rs) if row==0 else sum(r[field] for r in rs))
            axes[row,col].bar(MODELS,values,color=COLORS)
            axes[row,col].set_title(split.title()+': '+title)
    fig.suptitle('Descriptive continuity proxies: disconnected distant FP can also create gaps')
    save(extra,'prostate_continuity',fig)
    coverage = {'verified': read(OUT/'integrity_verification.json'), 'surface_status': dict(Counter(r['status'] for r in surfaces)),
                'prostate_region_observations': len(regions), 'models': MODELS,
                'test_is_fresh_blind': False, 'one_seed_per_recipe': True,
                'anatomical_autopsy_requires_manual_image_review': True,
                'clinical_dose_or_safety_evidence': False}
    write(OUT/'coverage.json',coverage)
    print(json.dumps({'summary':summaries,'paired':paired,'coverage':coverage},indent=2))


if __name__ == '__main__':
    main()
