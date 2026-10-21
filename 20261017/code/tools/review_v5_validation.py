"""Replay four Validation cases for C50/L45/S45 and draw shared-grid evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from hipmri_rewrite.data import load_case, downsample, zscore_nonzero
from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3
from hipmri_rewrite.inference import predict_volume
from hipmri_rewrite.metrics import confusion, overlap
from hipmri_rewrite.splits import discover

NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']
EPOCHS = {'C': 50, 'L': 45, 'S': 45}
HASHES = {'C': '72f8dae5a36f00779d01f58e224c02a2e7e01a0d1ae304e405bf6a96728bd791',
          'L': '91cdfd9e0fe2998948b513c1db14ac4ad782f659d3ab880d6571f98e974ed56c',
          'S': 'c6895035ab4dc0c89a345fa9527e4d6bd903e32da9350d2b6aeccdd61449583b'}
CASES = {'S022_Week6': [4, 3], 'J026_Week6': [5], 'K018_Week4': [1], 'K018_Week2': [5]}
REASONS = {'S022_Week6': 'L maximum Rectum HD95 and largest Bladder Dice gain versus C',
           'J026_Week6': 'L maximum Prostate HD95', 'K018_Week4': 'L maximum distant Body FP',
           'K018_Week2': 'S largest Prostate Dice gain versus C'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paint(ax, image, truth, found, limits, aspect):
    ax.imshow(image.T, cmap='gray', origin='lower', vmin=limits[0], vmax=limits[1], aspect=aspect)
    for mask, color in [(truth.T, 'cyan'), (found.T, 'orange')]:
        if mask.any() and not mask.all():
            ax.contour(mask.astype(float), [.5], colors=[color], linewidths=.8)
    error = np.zeros((*truth.T.shape, 4))
    error[(found & ~truth).T] = [1, 0, 0, .45]
    error[(truth & ~found).T] = [0, .3, 1, .45]
    ax.imshow(error, origin='lower', aspect=aspect)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside an allocated GPU job')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    out = ROOT/'results/experiments/improved_3d_v5/case_studies/validation/best_cls_review'
    if out.exists():
        raise ValueError('Output exists; inspect before any rerun')
    # Verify the original training package and shared dependencies without modifying them.
    frozen = json.loads((ROOT/'code/config/v5_transfer_checksums.json').read_text())
    for relative, expected in frozen.items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Frozen source/reference mismatch: {relative}')
    models, records, provenance = {}, {}, {}
    for arm, epoch in EPOCHS.items():
        run = ROOT/f'logs/training/improved_3d_v5_{arm.lower()}_20261021'
        digest = hashlib.sha256(''.join(p.name+' '+sha(p)+'\n' for p in sorted(run.glob('*.json'))).encode()).hexdigest()
        if digest != HASHES[arm]:
            raise ValueError(f'{arm}: records differ from verified analysis')
        manifest = json.loads((run/'manifest.json').read_text())
        record = json.loads((run/f'epoch_{epoch:03d}.json').read_text())
        final = json.loads((run/'epoch_050.json').read_text())
        assert final['best_epoch'] == epoch and final['best_score'] == record['validation']['selection_score']
        checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == epoch and checkpoint['model'] == f'improved_3d_v5_{arm.lower()}'
        assert checkpoint['config_sha256'] == manifest['config_sha256']
        assert str(torch.__version__) == manifest['torch']
        model = ImprovedUNet3DV3().cuda().eval()
        model.load_state_dict(checkpoint['model_state'], strict=True)
        models[arm] = model
        records[arm] = {c['case']: c for c in record['validation']['cases']}
        provenance[arm] = dict(epoch=epoch, checkpoint_sha256=sha(run/'best.pt'),
                               record_sha256=sha(run/f'epoch_{epoch:03d}.json'), aggregate_records_sha256=digest)
        del checkpoint
    cases = {c.identifier: c for c in discover(args.dataset_root)['validation']}
    assert all(c in cases for c in CASES)
    out.mkdir(parents=True)
    receipt = dict(job_id=os.environ['SLURM_JOB_ID'], script_sha256=sha(Path(__file__)),
                   checkpoints=provenance, scope='Validation only; selected contrasting cases, not unbiased prevalence',
                   cases=[])
    for case_id, focused_labels in CASES.items():
        case = cases[case_id]
        volume = zscore_nonzero(downsample(load_case(case)))
        predictions = {}
        for arm, model in models.items():
            pred = predict_volume(model, volume, mode='3d', device='cuda')
            if confusion(volume.mask, pred).tolist() != records[arm][case_id]['confusion']:
                raise ValueError(f'{case_id}/{arm}: original confusion not reproduced; stop interpretation')
            predictions[arm] = pred
        limits = np.percentile(volume.image[volume.image != 0], [1, 99])
        spacing = volume.spacing_mm
        fig, axes = plt.subplots(3, 6, figsize=(22, 11), layout='constrained')
        for label, name in enumerate(NAMES):
            z = int(np.argmax((volume.mask == label).sum(axis=(0, 1))))
            for row, arm in enumerate(EPOCHS):
                metrics = overlap(np.asarray(records[arm][case_id]['confusion']))
                paint(axes[row, label], volume.image[:, :, z], volume.mask[:, :, z] == label,
                      predictions[arm][:, :, z] == label, limits, spacing[1]/spacing[0])
                axes[row, label].set_title(f'{arm} epoch {EPOCHS[arm]} | {name}\n3D Dice={metrics["dice"][label]:.3f} | XY z={z}', fontsize=10)
        fig.suptitle(f'Validation {case_id} | six classes | common GT maximum-area XY slice per class', fontsize=14)
        fig.legend([Line2D([0], [0], color=c) for c in ['cyan', 'orange', 'red', 'blue']],
                   ['GT contour', 'Prediction contour', 'False positive', 'False negative'],
                   loc='outside lower center', ncol=4)
        fig.savefig(out/f'{case_id}_six_classes.png', dpi=140)
        plt.close(fig)
        planes = []
        for label in focused_labels:
            if label == 1:
                fp = (volume.mask == 0) & np.logical_or.reduce([p == 1 for p in predictions.values()])
                distance = distance_transform_edt(volume.mask != 1, sampling=spacing)
                center = np.unravel_index(np.argmax(np.where(fp, distance, -1)), fp.shape) if fp.any() else tuple(n//2 for n in fp.shape)
                rule = 'most distant Background-to-Body FP among all three models'
            else:
                z = int(np.argmax((volume.mask == label).sum(axis=(0, 1))))
                xy = np.argwhere(volume.mask[:, :, z] == label)
                x, y = np.rint(np.mean(xy, axis=0)).astype(int)
                center = (x, y, z)
                rule = 'GT maximum-area XY slice and in-slice GT centroid; descriptive display only'
            x, y, z = map(int, center)
            selectors = [(slice(None), slice(None), z), (slice(None), y, slice(None)), (x, slice(None), slice(None))]
            aspects = [spacing[1]/spacing[0], spacing[2]/spacing[0], spacing[2]/spacing[1]]
            fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout='constrained')
            for row, arm in enumerate(EPOCHS):
                for col, (selection, aspect, name) in enumerate(zip(selectors, aspects, ['XY', 'XZ', 'YZ'])):
                    paint(axes[row, col], volume.image[selection], volume.mask[selection] == label,
                          predictions[arm][selection] == label, limits, aspect)
                    axes[row, col].set_title(f'{arm} epoch {EPOCHS[arm]} | {name}')
            fig.suptitle(f'Validation {case_id} | {NAMES[label]} | shared voxel XYZ=({x},{y},{z})\nGT cyan / prediction orange / FP red / FN blue')
            fig.savefig(out/f'{case_id}_{NAMES[label]}_three_planes.png', dpi=150)
            plt.close(fig)
            planes.append(dict(label=NAMES[label], center_xyz=[x, y, z], selection_rule=rule))
        receipt['cases'].append(dict(case=case_id, selection_reason=REASONS[case_id], planes=planes,
                                      mri_sha256=sha(case.mri), mask_sha256=sha(case.mask),
                                      all_three_confusions_reproduced=True))
        print(f'{case_id}: all three best-checkpoint matrices reproduced; panels saved', flush=True)
    (out/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (out/'checksums.json').write_text(json.dumps({p.name: sha(p) for p in sorted(out.glob('*.png'))}, indent=2)+'\n')
    (out/'README.md').write_text('# v5 Validation best-checkpoint review\n\nC50 / L45 / S45, same cases and planes, six classes plus focused three-plane views. Selected from recorded Validation errors and gains; not representative prevalence. Every original case confusion matrix is exactly reproduced before plotting. GT cyan, prediction orange, FP red, FN blue. XY/XZ/YZ refer to NIfTI voxel axes, not independently verified anatomical orientation. Physical spacing determines image aspect; coordinates are indices. GT only selects display planes, never inference regions or post-processing. No Test arrays are decoded.\n')
    print(out, flush=True)


if __name__ == '__main__':
    main()
