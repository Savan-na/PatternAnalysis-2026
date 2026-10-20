"""Stateless Train-only physical-space paired affine patches and cosine schedule."""
import hashlib
import math
import numpy as np
from scipy.ndimage import affine_transform
from .sampling import PATCH_XYZ, crop_for_3d_model


def learning_rate(completed_updates, total=6500):
    if not 0 <= completed_updates <= total:
        raise ValueError('Update outside frozen schedule')
    return 1e-5 + .5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * completed_updates / total))


def make_optimizer(model):
    import torch
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith('.bias') else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': 1e-5},
                              {'params': no_decay, 'weight_decay': 0.0}], lr=1e-3)


def paired_patch(volume, origin, *, seed, epoch, draw, enabled, force=None):
    if volume.case.split != 'train':
        raise ValueError('Augmentation accepts Train only')
    key = f'{seed}|{epoch}|{volume.case.identifier}|{draw}|v4-affine'.encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
    active = bool(rng.random() < .3) if enabled else False
    if force is not None:
        active = True
    before_image, before_mask = crop_for_3d_model(volume, origin)
    record = {'case': volume.case.identifier, 'draw': draw, 'origin_xyz': list(origin),
              'applied': active, 'before_counts': np.bincount(before_mask.ravel(), minlength=6).tolist()}
    if not active:
        record.update(after_counts=record['before_counts'], outside_fraction=0.0,
                      angles_deg=[0., 0., 0.], scale=1.)
        return before_image, before_mask, record
    angles, scale = (rng.uniform(-10, 10, 3), float(rng.uniform(.9, 1.1))) if force is None else force
    a, b, c = np.deg2rad(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    basis = np.asarray(volume.affine[:3, :3], dtype=np.float64)
    if not np.isfinite(basis).all() or abs(np.linalg.det(basis)) < 1e-12:
        raise ValueError('Invalid physical affine')
    # World-coordinate rotation about crop centre; inverse mapping output -> source.
    matrix = np.linalg.solve(basis, (rx @ ry @ rz).T @ basis / scale)
    centre = (np.asarray(PATCH_XYZ) - 1) / 2
    offset = np.asarray(origin) + centre - matrix @ centre
    axes = np.indices(PATCH_XYZ, dtype=np.float64).reshape(3, -1)
    mapped = matrix @ axes + offset[:, None]
    outside = np.any((mapped < 0) | (mapped > (np.asarray(volume.mask.shape) - 1)[:, None]), axis=0)
    image = affine_transform(volume.image, matrix, offset, output_shape=PATCH_XYZ,
                             order=1, mode='constant', cval=0, prefilter=False)
    mask = affine_transform(volume.mask, matrix, offset, output_shape=PATCH_XYZ,
                            order=0, mode='constant', cval=0, prefilter=False)
    image = np.ascontiguousarray(image.transpose(2, 1, 0)[None], dtype=np.float32)
    mask = np.ascontiguousarray(mask.transpose(2, 1, 0), dtype=np.int64)
    if not np.isfinite(image).all() or np.any((mask < 0) | (mask > 5)):
        raise ValueError('Invalid transformed patch')
    record.update(after_counts=np.bincount(mask.ravel(), minlength=6).tolist(),
                  outside_fraction=float(outside.mean()), angles_deg=list(map(float, angles)), scale=float(scale))
    return image, mask, record
