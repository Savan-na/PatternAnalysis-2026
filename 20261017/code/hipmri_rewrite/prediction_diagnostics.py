"""Isolated fixed-weight diagnostics; no change to frozen G5 model/inference.

Replay uses a case's image-derived full-volume normalization statistics. It is
a counterfactual mechanism probe, not a trained or deployable replacement.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import numpy as np
import torch
from scipy import ndimage as ndi


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def capture_instance_stats(model):
    """Capture biased per-channel input variance, matching InstanceNorm.

    Each module must be called once, batch size one, native eval behaviour.
    No hooks, parameters or running buffers survive this context.
    """
    bank, handles = {}, []
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.InstanceNorm3d)}
    if not modules:
        raise ValueError("No InstanceNorm3d modules")
    def hook(name):
        def capture(module, args):
            x = args[0]
            if name in bank or x.shape[0] != 1 or module.training or module.track_running_stats:
                raise ValueError("Expected one eval pass with native input statistics")
            variance, mean = torch.var_mean(x.detach(), dim=(0, 2, 3, 4), correction=0)
            bank[name] = {"mean": mean, "variance": variance, "eps": module.eps,
                          "shape_dhw": list(x.shape[2:])}
        return capture
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(hook(name)))
        yield bank
        if set(bank) != set(modules):
            raise ValueError("Not all normalization modules were observed")
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def replay_instance_stats(model, bank):
    """Override only normalization outputs; weights and input patches unchanged."""
    modules = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.InstanceNorm3d)}
    if set(modules) != set(bank):
        raise ValueError("Captured normalization modules differ")
    handles = []
    def hook(name):
        def replay(module, args, native_output):
            x, record = args[0], bank[name]
            if module.training or module.eps != record["eps"]:
                raise ValueError("Normalization mode or epsilon changed")
            mean = record["mean"].to(device=x.device, dtype=x.dtype)[None, :, None, None, None]
            var = record["variance"].to(device=x.device, dtype=x.dtype)[None, :, None, None, None]
            y = (x-mean) * torch.rsqrt(var + module.eps)
            if module.affine:
                y = y * module.weight[None, :, None, None, None] + module.bias[None, :, None, None, None]
            return y
        return replay
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return {"n": 0, **dict.fromkeys(("min", "p05", "median", "p95", "max"))}
    q = np.percentile(values, (0, 5, 50, 95, 100))
    return {"n": int(values.size), **dict(zip(("min", "p05", "median", "p95", "max"), map(float, q)))}


def confidence(logits, regions):
    """Six-way uncalibrated softmax plus margins; bins are descriptive only."""
    shifted = logits.astype(np.float64)-logits.max(axis=0, keepdims=True)
    exponential = np.exp(shifted)
    probability = exponential / exponential.sum(axis=0, keepdims=True)
    margin = logits[1]-logits[0]
    competitor = logits[1]-logits[[0, 2, 3, 4, 5]].max(axis=0)
    prediction = logits.argmax(0)
    disagreement = prediction != probability.argmax(0)
    ordered = np.partition(logits, -2, axis=0)
    top_gap = ordered[-1]-ordered[-2]
    if (disagreement & (top_gap > 1e-6)).any():
        raise ValueError("Softmax/argmax identity failed away from numerical ties")
    return {name: {"voxels": int(region.sum()),
                   "softmax_argmax_near_tie_disagreements": int(disagreement[region].sum()),
                   "body_probability": quantiles(probability[1][region]),
                   "background_probability": quantiles(probability[0][region]),
                   "body_minus_background": quantiles(margin[region]),
                   "body_minus_best_other": quantiles(competitor[region]),
                   "body_probability_ge_0p9": int((probability[1][region] >= .9).sum()),
                   "body_winner_margin_0_to_0p25": int(((competitor[region] >= 0) & (competitor[region] <= .25)).sum()),
                   "body_winner_margin_gt_2": int((competitor[region] > 2).sum())}
            for name, region in regions.items()}


def validate_grid(affine, spacing):
    axes = np.asarray(affine)[:3, :3]
    norms = np.linalg.norm(axes, axis=0)
    if not np.allclose(norms, spacing, atol=1e-4, rtol=1e-5):
        raise ValueError("Affine axis lengths differ from spacing")
    if not np.allclose((axes/norms).T @ (axes/norms), np.eye(3), atol=1e-5):
        raise ValueError("Sheared grid needs a full-affine distance method")


def geometry(region, distance_to_body, spacing, affine, window_origins=None):
    """Record physical extents and border bins without retaining voxel arrays.

    Bounds are outer voxel faces. Sliding-edge distances refer to every
    contributing window, not a presumed stitched seam in the historical model.
    """
    validate_grid(affine, spacing)
    points = np.argwhere(region)
    if not len(points):
        return {"voxels": 0, "component_count_6": 0, "components": []}
    shape = np.asarray(region.shape)
    scale = np.asarray(spacing)
    normalized = (points+.5)/shape
    face_distance = np.minimum((points+.5)*scale, (shape-points-.5)*scale).min(axis=1)
    world = points @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
    bins = (0, 5, 10, 20, 50, np.inf)
    ids, count = ndi.label(region)
    sizes = np.bincount(ids.ravel())[1:]
    boxes = ndi.find_objects(ids)
    largest = np.argsort(sizes)[::-1][:10]
    result = {"voxels": int(len(points)), "volume_mm3": float(len(points)*np.prod(scale)),
              "distance_to_gt_body_voxel_centres_mm": quantiles(distance_to_body[region]),
              "distance_to_original_fov_faces_mm": quantiles(face_distance),
              "fov_face_distance_bins_0_5_10_20_50_inf_mm": np.histogram(face_distance, bins)[0].tolist(),
              "normalized_xyz_quantiles": [quantiles(normalized[:, i]) for i in range(3)],
              "world_bbox_min_mm": world.min(axis=0).tolist(), "world_bbox_max_mm": world.max(axis=0).tolist(),
              "component_count_6": int(count), "component_sizes_voxels": quantiles(sizes),
              "largest_10_components": []}
    for i in largest:
        box = boxes[i]
        result["largest_10_components"].append({"voxels": int(sizes[i]),
             "volume_mm3": float(sizes[i]*np.prod(scale)),
             "bbox_min_xyz": [s.start for s in box], "bbox_max_xyz_inclusive": [s.stop-1 for s in box],
             "bbox_extent_mm": [(s.stop-s.start)*float(v) for s, v in zip(box, scale)]})
    if window_origins is not None:
        # Known fixed training-size windows; coordinates are XYZ, not DHW.
        nearest = np.full(len(points), np.inf)
        edge_pairs = total_pairs = 0
        for origin in window_origins:
            start = np.asarray(origin)
            window = np.asarray((64, 64, 32))
            inside = ((points >= start) & (points < start+window)).all(axis=1)
            delta = points[inside]-start
            distances = np.minimum((delta+.5)*scale, (window-delta-.5)*scale).min(axis=1)
            nearest[inside] = np.minimum(nearest[inside], distances)
            total_pairs += len(distances)
            edge_pairs += int((distances <= 5).sum())
        if not np.isfinite(nearest).all():
            raise ValueError("False-positive voxel not covered by a window")
        result["nearest_contributing_window_face_mm"] = quantiles(nearest)
        result["contributing_window_voxel_pairs_within_5mm"] = edge_pairs
        result["contributing_window_voxel_pairs"] = total_pairs
    return result


def conservative_component_filter(logits, spacing):
    """One predefined diagnostic: remove only tiny AND relatively tiny Body CCs.

    Six-neighbour components <100 mm3 AND <1% of the largest predicted Body
    component are reassigned to their best non-Body logit. No GT is accepted.
    This can remove real anatomy and must not become a default automatically.
    """
    prediction = logits.argmax(0).astype(np.uint8)
    ids, count = ndi.label(prediction == 1)
    sizes = np.bincount(ids.ravel())
    if not count:
        return prediction, {"removed_voxels": 0, "removed_components": 0}
    largest = sizes[1:].max()
    removed_ids = np.flatnonzero((sizes < .01*largest) & (sizes*np.prod(spacing) < 100))
    removed_ids = removed_ids[removed_ids != 0]
    removed = np.isin(ids, removed_ids)
    alternatives = np.asarray((0, 2, 3, 4, 5))
    prediction[removed] = alternatives[logits[alternatives][:, removed].argmax(0)]
    return prediction, {"removed_voxels": int(removed.sum()), "removed_components": int(len(removed_ids)),
                        "rule": "6-connected Body CC <100 mm3 AND <1% of largest; best non-Body logit"}
