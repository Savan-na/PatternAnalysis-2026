"""Separate P2 diagnostic inference; frozen G5 inference remains unchanged.

Inputs are image-only, already volume-normalized DHW arrays. Uniform averaging
of raw logits is fixed before evaluation. No label-based ROI or postprocessing.
"""
from __future__ import annotations

from itertools import product
import numpy as np
import torch


def axis_starts(length: int, window: int) -> tuple[int, ...]:
    if length < window or window < 16 or window % 16:
        raise ValueError("Padded length must cover a window divisible by 16")
    starts = list(range(0, length - window + 1, window // 2))
    if starts[-1] != length - window:
        starts.append(length - window)
    return tuple(starts)


@torch.inference_mode()
def infer_logits(model, image_dhw: np.ndarray, *, device: str,
                 window_dhw: tuple[int, int, int] | None = None,
                 components: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """Return CPU logits and coverage receipt on the supplied padded DHW grid.

    A single full window reproduces ordinary full-volume inference. Windows
    have fixed half-window strides; a final anchored window covers each edge.
    Accumulation uses float32. Head maps use the same weights as combined maps.
    """
    if image_dhw.ndim != 3 or not np.isfinite(image_dhw).all():
        raise ValueError("Expected a finite DHW image")
    if any(n < 16 or n % 16 for n in image_dhw.shape):
        raise ValueError("Use the existing high-end multiple-of-16 padding first")
    window = tuple(image_dhw.shape) if window_dhw is None else tuple(window_dhw)
    if len(window) != 3 or any(type(n) is not int or n < 16 or n % 16 for n in window):
        raise ValueError("Window dimensions must be integer multiples of 16")
    padded_shape = tuple(max(n, w) for n, w in zip(image_dhw.shape, window))
    image = np.pad(image_dhw, tuple((0, m-n) for n, m in zip(image_dhw.shape, padded_shape)))
    axes = [axis_starts(n, w) for n, w in zip(padded_shape, window)]
    names = ("combined", "main", "half", "quarter") if components else ("combined",)
    totals = {name: np.zeros((6, *padded_shape), dtype=np.float32) for name in names}
    counts = np.zeros(padded_shape, dtype=np.uint16)
    model.eval()
    count = 0
    for origin in product(*axes):
        region = tuple(slice(s, s+w) for s, w in zip(origin, window))
        tensor = torch.from_numpy(np.ascontiguousarray(image[region]))[None, None].to(device)
        if components:
            combined, heads = model(tensor, return_components=True)
            if len(heads) != 3 or not torch.allclose(combined, sum(heads), atol=1e-6):
                raise ValueError("Three-head additive identity changed")
            outputs = (combined, *heads)
        else:
            outputs = (model(tensor),)
        for name, logits in zip(names, outputs):
            if logits.shape != (1, 6, *window) or not torch.isfinite(logits).all():
                raise ValueError("Unexpected or nonfinite logits")
            totals[name][(slice(None), *region)] += logits[0].cpu().numpy()
        counts[region] += 1
        count += 1
    if not counts.all():
        raise ValueError("Sliding-window grid left uncovered voxels")
    crop = tuple(slice(0, n) for n in image_dhw.shape)
    result = {name: (value / counts[None])[(slice(None), *crop)] for name, value in totals.items()}
    return result, {"window_dhw": list(window), "padded_shape_dhw": list(padded_shape),
                    "windows": count, "axis_starts_dhw": [list(a) for a in axes],
                    "coverage_min": int(counts[crop].min()),
                    "coverage_max": int(counts[crop].max()),
                    "fusion": "uniform arithmetic mean of raw logits; no TTA",
                    "normalization": "unchanged per-volume nonzero Z-score; no per-window renormalization"}
