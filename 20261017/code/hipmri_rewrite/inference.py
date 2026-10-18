"""Common-grid 2D/3D inference without changing the source NIfTI files."""

from __future__ import annotations

import numpy as np
import torch

from .data import Volume, from_3d_model_prediction, to_3d_model_arrays


@torch.inference_mode()
def predict_volume(model: torch.nn.Module, volume: Volume, *, mode: str,
                   device: torch.device | str, slice_batch: int = 8) -> np.ndarray:
    """Return XYZ integer labels on the supplied volume's physical grid.

    2D and 3D arms use the same prepared MRI, padding and inverse mapping.
    Peak VRAM and latency are measured outside this function so preprocessing
    and model time can be reported separately.
    """
    if mode not in ("2d", "3d") or slice_batch < 1:
        raise ValueError("mode must be 2d/3d and slice_batch positive")
    image, _, original_shape = to_3d_model_arrays(volume)
    model.eval()
    if mode == "3d":
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(device)
        logits = model(tensor)
        if logits.shape != (1, 6, *image.shape):
            raise ValueError("3D model returned an unexpected logit shape")
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    else:
        predictions = []
        for start in range(0, image.shape[0], slice_batch):
            batch = np.ascontiguousarray(image[start:start + slice_batch, None])
            logits = model(torch.from_numpy(batch).to(device))
            if logits.shape != (len(batch), 6, *image.shape[1:]):
                raise ValueError("2D model returned an unexpected logit shape")
            predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
        prediction = np.concatenate(predictions, axis=0)
    return from_3d_model_prediction(prediction, original_shape)
