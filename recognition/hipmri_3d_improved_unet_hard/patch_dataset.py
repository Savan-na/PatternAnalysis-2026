"""On-demand, aligned 3D HipMRI patches for the approved train/validation split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import APPROVED_PATIENTS, DEFAULT_DATASET_ROOT, VolumePair, discover_dataset
from .intensity_normalization import normalize_volume_pair
from .resampling import resample_volume_pair
from .volume_io import load_volume_pair


def _config() -> dict[str, object]:
    settings = json.loads(Path(__file__).with_name("patch_dataset_config.json").read_text(encoding="utf-8"))
    expected = {
        "patch_shape_dhw", "train_patches_per_volume", "train_foreground_probability",
        "foreground_label", "seed", "validation_crop",
    }
    if set(settings) != expected or settings["foreground_label"] != 5 or settings["validation_crop"] != "center":
        raise ValueError("patch_dataset_config.json has unsupported settings.")
    return settings


def _patch_shape(shape: object) -> tuple[int, int, int]:
    if not isinstance(shape, (tuple, list)) or len(shape) != 3 or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 16
        for size in shape
    ):
        raise ValueError("patch_shape_dhw must be three positive integers divisible by 16.")
    return tuple(shape)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


class HipMRIPatchDataset(Dataset):
    """One patch per index; no processed full-volume cache is held on self."""

    def __init__(
        self,
        split: str,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        *,
        patch_shape_dhw: tuple[int, int, int] | None = None,
        train_patches_per_volume: int | None = None,
        train_foreground_probability: float | None = None,
        seed: int | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'; test patients are not exposed.")
        settings = _config()
        self.patch_shape_dhw = _patch_shape(
            settings["patch_shape_dhw"] if patch_shape_dhw is None else patch_shape_dhw
        )
        repeat = settings["train_patches_per_volume"] if train_patches_per_volume is None else train_patches_per_volume
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError("train_patches_per_volume must be a positive integer.")
        self.train_patches_per_volume = repeat if split == "train" else 1
        probability = (
            settings["train_foreground_probability"]
            if train_foreground_probability is None else train_foreground_probability
        )
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("train_foreground_probability must be finite and between 0 and 1.")
        self.train_foreground_probability = float(probability)
        self.seed = _nonnegative_int(settings["seed"] if seed is None else seed, "seed")
        self.epoch = 0
        self.split = split

        resampling_settings = json.loads(
            Path(__file__).with_name("resampling_config.json").read_text(encoding="utf-8")
        )
        self.target_spacing_mm = tuple(resampling_settings["target_spacing_mm"])
        self.max_estimated_bytes = resampling_settings["max_estimated_bytes"]
        # Production discovery validates all 211 pairs and the frozen split.
        self.pairs: tuple[VolumePair, ...] = discover_dataset(dataset_root)[split]
        approved = set(APPROVED_PATIENTS[split])
        if not self.pairs or any(pair.patient_id not in approved for pair in self.pairs):
            raise ValueError(f"{split}: discovered pairs are empty or include patients outside the approved split.")

    def __len__(self) -> int:
        return len(self.pairs) * self.train_patches_per_volume

    def set_epoch(self, epoch: int) -> None:
        """Change reproducible training crops between epochs; validation stays centered."""

        self.epoch = _nonnegative_int(epoch, "epoch")

    def __getitem__(self, index: int) -> dict[str, object]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(f"patch index {index!r} is outside 0..{len(self) - 1}.")
        pair = self.pairs[index // self.train_patches_per_volume]
        loaded = load_volume_pair(pair)
        resampled = resample_volume_pair(
            loaded, self.target_spacing_mm, max_estimated_bytes=self.max_estimated_bytes
        )
        del loaded
        normalized = normalize_volume_pair(resampled)
        del resampled

        mri_xyz = normalized.mri
        mask_xyz = normalized.segmentation
        if (
            not isinstance(mri_xyz, np.ndarray) or not isinstance(mask_xyz, np.ndarray)
            or mri_xyz.ndim != 3 or mask_xyz.shape != mri_xyz.shape
            or mri_xyz.dtype != np.float32 or mask_xyz.dtype != np.uint8
            or np.any(mask_xyz > 5) or not np.isfinite(mri_xyz).all()
        ):
            raise ValueError(f"{pair.patient_id}_Week{pair.week}: invalid normalised MRI/mask arrays.")

        # Tensor-axis permutation only: source (X,Y,Z) -> tensor (D,H,W)=(Z,Y,X).
        mri_dhw = np.transpose(mri_xyz, (2, 1, 0))
        mask_dhw = np.transpose(mask_xyz, (2, 1, 0))
        volume_shape = mri_dhw.shape
        if any(volume < patch for volume, patch in zip(volume_shape, self.patch_shape_dhw)):
            raise ValueError(
                f"{pair.patient_id}_Week{pair.week}: volume shape (D,H,W) {volume_shape} "
                f"is smaller than patch {self.patch_shape_dhw}; padding is not supported."
            )

        if self.split == "validation":
            origin = tuple((volume - patch) // 2 for volume, patch in zip(volume_shape, self.patch_shape_dhw))
            sampling_mode = "center"
        else:
            rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))
            use_foreground = rng.random() < self.train_foreground_probability
            foreground = np.flatnonzero(mask_dhw == 5) if use_foreground else np.empty(0, dtype=np.intp)
            if foreground.size:
                selected = np.unravel_index(int(foreground[rng.integers(foreground.size)]), volume_shape)
                origin = tuple(
                    int(rng.integers(max(0, voxel - patch + 1), min(voxel, volume - patch) + 1))
                    for voxel, volume, patch in zip(selected, volume_shape, self.patch_shape_dhw)
                )
                sampling_mode = "prostate"
            else:
                origin = tuple(
                    int(rng.integers(volume - patch + 1))
                    for volume, patch in zip(volume_shape, self.patch_shape_dhw)
                )
                sampling_mode = "random_fallback" if use_foreground else "random"

        crop = tuple(slice(start, start + width) for start, width in zip(origin, self.patch_shape_dhw))
        image = torch.from_numpy(np.ascontiguousarray(mri_dhw[crop])).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask_dhw[crop], dtype=np.int64))
        return {
            "mri": image,
            "mask": mask,
            "patient_id": pair.patient_id,
            "week": pair.week,
            "crop_origin_dhw": torch.tensor(origin, dtype=torch.long),
            "sampling_mode": sampling_mode,
        }
