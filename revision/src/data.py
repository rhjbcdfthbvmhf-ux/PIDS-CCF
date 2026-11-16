"""Leakage-audited HSI loading, splitting, scaling, and patch extraction."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import binary_dilation
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DATASETS = {
    "IndianPines": (
        "Indian_pines_corrected.mat",
        "Indian_pines_gt.mat",
        "indian_pines_corrected",
        "indian_pines_gt",
    ),
    "PaviaU": ("PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt"),
    "Salinas": (
        "Salinas_corrected.mat",
        "Salinas_gt.mat",
        "salinas_corrected",
        "salinas_gt",
    ),
}


@dataclass(frozen=True)
class SplitBundle:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_cube(data_root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    cube_file, gt_file, cube_key, gt_key = DATASETS[name]
    cube = np.asarray(loadmat(data_root / cube_file)[cube_key], dtype=np.float32)
    gt = np.asarray(loadmat(data_root / gt_file)[gt_key], dtype=np.int64)
    if cube.shape[:2] != gt.shape:
        raise ValueError(f"cube/ground-truth mismatch: {cube.shape} versus {gt.shape}")
    return cube, gt


def stratified_split(
    gt: np.ndarray,
    seed: int,
    development_fraction: float = 0.15,
    validation_fraction_of_development: float = 0.10,
) -> SplitBundle:
    coords = np.argwhere(gt > 0)
    labels = gt[coords[:, 0], coords[:, 1]] - 1
    dev, test = train_test_split(
        np.arange(len(coords)),
        train_size=development_fraction,
        stratify=labels,
        random_state=seed,
    )
    train, val = train_test_split(
        dev,
        test_size=validation_fraction_of_development,
        stratify=labels[dev],
        random_state=seed + 10000,
    )
    arrays = [coords[x].astype(np.int32, copy=False) for x in (train, val, test)]
    for left, right in ((arrays[0], arrays[1]), (arrays[0], arrays[2]), (arrays[1], arrays[2])):
        a = {tuple(x) for x in left.tolist()}
        b = {tuple(x) for x in right.tolist()}
        if a.intersection(b):
            raise AssertionError("split center coordinates overlap")
    return SplitBundle(train=arrays[0], val=arrays[1], test=arrays[2], seed=seed)


def spatial_cluster_split(
    gt: np.ndarray,
    seed: int,
    development_fraction: float = 0.15,
    validation_fraction_of_development: float = 0.10,
    buffer_radius: int = 5,
) -> SplitBundle:
    """Build class-covering compact development clusters and a buffered test set.

    Development coordinates for each class are selected near a seeded class pixel.
    Every labeled center within Chebyshev ``buffer_radius`` of any development
    center is excluded from testing. This is a conservative sensitivity split,
    not an 85% random-pixel split.
    """
    if buffer_radius < 0:
        raise ValueError("buffer_radius must be nonnegative")
    rng = np.random.default_rng(seed)
    train_parts, val_parts, dev_parts = [], [], []
    for class_index in range(1, int(gt.max()) + 1):
        class_coords = np.argwhere(gt == class_index)
        if len(class_coords) < 3:
            raise ValueError(f"class {class_index} has fewer than three pixels")
        anchor = class_coords[int(rng.integers(len(class_coords)))]
        distance = np.sum((class_coords - anchor[None, :]) ** 2, axis=1)
        # Seeded jitter resolves equal-distance ties without favoring raster order.
        order = np.lexsort((rng.random(len(class_coords)), distance))
        development_count = min(
            len(class_coords) - 1,
            max(2, int(round(development_fraction * len(class_coords)))),
        )
        class_dev = class_coords[order[:development_count]].astype(np.int32)
        validation_count = min(
            development_count - 1,
            max(1, int(round(validation_fraction_of_development * development_count))),
        )
        validation_indices = rng.choice(
            development_count, validation_count, replace=False
        )
        validation_mask = np.zeros(development_count, dtype=bool)
        validation_mask[validation_indices] = True
        train_parts.append(class_dev[~validation_mask])
        val_parts.append(class_dev[validation_mask])
        dev_parts.append(class_dev)

    train = np.concatenate(train_parts).astype(np.int32, copy=False)
    val = np.concatenate(val_parts).astype(np.int32, copy=False)
    dev = np.concatenate(dev_parts).astype(np.int32, copy=False)
    dev_mask = np.zeros(gt.shape, dtype=bool)
    dev_mask[dev[:, 0], dev[:, 1]] = True
    if buffer_radius > 0:
        footprint = np.ones(
            (2 * buffer_radius + 1, 2 * buffer_radius + 1), dtype=bool
        )
        excluded = binary_dilation(dev_mask, structure=footprint)
    else:
        excluded = dev_mask
    test = np.argwhere((gt > 0) & ~excluded).astype(np.int32)
    observed = set(gt[test[:, 0], test[:, 1]].tolist()) if len(test) else set()
    missing = sorted(set(range(1, int(gt.max()) + 1)) - observed)
    if missing:
        raise ValueError(
            f"buffered split leaves no test centers for classes {missing}; "
            "use a different seed or smaller buffer"
        )
    for left, right in ((train, val), (train, test), (val, test)):
        if {tuple(x) for x in left.tolist()}.intersection(
            {tuple(x) for x in right.tolist()}
        ):
            raise AssertionError("split center coordinates overlap")
    return SplitBundle(train=train, val=val, test=test, seed=seed)


def save_split(bundle: SplitBundle, gt: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": bundle.seed,
        "train": bundle.train.tolist(),
        "val": bundle.val.tolist(),
        "test": bundle.test.tolist(),
        "counts": {
            key: {
                str(int(c)): int(np.sum(gt[arr[:, 0], arr[:, 1]] == c))
                for c in np.unique(gt[arr[:, 0], arr[:, 1]])
            }
            for key, arr in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test))
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_only_scale(cube: np.ndarray, train_coords: np.ndarray) -> tuple[np.ndarray, dict]:
    train_spectra = cube[train_coords[:, 0], train_coords[:, 1]]
    lo = train_spectra.min(axis=0)
    hi = train_spectra.max(axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    normalized = np.clip((cube - lo[None, None, :]) / scale[None, None, :], 0.0, 1.0)
    return normalized.astype(np.float32, copy=False), {
        "fit_scope": "training center spectra only",
        "minimum": lo.tolist(),
        "maximum": hi.tolist(),
    }


class HSIPatchDataset(Dataset):
    def __init__(
        self,
        cube: np.ndarray,
        gt: np.ndarray,
        coords: np.ndarray,
        patch_size: int,
    ) -> None:
        if patch_size % 2 != 1:
            raise ValueError("patch size must be odd")
        self.coords = np.asarray(coords, dtype=np.int32)
        self.labels = gt[self.coords[:, 0], self.coords[:, 1]].astype(np.int64) - 1
        self.patch_size = int(patch_size)
        margin = patch_size // 2
        padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
        self.padded = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = self.coords[index]
        patch = self.padded[:, row : row + self.patch_size, col : col + self.patch_size]
        return patch, torch.tensor(self.labels[index]), torch.tensor(index)


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: HSIPatchDataset,
    batch_size: int,
    workers: int,
    seed: int,
    balanced: bool,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balanced:
        counts = np.bincount(dataset.labels)
        sample_weights = 1.0 / np.maximum(counts[dataset.labels], 1)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=balanced,
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=workers > 0,
    )


def class_counts(gt: np.ndarray, coords: Iterable[np.ndarray]) -> np.ndarray:
    coords = np.asarray(list(coords))
    labels = gt[coords[:, 0], coords[:, 1]] - 1
    return np.bincount(labels, minlength=int(gt.max()))
