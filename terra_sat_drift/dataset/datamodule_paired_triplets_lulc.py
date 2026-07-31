"""LightningDataModule for the co-registered two-sensor LULC dataset."""

from typing import Optional

import albumentations as A
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from .dataset_paired_triplets_lulc import PhisatPairedLULCDataset, build_paired_transform


class PhisatPairedLULCDataModule(pl.LightningDataModule):
    """Serves batches holding both sensor views of the same patches.

    Args:
        h5_images_path: triplets HDF5 (real / sim / s2b under a shared index).
        h5_labels_path: WorldCover labels HDF5.
        manifest_path: manifest CSV.
        batch_size: patches per batch. Note each batch costs two forward passes
            through the student (one per sensor), and the contrastive terms draw
            their negatives from within the batch, so batch size trades off
            against negative-sample count. See ``kd_contrastive_module``.
        num_workers: DataLoader workers.
        train_transform: joint augmentation; defaults to
            ``build_paired_transform(train=True)`` when left as ``None``. Pass
            ``A.Compose([])`` explicitly to disable augmentation.
        val_transform: joint augmentation for val/test; ``None`` (no-op) is the
            sensible default.
        max_samples: cap on training samples (val/test get a tenth of it).
        target_domain / source_domain: which HDF5 views to pair.
    """

    def __init__(
        self,
        h5_images_path: str,
        h5_labels_path: str,
        manifest_path: str,
        batch_size: int = 8,
        num_workers: int = 8,
        train_transform: A.Compose | None = None,
        val_transform: A.Compose | None = None,
        max_samples: Optional[int] = None,
        target_domain: str = "real",
        source_domain: Optional[str] = "s2b",
        augment: bool = True,
    ):
        super().__init__()
        self.h5_images_path = h5_images_path
        self.h5_labels_path = h5_labels_path
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        # `augment=False` yields no train-time transform at all; an explicit
        # `train_transform` still wins over both.
        self.train_transform = (
            train_transform if train_transform is not None
            else (build_paired_transform(train=True) if augment else None)
        )
        self.val_transform = val_transform
        self.max_samples = max_samples
        self.target_domain = target_domain
        self.source_domain = source_domain

    def _build(self, split: str, transform, max_samples) -> PhisatPairedLULCDataset:
        return PhisatPairedLULCDataset(
            h5_images_path=self.h5_images_path,
            h5_labels_path=self.h5_labels_path,
            manifest_path=self.manifest_path,
            split=split,
            transform=transform,
            max_samples=max_samples,
            target_domain=self.target_domain,
            source_domain=self.source_domain,
        )

    def setup(self, stage: Optional[str] = None):
        train_max = self.max_samples
        eval_max = max(1, self.max_samples // 10) if self.max_samples else None

        if stage in (None, "fit"):
            self.train_dataset = self._build("train", self.train_transform, train_max)
            self.val_dataset = self._build("val", self.val_transform, eval_max)
        if stage in (None, "fit", "test"):
            self.test_dataset = self._build("test", self.val_transform, eval_max)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            # The in-batch negatives of NT-Xent make the loss batch-size
            # dependent, and a short trailing batch also destabilises the
            # prototype EMA, so incomplete training batches are dropped.
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
