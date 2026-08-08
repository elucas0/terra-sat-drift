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
        max_samples: cap on *training* patches. Deliberately does **not** size
            the evaluation sets -- see below.
        val_max_samples: cap on validation patches, independent of
            ``max_samples``. Validation runs every epoch, so it has to stay
            cheap; 1000 is the smallest size that still contains all 11
            WorldCover classes (snow/ice appears in 14 of them).
        test_max_samples: cap on test patches; ``None`` uses the full 25,323.
            Test runs once, so it can afford to be thorough -- the full split
            gives snow/ice 255 patches instead of 14.
        target_domain / source_domain: which HDF5 views to pair.

    Why the evaluation sets are decoupled from ``max_samples``: they used to be
    ``max_samples // 10``, which meant a label-scarcity study changed the
    *measurement* at the same time as the treatment. At ``--max-samples 1000``
    that left 100 evaluation patches, in which snow/ice is absent entirely --
    and an absent class silently changes the denominator of
    ``torchmetrics.JaccardIndex``, so two models were being averaged over
    different numbers of classes and their mIoU was not comparable.
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
        val_max_samples: Optional[int] = 1000,
        test_max_samples: Optional[int] = None,
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
        self.val_max_samples = val_max_samples
        self.test_max_samples = test_max_samples
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
        if stage in (None, "fit"):
            self.train_dataset = self._build("train", self.train_transform, self.max_samples)
            self.val_dataset = self._build("val", self.val_transform, self.val_max_samples)
        if stage in (None, "fit", "test"):
            self.test_dataset = self._build("test", self.val_transform, self.test_max_samples)

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
