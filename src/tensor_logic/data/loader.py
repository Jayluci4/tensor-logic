"""Data loading utilities for Tensor Logic."""

from torch.utils.data import DataLoader
from typing import Tuple, Optional

from .dataset import TripleDataset


def create_dataloaders(
    train_dataset: TripleDataset,
    val_dataset: Optional[TripleDataset] = None,
    test_dataset: Optional[TripleDataset] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for train/val/test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
