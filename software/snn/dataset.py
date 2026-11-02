"""MNIST training and test data pipelines."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from software.paths import DATA_DIR


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str | Path = DATA_DIR,
) -> tuple[DataLoader, DataLoader]:
    """Create separate shuffled training and ordered test MNIST loaders.

    ``ToTensor`` converts grayscale pixels to float tensors in [0, 1]. No
    mean/std normalization is applied because the SNN uses pixel brightness
    directly as a current amplitude.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
