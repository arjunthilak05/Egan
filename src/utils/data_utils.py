"""
Data utilities for loading and preprocessing datasets.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def get_mnist_dataloader(
    batch_size: int = 64,
    image_size: int = 28,
    normalize: bool = True,
    flatten: bool = True,
    train: bool = True,
    download: bool = True
) -> DataLoader:
    """
    Get MNIST dataloader.

    Args:
        batch_size: Batch size
        image_size: Target image size (will resize if different from 28)
        normalize: Whether to normalize to [-1, 1]
        flatten: Whether to flatten images
        train: Load training set (True) or test set (False)
        download: Download dataset if not present

    Returns:
        DataLoader
    """
    transform_list = []

    # Resize if needed
    if image_size != 28:
        transform_list.append(transforms.Resize(image_size))

    transform_list.append(transforms.ToTensor())

    # Normalize to [-1, 1]
    if normalize:
        transform_list.append(transforms.Normalize([0.5], [0.5]))

    transform = transforms.Compose(transform_list)

    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=train,
        download=download,
        transform=transform
    )

    # Flatten if requested
    if flatten:
        # Convert to flat tensors
        images = []
        labels = []

        for img, label in dataset:
            images.append(img.flatten())
            labels.append(label)

        images = torch.stack(images)
        labels = torch.tensor(labels)

        dataset = TensorDataset(images, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )

    return dataloader


def get_fashion_mnist_dataloader(
    batch_size: int = 64,
    image_size: int = 28,
    normalize: bool = True,
    flatten: bool = True,
    train: bool = True,
    download: bool = True
) -> DataLoader:
    """
    Get Fashion-MNIST dataloader.

    Args: Same as get_mnist_dataloader

    Returns:
        DataLoader
    """
    transform_list = []

    if image_size != 28:
        transform_list.append(transforms.Resize(image_size))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize([0.5], [0.5]))

    transform = transforms.Compose(transform_list)

    dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=train,
        download=download,
        transform=transform
    )

    if flatten:
        images = []
        labels = []

        for img, label in dataset:
            images.append(img.flatten())
            labels.append(label)

        images = torch.stack(images)
        labels = torch.tensor(labels)

        dataset = TensorDataset(images, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0
    )

    return dataloader


def downsample_images(images: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Downsample images to target size.

    Args:
        images: Image tensor [batch, channels, height, width]
        target_size: Target height/width

    Returns:
        Downsampled images
    """
    resize = transforms.Resize(target_size)
    return resize(images)


def prepare_reduced_mnist(
    n_samples: int = 10000,
    image_size: int = 8,
    seed: int = 42
) -> tuple:
    """
    Prepare reduced MNIST dataset (smaller images, fewer samples).

    Useful for faster quantum experiments.

    Args:
        n_samples: Number of samples to use
        image_size: Target image size (e.g., 8x8)
        seed: Random seed

    Returns:
        (images, labels)
    """
    torch.manual_seed(seed)

    # Load full MNIST
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Sample subset
    indices = torch.randperm(len(dataset))[:n_samples]
    images = []
    labels = []

    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels
