"""
MNIST dataset loading utilities for training and testing.
"""

import os
import numpy as np
import torch


def get_batch(
    data_dir: str,
    split: str,
    batch_size: int,
    device: str = "cpu",
    filter_label: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of MNIST images and labels.

    Args:
        data_dir: Path to the directory containing MNIST data files
        split: Either 'train' or 'test'
        batch_size: Number of samples in the batch
        device: Device to place tensors on ('cpu' or 'cuda')
        filter_label: Optional label (0-9) to filter samples by. If provided,
                      only samples with this label will be included in the batch.

    Returns:
        Tuple of (images, labels) tensors
        - images: shape (batch_size, 28, 28), float32, normalized to [0, 1]
        - labels: shape (batch_size,), int64
    """
    if split == "train":
        images = np.load(os.path.join(data_dir, "train_images.npy"), mmap_mode="r")
        labels = np.load(os.path.join(data_dir, "train_labels.npy"), mmap_mode="r")
    elif split == "test" or split == "val":
        images = np.load(os.path.join(data_dir, "test_images.npy"), mmap_mode="r")
        labels = np.load(os.path.join(data_dir, "test_labels.npy"), mmap_mode="r")
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'val'")

    # Filter by label if specified
    if filter_label is not None:
        if not (0 <= filter_label <= 9):
            raise ValueError(f"Invalid label: {filter_label}. Must be between 0 and 9.")
        # Get indices of samples matching the specified label
        valid_indices = np.where(labels == filter_label)[0]
        if len(valid_indices) == 0:
            raise ValueError(f"No samples found with label {filter_label}")
        # Randomly sample from the filtered indices
        indices = np.random.choice(
            valid_indices, size=batch_size, replace=len(valid_indices) < batch_size
        )
    else:
        # Randomly sample batch_size indices from all samples
        num_samples = len(labels)
        indices = np.random.randint(0, num_samples, size=batch_size)

    # Get batch
    batch_images = images[indices]
    batch_labels = labels[indices]

    # Convert to tensors
    images_tensor = torch.from_numpy(batch_images.copy()).to(device)
    labels_tensor = torch.from_numpy(batch_labels.copy()).long().to(device)

    return images_tensor, labels_tensor


def get_full_dataset(
    data_dir: str, split: str, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the full MNIST dataset for a given split.

    Args:
        data_dir: Path to the directory containing MNIST data files
        split: Either 'train' or 'test'
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        Tuple of (images, labels) tensors
        - images: shape (N, 28, 28), float32, normalized to [0, 1]
        - labels: shape (N,), int64
    """
    if split == "train":
        images = np.load(os.path.join(data_dir, "train_images.npy"))
        labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    elif split == "test" or split == "val":
        images = np.load(os.path.join(data_dir, "test_images.npy"))
        labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'val'")

    # Convert to tensors
    images_tensor = torch.from_numpy(images).to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)

    return images_tensor, labels_tensor


def get_dataset_info(data_dir: str) -> dict:
    """
    Get information about the MNIST dataset.

    Args:
        data_dir: Path to the directory containing MNIST data files

    Returns:
        Dictionary containing dataset information
    """
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"), mmap_mode="r")
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"), mmap_mode="r")

    info = {
        "num_classes": 10,
        "image_shape": (28, 28),
        "num_train": len(train_labels),
        "num_test": len(test_labels),
        "train_class_counts": {i: int(np.sum(train_labels == i)) for i in range(10)},
        "test_class_counts": {i: int(np.sum(test_labels == i)) for i in range(10)},
    }

    return info


class MNISTDataLoader:
    """
    Simple data loader for MNIST dataset that provides iteration support.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int,
        device: str = "cpu",
        shuffle: bool = True,
        filter_label: int | None = None,
    ):
        """
        Initialize MNIST data loader.

        Args:
            data_dir: Path to the directory containing MNIST data files
            split: Either 'train' or 'test'
            batch_size: Number of samples per batch
            device: Device to place tensors on ('cpu' or 'cuda')
            shuffle: Whether to shuffle the data
            filter_label: Optional label (0-9) to filter samples by. If provided,
                          only samples with this label will be included.
        """
        self.data_dir = data_dir
        self.split = split
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.filter_label = filter_label

        # Load data
        if split == "train":
            images = np.load(os.path.join(data_dir, "train_images.npy"), mmap_mode="r")
            labels = np.load(os.path.join(data_dir, "train_labels.npy"), mmap_mode="r")
        elif split == "test" or split == "val":
            images = np.load(os.path.join(data_dir, "test_images.npy"), mmap_mode="r")
            labels = np.load(os.path.join(data_dir, "test_labels.npy"), mmap_mode="r")
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'test', or 'val'"
            )

        # Filter by label if specified
        if filter_label is not None:
            if not (0 <= filter_label <= 9):
                raise ValueError(
                    f"Invalid label: {filter_label}. Must be between 0 and 9."
                )
            valid_indices = np.where(labels == filter_label)[0]
            if len(valid_indices) == 0:
                raise ValueError(f"No samples found with label {filter_label}")
            self.images = images[valid_indices]
            self.labels = labels[valid_indices]
        else:
            self.images = images
            self.labels = labels

        self.num_samples = len(self.labels)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """Iterate over batches."""
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            batch_images = self.images[batch_indices]
            batch_labels = self.labels[batch_indices]

            # Convert to tensors
            images_tensor = torch.from_numpy(batch_images.copy()).to(self.device)
            labels_tensor = torch.from_numpy(batch_labels.copy()).long().to(self.device)

            yield images_tensor, labels_tensor
