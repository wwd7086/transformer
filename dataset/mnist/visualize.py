"""
Visualization script for MNIST dataset.
Displays sample images in a grid with their labels.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist.data import get_dataset_info


def visualize_samples(
    data_dir: str, split: str = "train", num_samples: int = 25, save_path: str = None
):
    """
    Visualize random samples from MNIST dataset.

    Args:
        data_dir: Path to MNIST data directory
        split: 'train' or 'test'
        num_samples: Number of samples to display (must be perfect square)
        save_path: Optional path to save the figure
    """
    # Validate num_samples is a perfect square
    grid_size = int(np.sqrt(num_samples))
    if grid_size * grid_size != num_samples:
        raise ValueError(f"num_samples must be a perfect square, got {num_samples}")

    # Load data
    if split == "train":
        images = np.load(os.path.join(data_dir, "train_images.npy"))
        labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    else:
        images = np.load(os.path.join(data_dir, "test_images.npy"))
        labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Randomly sample indices
    indices = np.random.choice(len(images), num_samples, replace=False)

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle(f"MNIST {split.capitalize()} Set - Random Samples", fontsize=16)

    for i, idx in enumerate(indices):
        row = i // grid_size
        col = i % grid_size
        ax = axes[row, col]

        # Display image
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(f"Label: {labels[idx]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_class_samples(
    data_dir: str,
    split: str = "train",
    samples_per_class: int = 10,
    save_path: str = None,
):
    """
    Visualize samples from each class.

    Args:
        data_dir: Path to MNIST data directory
        split: 'train' or 'test'
        samples_per_class: Number of samples to show per class
        save_path: Optional path to save the figure
    """
    # Load data
    if split == "train":
        images = np.load(os.path.join(data_dir, "train_images.npy"))
        labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    else:
        images = np.load(os.path.join(data_dir, "test_images.npy"))
        labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    # Create figure
    fig, axes = plt.subplots(
        10, samples_per_class, figsize=(samples_per_class * 1.5, 15)
    )
    fig.suptitle(f"MNIST {split.capitalize()} Set - Samples per Class", fontsize=16)

    # For each class
    for digit in range(10):
        # Get indices for this digit
        digit_indices = np.where(labels == digit)[0]

        # Randomly sample
        sample_indices = np.random.choice(
            digit_indices, samples_per_class, replace=False
        )

        # Display samples
        for i, idx in enumerate(sample_indices):
            ax = axes[digit, i]
            ax.imshow(images[idx], cmap="gray")

            # Only show label on first column
            if i == 0:
                ax.set_ylabel(
                    f"Digit {digit}", fontsize=12, rotation=0, ha="right", va="center"
                )

            ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_dataset_statistics(data_dir: str, save_path: str = None):
    """
    Plot dataset statistics including class distributions.

    Args:
        data_dir: Path to MNIST data directory
        save_path: Optional path to save the figure
    """
    info = get_dataset_info(data_dir)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training set distribution
    classes = list(info["train_class_counts"].keys())
    train_counts = list(info["train_class_counts"].values())

    axes[0].bar(classes, train_counts, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Digit Class", fontsize=12)
    axes[0].set_ylabel("Number of Samples", fontsize=12)
    axes[0].set_title(
        f"Training Set Distribution (Total: {info['num_train']:,})", fontsize=13
    )
    axes[0].set_xticks(classes)
    axes[0].grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, count in enumerate(train_counts):
        axes[0].text(i, count + 50, str(count), ha="center", va="bottom", fontsize=9)

    # Plot test set distribution
    test_counts = list(info["test_class_counts"].values())

    axes[1].bar(classes, test_counts, color="coral", alpha=0.8)
    axes[1].set_xlabel("Digit Class", fontsize=12)
    axes[1].set_ylabel("Number of Samples", fontsize=12)
    axes[1].set_title(
        f"Test Set Distribution (Total: {info['num_test']:,})", fontsize=13
    )
    axes[1].set_xticks(classes)
    axes[1].grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, count in enumerate(test_counts):
        axes[1].text(i, count + 10, str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize MNIST dataset")
    parser.add_argument(
        "--mode",
        choices=["random", "classes", "stats", "all"],
        default="random",
        help="Visualization mode",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of samples for random mode (must be perfect square)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="Number of samples per class for classes mode",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save figures instead of displaying"
    )

    args = parser.parse_args()

    # Get data directory (absolute path)
    data_dir = os.path.abspath(os.path.dirname(__file__))

    if args.mode == "random" or args.mode == "all":
        save_path = f"mnist_{args.split}_random.png" if args.save else None
        print(f"\nVisualizing random samples from {args.split} set...")
        visualize_samples(data_dir, args.split, args.num_samples, save_path)

    if args.mode == "classes" or args.mode == "all":
        save_path = f"mnist_{args.split}_classes.png" if args.save else None
        print(f"\nVisualizing class samples from {args.split} set...")
        visualize_class_samples(data_dir, args.split, args.samples_per_class, save_path)

    if args.mode == "stats" or args.mode == "all":
        save_path = "mnist_statistics.png" if args.save else None
        print("\nVisualizing dataset statistics...")
        plot_dataset_statistics(data_dir, save_path)


if __name__ == "__main__":
    main()
