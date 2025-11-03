"""
Prepare the MNIST dataset for training and testing.
Downloads the MNIST dataset and saves it as numpy arrays.
Will save train_images.npy, train_labels.npy, test_images.npy, test_labels.npy.
"""
import os
import gzip
import numpy as np
import requests


def download_file(url, filename):
    """Download file from URL if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists, skipping download")


def load_mnist_images(filename):
    """Load MNIST images from gzipped file."""
    with gzip.open(filename, 'rb') as f:
        # Read header
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Read image data
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

    return images


def load_mnist_labels(filename):
    """Load MNIST labels from gzipped file."""
    with gzip.open(filename, 'rb') as f:
        # Read header
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        _ = int.from_bytes(f.read(4), 'big')  # num_labels

        # Read label data
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

    return labels


# Base URL for MNIST dataset (using mirror since original site is down)
base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

# File names
files = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz',
}

# Download all files
data_dir = os.path.dirname(__file__)
for key, filename in files.items():
    filepath = os.path.join(data_dir, filename)
    download_file(base_url + filename, filepath)

# Load and process the data
print("\nLoading MNIST dataset...")
train_images = load_mnist_images(os.path.join(data_dir, files['train_images']))
train_labels = load_mnist_labels(os.path.join(data_dir, files['train_labels']))
test_images = load_mnist_images(os.path.join(data_dir, files['test_images']))
test_labels = load_mnist_labels(os.path.join(data_dir, files['test_labels']))

print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Normalize images to [0, 1] range and save as float32
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Save as numpy arrays
print("\nSaving processed data...")
np.save(os.path.join(data_dir, 'train_images.npy'), train_images)
np.save(os.path.join(data_dir, 'train_labels.npy'), train_labels)
np.save(os.path.join(data_dir, 'test_images.npy'), test_images)
np.save(os.path.join(data_dir, 'test_labels.npy'), test_labels)

print("\nDataset preparation complete!")
print(f"Files saved to {data_dir}")
print(f"  - train_images.npy: {train_images.shape} (float32, normalized)")
print(f"  - train_labels.npy: {train_labels.shape} (uint8)")
print(f"  - test_images.npy: {test_images.shape} (float32, normalized)")
print(f"  - test_labels.npy: {test_labels.shape} (uint8)")
