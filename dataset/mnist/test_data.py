"""
Test script to verify MNIST dataset loading functionality.
"""

import os

from dataset.mnist.data import get_batch, get_dataset_info, MNISTDataLoader

# Get data directory
data_dir = os.path.dirname(__file__)

print("Testing MNIST data loading functionality...")
print("=" * 60)

# Test 1: Get dataset info
print("\n1. Testing get_dataset_info()...")
info = get_dataset_info(data_dir)
print(f"   Number of classes: {info['num_classes']}")
print(f"   Image shape: {info['image_shape']}")
print(f"   Number of training samples: {info['num_train']}")
print(f"   Number of test samples: {info['num_test']}")
print(f"   Training class distribution: {info['train_class_counts']}")
print(f"   Test class distribution: {info['test_class_counts']}")

# Test 2: Get a batch from training set
print("\n2. Testing get_batch() for training set...")
train_images, train_labels = get_batch(data_dir, "train", batch_size=32)
print(f"   Training batch images shape: {train_images.shape}")
print(f"   Training batch labels shape: {train_labels.shape}")
print(f"   Images dtype: {train_images.dtype}")
print(f"   Labels dtype: {train_labels.dtype}")
print(f"   Image value range: [{train_images.min():.3f}, {train_images.max():.3f}]")
print(f"   Sample labels: {train_labels[:10].tolist()}")

# Test 3: Get a batch from test set
print("\n3. Testing get_batch() for test set...")
test_images, test_labels = get_batch(data_dir, "test", batch_size=16)
print(f"   Test batch images shape: {test_images.shape}")
print(f"   Test batch labels shape: {test_labels.shape}")
print(f"   Sample labels: {test_labels[:10].tolist()}")

# Test 4: Test MNISTDataLoader
print("\n4. Testing MNISTDataLoader...")
train_loader = MNISTDataLoader(data_dir, "train", batch_size=64, shuffle=True)
print(f"   Number of batches: {len(train_loader)}")
print(f"   Number of samples: {train_loader.num_samples}")

# Iterate through a few batches
print("   Iterating through first 3 batches...")
for i, (images, labels) in enumerate(train_loader):
    if i >= 3:
        break
    print(f"   Batch {i+1}: images {images.shape}, labels {labels.shape}")

# Test 5: Test test set loader
print("\n5. Testing test set loader...")
test_loader = MNISTDataLoader(data_dir, "test", batch_size=100, shuffle=False)
print(f"   Number of batches: {len(test_loader)}")
print(f"   Number of samples: {test_loader.num_samples}")

print("\n" + "=" * 60)
print("All tests passed successfully!")
