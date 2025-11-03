# MNIST Dataset

This directory contains utilities for downloading and loading the MNIST dataset for training and testing.

## Setup

1. First, download and prepare the dataset:

```bash
uv run dataset/mnist/prepare.py
```

This will:
- Download the MNIST dataset from a mirror
- Process and normalize the images (28x28 pixels, float32, normalized to [0, 1])
- Save the data as numpy arrays:
  - `train_images.npy`: 60,000 training images
  - `train_labels.npy`: 60,000 training labels
  - `test_images.npy`: 10,000 test images
  - `test_labels.npy`: 10,000 test labels

## Usage

### Quick Batch Loading

```python
from dataset.mnist.data import get_batch

# Get a batch of training data
images, labels = get_batch(
    data_dir='dataset/mnist',
    split='train',
    batch_size=32,
    device='cpu'
)

# Get a batch of test data
test_images, test_labels = get_batch(
    data_dir='dataset/mnist',
    split='test',
    batch_size=32,
    device='cpu'
)
```

### Using the Data Loader

```python
from dataset.mnist.data import MNISTDataLoader

# Create a training data loader
train_loader = MNISTDataLoader(
    data_dir='dataset/mnist',
    split='train',
    batch_size=64,
    device='cpu',
    shuffle=True
)

# Iterate through batches
for images, labels in train_loader:
    # images: (batch_size, 28, 28) torch.Tensor
    # labels: (batch_size,) torch.Tensor
    pass
```

### Getting Dataset Information

```python
from dataset.mnist.data import get_dataset_info

info = get_dataset_info('dataset/mnist')
print(f"Number of classes: {info['num_classes']}")
print(f"Training samples: {info['num_train']}")
print(f"Test samples: {info['num_test']}")
```

## Dataset Details

- **Image size**: 28x28 pixels
- **Number of classes**: 10 (digits 0-9)
- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Data format**: Float32, normalized to [0, 1] range
- **Labels**: Integer labels from 0 to 9

## Testing

Run the test script to verify everything works:

```bash
uv run python -m dataset.mnist.test_data
```

## Visualization

Visualize the MNIST dataset using the visualization script:

```bash
# Visualize random samples (default: 25 images)
uv run python -m dataset.mnist.visualize --mode random --save

# Visualize samples from each class
uv run python -m dataset.mnist.visualize --mode classes --samples-per-class 8 --save

# Visualize dataset statistics
uv run python -m dataset.mnist.visualize --mode stats --save

# Generate all visualizations
uv run python -m dataset.mnist.visualize --mode all --save

# Display interactively (without --save flag)
uv run python -m dataset.mnist.visualize --mode random

# Options:
#   --mode: random, classes, stats, or all
#   --split: train or test
#   --num-samples: number of samples for random mode (must be perfect square)
#   --samples-per-class: samples per class for classes mode
#   --save: save figures instead of displaying
```
