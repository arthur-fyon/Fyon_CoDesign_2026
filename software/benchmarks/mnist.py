"""MNIST benchmark tasks (sequential and permuted)."""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# Global cache for PMNIST data
_PMNIST_CACHE = {}


def download_mnist(data_dir: str = "./data/mnist"):
    """Download MNIST dataset from alternative sources."""
    import urllib.request
    import gzip
    import numpy as np
    from pathlib import Path

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def download_file(filename):
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                # Add user agent to avoid 403 errors
                req = urllib.request.Request(
                    base_url + filename,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=30) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                print(f"Network access to {base_url} may be blocked.")
                print(f"Please enable network access in Claude settings or provide MNIST data manually.")
                raise RuntimeError(
                    f"Cannot download MNIST. Network access required to {base_url}. "
                    f"Please update network settings to allow access to S3 datasets."
                )
        return filepath

    def load_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            # Read header
            _ = int.from_bytes(f.read(4), 'big')  # magic number
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')

            # Read image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, rows * cols)

            # Normalize to [0, 1]
            return (data.astype(np.float32) / 255.0) - 0.5 # Standard MNIST normalization

    def load_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            # Read header
            _ = int.from_bytes(f.read(4), 'big')  # magic number
            num_labels = int.from_bytes(f.read(4), 'big')

            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    # Download all files
    for key, filename in files.items():
        download_file(filename)

    # Load data
    train_images = load_images(data_dir / files['train_images'])
    train_labels = load_labels(data_dir / files['train_labels'])
    test_images = load_images(data_dir / files['test_images'])
    test_labels = load_labels(data_dir / files['test_labels'])

    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }


def get_pmnist_data(permutation_seed: int = 42):
    """
    Load and prepare permuted MNIST data with train/val/test splits.
    Converted to JAX arrays and placed on device for fast sampling.

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (images, labels) tuples.
        Images are permuted and have shape (N, 784), labels are integers (0-9).
    """
    import numpy as np
    global _PMNIST_CACHE

    cache_key = f"pmnist_{permutation_seed}"
    if cache_key in _PMNIST_CACHE:
        return _PMNIST_CACHE[cache_key]

    # Download MNIST
    mnist = download_mnist()

    # Create fixed permutation ONCE
    if permutation_seed is None:
        permutation = np.arange(784)
    else:
        rng = np.random.RandomState(permutation_seed)
        permutation = rng.permutation(784)

    # Apply permutation to all images ONCE at initialization
    train_images_perm = mnist['train_images'][:, permutation]
    test_images_perm = mnist['test_images'][:, permutation]

    # Create train/val split (50k train, 10k val)
    val_split = 50000
    train_images = train_images_perm[:val_split]
    train_labels = mnist['train_labels'][:val_split]

    val_images = train_images_perm[val_split:]
    val_labels = mnist['train_labels'][val_split:]

    # Test set
    test_images = test_images_perm
    test_labels = mnist['test_labels']

    # Convert to JAX arrays and place on device (critical for speed)
    train_images = jax.device_put(jnp.array(train_images))
    train_labels = jax.device_put(jnp.array(train_labels))
    val_images = jax.device_put(jnp.array(val_images))
    val_labels = jax.device_put(jnp.array(val_labels))
    test_images = jax.device_put(jnp.array(test_images))
    test_labels = jax.device_put(jnp.array(test_labels))

    result = {
        'train': (train_images, train_labels),
        'val': (val_images, val_labels),
        'test': (test_images, test_labels),
        'permutation': permutation
    }

    _PMNIST_CACHE[cache_key] = result

    print(f"Loaded pMNIST dataset (JAX arrays on device):")
    print(f"  Train: {train_images.shape[0]} samples")
    print(f"  Val: {val_images.shape[0]} samples")
    print(f"  Test: {test_images.shape[0]} samples")
    print(f"  Permutation seed: {permutation_seed}")

    return result


@jax.jit
def _sample_batch_pmnist(images, labels, indices):
    """JIT-compiled batch sampler for MNIST."""
    batch_images = images[indices]  # (batch_size, 784)
    batch_labels = labels[indices]  # (batch_size,)

    inputs = batch_images

    # One-hot encode labels (only for last timestep since compute_loss_only_last_timestep=True)
    labels_one_hot = jax.nn.one_hot(batch_labels, 10)  # (batch_size, 10)
    targets = labels_one_hot[:, None, :]  # (batch_size, 1, 10) - only last timestep

    return inputs, targets


def create_pmnist_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = 'train',
    permutated: bool = True,
    permutation_seed: int = 42
) -> Tuple[jax.Array, jax.Array]:
    """
    Sample a batch from the permuted MNIST dataset using jitted sampler.

    Args:
        batch_size: Number of sequences in the batch
        seq_length: Sequence length (should be 784 for MNIST)
        input_size: Input size (should be 1 for MNIST)
        key: JAX random key for sampling
        split: 'train', 'val', or 'test'

    Returns:
        inputs: (batch_size, seq_length, input_size) - pixel sequences
        targets: (batch_size, 1, num_classes) - one-hot labels for last timestep only
    """
    # Load data (already JAX arrays on device)
    pmnist_data = get_pmnist_data(permutation_seed=None if not permutated else permutation_seed)
    images, labels = pmnist_data[split]

    # Sample batch indices
    num_samples = images.shape[0]
    indices = jax.random.permutation(key, num_samples)[:batch_size]

    inputs, targets = _sample_batch_pmnist(images, labels, indices)
    # just need to add the input_size dimension
    inputs = inputs[..., None]  # (batch_size, seq_length, 1)

    # Use jitted sampler
    return inputs, targets


MNIST_TASKS = {
    "pmnist_cmos": {
        "name": "Permuted MNIST (pmnist)",
        "data_fn": lambda b, s, i, k, split='train': create_pmnist_data(b, s, i, k, permutated=True, split=split),
        "description": "Standard permuted MNIST task for RNNs",
        "default_params": {"seq_length": 784, "input_size": 1, "output_size": 10},
        "task_type": "classification",
        'should_split': True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "pmnist": {
        "name": "Permuted MNIST (pmnist)",
        "data_fn": lambda b, s, i, k, split='train': create_pmnist_data(b, s, i, k, permutated=True, split=split),
        "description": "Standard permuted MNIST task for RNNs",
        "default_params": {"seq_length": 784, "input_size": 1, "output_size": 10},
        "task_type": "classification",
        'should_split': True,
        "compute_loss_only_last_timestep": True,
        "compute_accuracy_only_last_timestep": True,
    },
    "pmnist36": {
        "name": "Permuted MNIST (pmnist)",
        "data_fn": lambda b, s, i, k, split='train': create_pmnist_data(b, s, i, k, permutated=True, split=split, permutation_seed=36),
        "description": "Standard permuted MNIST task for RNNs",
        "default_params": {"seq_length": 784, "input_size": 1, "output_size": 10},
        "task_type": "classification",
        'should_split': True,
        "compute_loss_only_last_timestep": True,
        "compute_accuracy_only_last_timestep": True,
    },
    "pmnist24": {
        "name": "Permuted MNIST (pmnist)",
        "data_fn": lambda b, s, i, k, split='train': create_pmnist_data(b, s, i, k, permutated=True, split=split, permutation_seed=24),
        "description": "Standard permuted MNIST task for RNNs",
        "default_params": {"seq_length": 784, "input_size": 1, "output_size": 10},
        "task_type": "classification",
        'should_split': True,
        "compute_loss_only_last_timestep": True,
        "compute_accuracy_only_last_timestep": True,
    },
    "smnist_cmos": {
        "name": "Sequential MNIST (smnist)",
        "data_fn": lambda b, s, i, k, split='train': create_pmnist_data(b, s, i, k, permutated=False, split=split),
        "description": "Standard sequential MNIST task for RNNs",
        "default_params": {"seq_length": 784, "input_size": 1, "output_size": 10},
        "task_type": "classification",
        'should_split': True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "smnist": {
        "name": "Sequential MNIST (smnist)",
        "data_fn": lambda b, s, i, k, split='train': create_pmnist_data(b, s, i, k, permutated=False, split=split),
        "description": "Standard sequential MNIST task for RNNs",
        "default_params": {"seq_length": 784, "input_size": 1, "output_size": 10},
        "task_type": "classification",
        'should_split': True,
        "compute_loss_only_last_timestep": True,
        "compute_accuracy_only_last_timestep": True,
    },
}
