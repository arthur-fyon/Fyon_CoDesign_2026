"""Long Range Arena benchmark tasks: ListOps and IMDb (byte-level)."""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.byte_encoder import ByteEncoder


# ============================= LISTOPS DATASET ==============================

def load_listops_data(data_dir: str = "./data/lra_release/listops-1000"):
    """
    Load ListOps dataset from TSV files.
    Data is kept on CPU (numpy arrays) to save GPU memory.
    Only batches are moved to GPU during sampling.

    ListOps is a hierarchical reasoning task with operators:
    - [MAX, [MIN, [MED, [SM (operators with bracket prefix)
    - Numbers 0-9
    - Brackets: (, ), ]

    Returns:
        Dictionary with 'train', 'val', 'test' splits containing (sequences, labels) tuples.
        All data is stored as numpy arrays on CPU.
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    data_dir = Path(data_dir)

    # Define vocabulary for tokenization - CORRECTED
    # The operators in the actual data include the bracket prefix
    vocab = {
        'PAD': 0,
        ']': 1,  # Standalone brackets ([ is always part of operators)
        '[MAX': 2, '[MIN': 3, '[MED': 4, '[SM': 5,  # Operators with bracket
        '0': 6, '1': 7, '2': 8, '3': 9, '4': 10,
        '5': 11, '6': 12, '7': 13, '8': 14, '9': 15
    }

    vocab_size = len(vocab)

    def tokenize_sequence(sequence_str: str, max_length: int = 2000):
        """Tokenize a ListOps sequence string."""
        tokens = sequence_str.strip().split()
        # remove '(' and ')' from tokens if they are standalone
        processed_tokens = [token for token in tokens if token not in ('(', ')')]
        token_ids = [vocab.get(token, vocab['PAD']) for token in processed_tokens]

        if len(token_ids) < max_length:
            token_ids = token_ids + [vocab['PAD']] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
            print("=!= Warning: Sequence truncated to max_length =!=", flush=True)

        return np.array(token_ids, dtype=np.int32)

    def load_split(filepath):
        """Load a single split (train/val/test)."""
        df = pd.read_csv(filepath, sep='\t')

        sequences = []
        labels = []

        for _, row in df.iterrows():
            # Tokenize source sequence
            seq = tokenize_sequence(row['Source'])
            sequences.append(seq)

            # Target is a single digit 0-9 (10-way classification)
            labels.append(int(row['Target']))

        return np.array(sequences), np.array(labels)

    # Load all splits
    train_seqs, train_labels = load_split(data_dir / "basic_train.tsv")
    val_seqs, val_labels = load_split(data_dir / "basic_val.tsv")
    test_seqs, test_labels = load_split(data_dir / "basic_test.tsv")

    # Print dataset summary before padding
    print("\n" + "="*80, flush=True)
    print("ListOps Dataset Summary (Before Padding)", flush=True)
    print("="*80, flush=True)

    def analyze_split(seqs, labels, split_name):
        """Analyze and print statistics for a dataset split."""
        # Calculate sequence lengths (before padding)
        seq_lengths = np.array([np.sum(seq != vocab['PAD']) for seq in seqs])

        print(f"\n{split_name} Split:", flush=True)
        print(f"  Number of samples: {len(seqs)}", flush=True)
        print(f"  Sequence length distribution (before padding):", flush=True)
        print(f"    Min: {seq_lengths.min()}", flush=True)
        print(f"    Max: {seq_lengths.max()}", flush=True)
        print(f"    Mean: {seq_lengths.mean():.2f}", flush=True)
        print(f"    Median: {np.median(seq_lengths):.2f}", flush=True)
        print(f"    Std: {seq_lengths.std():.2f}", flush=True)

        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"    Percentiles:")
        for p in percentiles:
            print(f"      {p}th: {np.percentile(seq_lengths, p):.0f}", flush=True)

        # Target distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"  Target token distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = 100 * count / len(labels)
            print(f"    Class {label}: {count:6d} ({percentage:5.2f}%)", flush=True)

    analyze_split(train_seqs, train_labels, "Train")
    analyze_split(val_seqs, val_labels, "Validation")
    analyze_split(test_seqs, test_labels, "Test")

    print("\n" + "="*80 + "\n")

    # KEEP DATA ON CPU - DO NOT use jax.device_put()
    # Store as regular numpy arrays to save GPU memory
    result = {
        'train': (train_seqs, train_labels),
        'val': (val_seqs, val_labels),
        'test': (test_seqs, test_labels),
        'vocab': vocab,
        'vocab_size': vocab_size
    }

    print(f"Loaded ListOps dataset (numpy arrays on CPU - memory efficient):", flush=True)
    print(f"  Train: {train_seqs.shape[0]} samples, seq_len: {train_seqs.shape[1]}", flush=True)
    print(f"  Val: {val_seqs.shape[0]} samples", flush=True)
    print(f"  Test: {test_seqs.shape[0]} samples", flush=True)
    print(f"  Vocab size: {vocab_size}", flush=True)
    print(f"  Memory strategy: Data on CPU, batches moved to GPU on demand", flush=True)

    return result


# Global cache for ListOps data
_LISTOPS_CACHE = {}


def get_listops_data(data_dir: str = "./data/lra_release/listops-1000"):
    """Get cached ListOps data."""
    global _LISTOPS_CACHE

    if data_dir not in _LISTOPS_CACHE:
        _LISTOPS_CACHE[data_dir] = load_listops_data(data_dir)

    return _LISTOPS_CACHE[data_dir]


@jax.jit
def _sample_batch_listops_gpu(batch_seqs, batch_labels):
    """
    JIT-compiled function to process a batch that's already on GPU.
    This is separated to allow JIT compilation while keeping data on CPU.

    Args:
        batch_seqs: (batch_size, seq_length) - already on GPU
        batch_labels: (batch_size,) - already on GPU

    Returns:
        inputs: (batch_size, seq_length, vocab_size) - one-hot encoded
        targets: (batch_size, 1, 10) - one-hot labels
    """
    vocab_size = 16

    # Convert token IDs to one-hot encoding
    # Shape: (batch_size, seq_length, vocab_size)
    inputs = jax.nn.one_hot(batch_seqs, vocab_size)

    # One-hot encode labels - SINGLE TARGET (last timestep only)
    labels_one_hot = jax.nn.one_hot(batch_labels, 10)  # (batch_size, 10)
    targets = labels_one_hot[:, None, :]  # (batch_size, 1, 10)

    return inputs, targets


def create_listops_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = 'train'
) -> Tuple[jax.Array, jax.Array]:
    """
    Sample a batch from the ListOps dataset.

    This implementation keeps data on CPU and only moves the sampled batch to GPU.
    This is memory-efficient for large datasets.

    ListOps is a hierarchical reasoning task requiring the model to:
    1. Parse nested expressions with operators ([MAX, [MIN, [MED, [SM)
    2. Apply operators recursively on operands
    3. Output the final computed value (0-9)

    Example:
        Input:  [MAX 4 3 [MIN 2 3] 1 0 [MED 1 5 8 9 2]]
        Output: 5

    Args:
        batch_size: Number of sequences in the batch
        seq_length: Sequence length (should match data, typically ~2000)
        input_size: Input size (should be vocab_size=16)
        key: JAX random key for sampling
        split: 'train', 'val', or 'test'

    Returns:
        inputs: (batch_size, seq_length, vocab_size) - one-hot encoded sequences
        targets: (batch_size, 1, 10) - one-hot labels for last timestep only
    """
    import numpy as np

    # Load data (cached, stored as numpy arrays on CPU)
    listops_data = get_listops_data()
    sequences, labels = listops_data[split]  # These are numpy arrays on CPU

    # Sample batch indices on CPU
    num_samples = sequences.shape[0]
    indices = np.array(jax.random.permutation(key, num_samples)[:batch_size])

    # Extract batch data on CPU (numpy)
    batch_seqs_cpu = sequences[indices]  # (batch_size, seq_length)
    batch_labels_cpu = labels[indices]   # (batch_size,)

    # Move ONLY the batch to GPU
    batch_seqs_gpu = jax.device_put(jnp.array(batch_seqs_cpu))
    batch_labels_gpu = jax.device_put(jnp.array(batch_labels_cpu))

    # Process batch on GPU (JIT-compiled)
    inputs, targets = _sample_batch_listops_gpu(batch_seqs_gpu, batch_labels_gpu)

    return inputs, targets


# ============================== IMDB DATASET ==============================

# Global cache for IMDb data
_IMDB_CACHE = {}


def load_imdb_data(data_dir: str = "./data/lra_release/imdb", max_length: int = 4096):
    """
    Load and prepare IMDb dataset with train/val/test splits.
    Data is kept on CPU (numpy arrays) to save GPU memory.
    Only batches are moved to GPU during sampling.

    Args:
        data_dir: Path to directory containing IMDB_dataset.csv
        max_length: Maximum sequence length for byte encoding

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (sequences, labels) tuples.
        All data is stored as numpy arrays on CPU.
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    data_dir = Path(data_dir)
    csv_path = data_dir / "IMDB_dataset.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"IMDb CSV file not found at {csv_path}. "
            f"Please ensure the file exists at this location."
        )

    print(f"Loading IMDb from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Check for required columns
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError(
            f"CSV must contain 'review' and 'sentiment' columns. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Create byte encoder
    encoder = ByteEncoder(max_length=max_length)

    # Encode all reviews
    print("Encoding reviews as byte sequences...")
    encoded_reviews = encoder.encode_batch(df['review'].tolist())

    # Encode labels (positive=1, negative=0)
    labels = (df['sentiment'] == 'positive').astype(np.int32).values

    # Create train/val/test splits (80/10/10)
    num_samples = len(encoded_reviews)
    num_train = int(0.8 * num_samples)
    num_val = int(0.1 * num_samples)

    # Shuffle data with fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    train_sequences = encoded_reviews[train_indices]
    train_labels = labels[train_indices]

    val_sequences = encoded_reviews[val_indices]
    val_labels = labels[val_indices]

    test_sequences = encoded_reviews[test_indices]
    test_labels = labels[test_indices]

    # KEEP DATA ON CPU - DO NOT use jax.device_put()
    # Store as regular numpy arrays to save GPU memory
    result = {
        'train': (train_sequences, train_labels),
        'val': (val_sequences, val_labels),
        'test': (test_sequences, test_labels),
        'vocab_size': 256  # Bytes are 0-255
    }

    print(f"Loaded IMDb dataset (numpy arrays on CPU - memory efficient):")
    print(f"  Train: {train_sequences.shape[0]} samples, seq_len: {train_sequences.shape[1]}")
    print(f"  Val: {val_sequences.shape[0]} samples")
    print(f"  Test: {test_sequences.shape[0]} samples")
    print(f"  Vocab size: 256 (byte-level)")
    print(f"  Memory strategy: Data on CPU, batches moved to GPU on demand")

    return result


def get_imdb_data(data_dir: str = "./data/lra_release/imdb", max_length: int = 4096):
    """Get cached IMDb data."""
    global _IMDB_CACHE

    cache_key = f"imdb_{max_length}"
    if cache_key not in _IMDB_CACHE:
        _IMDB_CACHE[cache_key] = load_imdb_data(data_dir, max_length)

    return _IMDB_CACHE[cache_key]


@jax.jit
def _sample_batch_imdb_gpu(batch_seqs, batch_labels):
    """
    JIT-compiled function to process a batch that's already on GPU.
    This is separated to allow JIT compilation while keeping data on CPU.

    Args:
        batch_seqs: (batch_size, seq_length) - already on GPU
        batch_labels: (batch_size,) - already on GPU

    Returns:
        inputs: (batch_size, seq_length, vocab_size) - one-hot encoded
        targets: (batch_size, 1, 2) - one-hot labels
    """
    vocab_size = 256

    # Convert byte IDs to one-hot encoding
    inputs = jax.nn.one_hot(batch_seqs, vocab_size)

    # One-hot encode labels
    labels_one_hot = jax.nn.one_hot(batch_labels, 2)
    targets = labels_one_hot[:, None, :]

    return inputs, targets


def create_imdb_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = 'train',
    max_length: int = 4096
) -> Tuple[jax.Array, jax.Array]:
    """
    Sample a batch from the IMDb dataset for byte-level text classification.

    This implementation keeps data on CPU and only moves the sampled batch to GPU.
    This is memory-efficient for large datasets.

    This is a binary sentiment classification task using character-level inputs,
    which tests the model's ability to:
    1. Compose characters into words and phrases
    2. Capture semantic meaning over long contexts
    3. Handle compositional structure without explicit boundaries

    Args:
        batch_size: Number of sequences in the batch
        seq_length: Sequence length (should match data, typically 4096)
        input_size: Input size (should be vocab_size=256 for byte-level)
        key: JAX random key for sampling
        split: 'train', 'val', or 'test'
        max_length: Maximum sequence length (default 4096)

    Returns:
        inputs: (batch_size, seq_length, vocab_size) - one-hot encoded byte sequences
        targets: (batch_size, 1, 2) - one-hot labels for last timestep only
    """
    import numpy as np

    # Load data (cached, stored as numpy arrays on CPU)
    imdb_data = get_imdb_data(max_length=max_length)
    sequences, labels = imdb_data[split]  # These are numpy arrays on CPU

    # Sample batch indices on CPU
    num_samples = sequences.shape[0]
    indices = np.array(jax.random.permutation(key, num_samples)[:batch_size])

    # Extract batch data on CPU (numpy)
    batch_seqs_cpu = sequences[indices]  # (batch_size, seq_length)
    batch_labels_cpu = labels[indices]   # (batch_size,)

    # Move ONLY the batch to GPU
    batch_seqs_gpu = jax.device_put(jnp.array(batch_seqs_cpu))
    batch_labels_gpu = jax.device_put(jnp.array(batch_labels_cpu))

    # Process batch on GPU (JIT-compiled)
    inputs, targets = _sample_batch_imdb_gpu(batch_seqs_gpu, batch_labels_gpu)

    return inputs, targets


# ============================ PATHFINDER DATASET ============================

# Global cache for Pathfinder data
_PATHFINDER_CACHE = {}

# Normalisation statistics (computed from Pathfinder-32 training set)
_PATHFINDER_MU = 0.057706069201231
_PATHFINDER_SIGMA = 0.17094121873378754


def load_pathfinder_npz(npz_path: str):
    """Load Pathfinder dataset from NPZ file.

    Expects a pre-converted NPZ with keys:
        train_images, train_labels, val_images, val_labels,
        test_images, test_labels, resolution
    Images are uint8 of shape (N, seq_length) (already flattened).
    """
    from pathlib import Path

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Pathfinder NPZ file not found at {npz_path}. "
            f"Please run the conversion script first:\n"
            f"  python working_utils/convert_pathfinder_to_npz.py "
            f"--resolution {npz_path.stem[-2:]} --data_dir data/lra_release/"
        )

    print(f"Loading Pathfinder from {npz_path}...")
    data = np.load(npz_path)

    # Convert uint8 to float32, normalize to [0, 1], then standardize
    def prep(arr):
        return ((arr.astype(np.float32) / 255.0) - _PATHFINDER_MU) / _PATHFINDER_SIGMA

    train_images = prep(data['train_images'])
    val_images   = prep(data['val_images'])
    test_images  = prep(data['test_images'])

    resolution = int(data['resolution'])

    print(f"  Train: {len(train_images)} samples")
    print(f"  Val:   {len(val_images)} samples")
    print(f"  Test:  {len(test_images)} samples")
    print(f"  Resolution: {resolution}x{resolution}")

    # Keep on CPU — only batches are moved to GPU during sampling
    return {
        'train': (train_images,              data['train_labels']),
        'val':   (val_images,                data['val_labels']),
        'test':  (test_images,               data['test_labels']),
        'resolution': resolution,
    }


def get_pathfinder_data(data_dir: str = "./data/lra_release", resolution: int = 32):
    """Get cached Pathfinder data (numpy arrays on CPU)."""
    global _PATHFINDER_CACHE

    cache_key = f"pathfinder{resolution}"
    if cache_key not in _PATHFINDER_CACHE:
        from pathlib import Path
        npz_path = Path(data_dir) / f"pathfinder{resolution}.npz"
        _PATHFINDER_CACHE[cache_key] = load_pathfinder_npz(str(npz_path))

    return _PATHFINDER_CACHE[cache_key]


@jax.jit
def _sample_batch_pathfinder_gpu(batch_images, batch_labels):
    """JIT-compiled batch processor for Pathfinder.

    Args:
        batch_images: (batch_size, seq_length) float32, already on GPU
        batch_labels: (batch_size,) int32, already on GPU

    Returns:
        inputs:  (batch_size, seq_length, 1)
        targets: (batch_size, 1, 2)
    """
    inputs = batch_images[..., None]                   # (B, L, 1)
    labels_one_hot = jax.nn.one_hot(batch_labels, 2)  # (B, 2)
    targets = labels_one_hot[:, None, :]               # (B, 1, 2)
    return inputs, targets


def create_pathfinder_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = 'train',
    resolution: int = 32,
) -> Tuple[jax.Array, jax.Array]:
    """Sample a batch from the Pathfinder dataset.

    Data is kept on CPU; only the sampled batch is moved to GPU.

    Args:
        batch_size: Number of images per batch
        seq_length: Flattened image length (1024 for 32x32, 16384 for 128x128)
        input_size: Should be 1 (grayscale pixel)
        key: JAX random key
        split: 'train', 'val', or 'test'
        resolution: 32 or 128

    Returns:
        inputs:  (batch_size, seq_length, 1)
        targets: (batch_size, 1, 2)
    """
    pathfinder_data = get_pathfinder_data(resolution=resolution)
    images, labels = pathfinder_data[split]  # numpy arrays on CPU

    num_samples = images.shape[0]
    indices = np.array(jax.random.permutation(key, num_samples)[:batch_size])

    batch_images_gpu = jax.device_put(jnp.array(images[indices]))
    batch_labels_gpu = jax.device_put(jnp.array(labels[indices]))

    return _sample_batch_pathfinder_gpu(batch_images_gpu, batch_labels_gpu)


LRA_TASKS = {
    "listops": {
        "name": "ListOps (Long Range Arena)",
        "data_fn": lambda b, s, i, k, split='train': create_listops_data(b, s, i, k, split=split),
        "description": "Hierarchical reasoning with nested operators",
        "default_params": {"seq_length": 2000, "input_size": 16, "output_size": 10},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
        "is_padded": True,
        "padding_value": 0,
    },
    "imdb": {
        "name": "IMDb Sentiment (Byte-Level)",
        "data_fn": lambda b, s, i, k, split='train': create_imdb_data(b, s, i, k, split=split, max_length=4096),
        "description": "Binary sentiment classification on IMDb reviews at byte level",
        "default_params": {"seq_length": 4096, "input_size": 256, "output_size": 2},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "pathfinder32": {
        "name": "Pathfinder-32 (LRA)",
        "data_fn": lambda b, s, i, k, split='train': create_pathfinder_data(b, s, i, k, split=split, resolution=32),
        "description": "Binary classification: are two circles connected by a path? (32x32 images, seq_length=1024)",
        "default_params": {"seq_length": 1024, "input_size": 1, "output_size": 2},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "pathfinder128": {
        "name": "Pathfinder-128 / Path-X (LRA)",
        "data_fn": lambda b, s, i, k, split='train': create_pathfinder_data(b, s, i, k, split=split, resolution=128),
        "description": "Binary classification: are two circles connected by a path? (128x128 images, seq_length=16384)",
        "default_params": {"seq_length": 16384, "input_size": 1, "output_size": 2},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
}
