"""Audio benchmark tasks using Google Speech Commands dataset."""

from typing import Tuple
import functools

import jax
import jax.numpy as jnp
import numpy as np

try:
    from benchmarks.google_speech_commands import (
        download_dataset,
        create_binary_dataset,
        create_multiclass_dataset,
        DATASET_DIR
    )
    REAL_AUDIO_AVAILABLE = True
except ImportError:
    REAL_AUDIO_AVAILABLE = False
    print("Note: Real audio not available. Install google_speech_commands dependencies.")


_AUDIO_CACHE = {}


def get_audio_data(dataset_name: str, split: str, loader_fn):
    global _AUDIO_CACHE
    cache_key = f"{dataset_name}_{split}"
    if cache_key in _AUDIO_CACHE:
        return _AUDIO_CACHE[cache_key]

    print(f"Loading {dataset_name} audio data ({split} set)...")
    features_np, labels_np = loader_fn()

    # CRITICAL: Delete NumPy references BEFORE creating JAX arrays
    import gc
    del loader_fn
    gc.collect()

    # Create JAX arrays directly from copies of NumPy data
    features_jax = jax.device_put(jnp.array(np.copy(features_np)))
    labels_jax = jax.device_put(jnp.array(np.copy(labels_np)))

    # CRITICAL: Delete NumPy arrays immediately after copying
    del features_np, labels_np
    gc.collect()

    result = (features_jax, labels_jax)
    _AUDIO_CACHE[cache_key] = result

    print(f"Features shape: {features_jax.shape}, Labels shape: {labels_jax.shape}", flush=True)

    return result


@functools.partial(jax.jit, static_argnames=['num_classes'])
def _sample_batch_audio(features, labels, indices, num_classes):
    """
    JIT-compiled batch sampler for audio data (like _sample_batch_pmnist).

    All operations are on JAX arrays on GPU - no numpy mixing!

    Args:
        features: (N, seq_length, num_mfcc) - all features on GPU
        labels: (N,) - all labels on GPU
        indices: (batch_size,) - batch indices
        num_classes: Number of output classes

    Returns:
        batch_features: (batch_size, seq_length, num_mfcc)
        batch_targets: (batch_size, seq_length, num_classes) - repeated for all timesteps
    """
    # Sample batch (JAX indexing on JAX arrays)
    batch_features = features[indices]
    batch_labels = labels[indices]

    # Convert to one-hot
    batch_targets = jax.nn.one_hot(batch_labels, num_classes)

    batch_targets = batch_targets[:, None, :]  # Add time dimension

    return batch_features, batch_targets


@functools.partial(jax.jit, static_argnames=['num_classes'])
def _sample_batch_kws_in_noise(noise_features, word_features, word_labels,
                                noise_indices, keyword_indices, keyword_starts,
                                num_classes):
    """
    Build a KWS-in-noise batch.

    Each sequence is a full-length background noise clip with one keyword
    clip inserted at a random start position.

    Args:
        noise_features:  (N_noise, seq_length, 13) - pre-extracted noise MFCC
        word_features:   (N_word, 101, 13)
        word_labels:     (N_word,)
        noise_indices:   (B,)  - index into noise_features
        keyword_indices: (B,)  - index into word_features
        keyword_starts:  (B,)  - frame index where the keyword starts

    Returns:
        batch_x: (B, seq_length, 13)
        batch_y: (B, 1, num_classes)
    """
    noise_seqs = noise_features[noise_indices]    # (B, seq_length, 13)
    kw_clips   = word_features[keyword_indices]   # (B, 101, 13)
    kw_labels  = word_labels[keyword_indices]     # (B,)

    def insert(noise_seq, kw_clip, start):
        return jax.lax.dynamic_update_slice(noise_seq, kw_clip, (start, 0))

    batch_x = jax.vmap(insert)(noise_seqs, kw_clips, keyword_starts)
    batch_y = jax.nn.one_hot(kw_labels, num_classes)[:, None, :]
    return batch_x, batch_y


# Yes versus others
def create_real_audio_binary_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = "train"
) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-optimized binary classification (yes vs others).

    Returns:
        batch_features: (batch_size, 101, 13) - MFCC features
        batch_targets: (batch_size, 101, 2) - one-hot labels
    """
    if not REAL_AUDIO_AVAILABLE:
        raise ImportError("google_speech_commands module required for real audio")

    if not DATASET_DIR.exists():
        print("Downloading Google Speech Commands dataset...")
        download_dataset()

    # Load cached data (already on GPU as JAX arrays)
    def loader():
        max_samples = 4000 if split == "train" else 500
        return create_binary_dataset(
            positive_word="yes",
            max_positive=max_samples,
            max_negative=max_samples,
            split=split,
            return_paths=False
        )

    features, labels = get_audio_data("binary", split, loader)

    # Sample batch indices (on GPU)
    num_examples = features.shape[0]
    indices = jax.random.choice(key, num_examples, (batch_size,),
                                replace=(batch_size > num_examples))

    # Use JIT-compiled sampler (all JAX operations)
    return _sample_batch_audio(features, labels, indices, num_classes=2)


# Digits
def create_real_audio_digits_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = "train"
) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-optimized digit recognition (0-9 + silence).

    Args:
        batch_size: Number of samples in batch
        seq_length: Sequence length (unused, determined by audio = 101 frames)
        input_size: Input size (unused, determined by MFCC = 13 features)
        key: JAX random key
        split: 'train', 'validation', or 'test'

    Returns:
        batch_features: (batch_size, 101, 13) - MFCC features
        batch_targets: (batch_size, 101, 11) - one-hot labels
                       Classes: [silence, zero, one, two, three, four, five, six, seven, eight, nine]
    """
    if not REAL_AUDIO_AVAILABLE:
        raise ImportError("google_speech_commands module required for real audio")

    if not DATASET_DIR.exists():
        print("Downloading Google Speech Commands dataset...")
        download_dataset()

    # Load cached data (already on GPU as JAX arrays)
    def loader():
        max_samples = 4000 if split == "train" else 1000
        return create_multiclass_dataset(
            words=["_silence_", "zero", "one", "two", "three", "four",
                   "five", "six", "seven", "eight", "nine"],
            max_per_word=max_samples,
            split=split
        )

    features, labels = get_audio_data("digits", split, loader)

    # Sample batch indices (on GPU)
    num_examples = features.shape[0]
    indices = jax.random.choice(key, num_examples, (batch_size,),
                                replace=(batch_size > num_examples))

    # Use JIT-compiled sampler (all JAX operations)
    return _sample_batch_audio(features, labels, indices, num_classes=11)


# All 30 words from Google Speech Commands dataset
# 20 core command words + 10 auxiliary words
ALL_SPEECH_COMMANDS_WORDS = [
    # 20 core command words (most speakers said each 5 times)
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    # 10 auxiliary words (most speakers said once)
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"
]


# All
def create_real_audio_all_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = "train"
) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-optimized classification of all 30 Google Speech Commands words.

    Includes all 20 core command words and 10 auxiliary words:
    - Core words (20): yes, no, up, down, left, right, on, off, stop, go,
                       zero, one, two, three, four, five, six, seven, eight, nine
    - Auxiliary words (10): bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow

    Args:
        batch_size: Number of samples in batch
        seq_length: Sequence length (unused, determined by audio = 101 frames)
        input_size: Input size (unused, determined by MFCC = 13 features)
        key: JAX random key
        split: 'train', 'validation', or 'test'

    Returns:
        batch_features: (batch_size, 101, 13) - MFCC features
        batch_targets: (batch_size, 101, 30) - one-hot labels for 30 classes
    """
    if not REAL_AUDIO_AVAILABLE:
        raise ImportError("google_speech_commands module required for real audio")

    if not DATASET_DIR.exists():
        print("Downloading Google Speech Commands dataset...")
        download_dataset()

    # Load cached data (already on GPU as JAX arrays)
    def loader():
        max_samples = 2000 if split == "train" else 500
        return create_multiclass_dataset(
            words=ALL_SPEECH_COMMANDS_WORDS,
            max_per_word=max_samples,
            split=split
        )

    features, labels = get_audio_data("all_words", split, loader)

    # Sample batch indices (on GPU)
    num_examples = features.shape[0]
    indices = jax.random.choice(key, num_examples, (batch_size,),
                                replace=(batch_size > num_examples))

    # Use JIT-compiled sampler (all JAX operations)
    return _sample_batch_audio(features, labels, indices, num_classes=30)


_KWS_KEYWORD_FRAMES = 101   # MFCC frames for one 1-second word clip


def create_kws_all_in_noise_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = "train",
) -> tuple:
    """
    KWS-in-noise: 30 words embedded at a random position in a background noise sequence.

    One keyword clip (101 frames) is placed at a uniformly random start frame
    within a full-length background noise sequence (seq_length frames).
    Classification target is the keyword class label at the last timestep only.

    Args:
        batch_size: Number of samples in batch
        seq_length: Total sequence length in MFCC frames (default 1024 ≈ 10.2 s)
        input_size: Unused (fixed at 13 MFCC coefficients)
        key:        JAX random key
        split:      'train', 'validation', or 'test'

    Returns:
        batch_features: (batch_size, seq_length, 13)
        batch_targets:  (batch_size, 1, 30)
    """
    if not REAL_AUDIO_AVAILABLE:
        raise ImportError("google_speech_commands module required for real audio")

    if not DATASET_DIR.exists():
        print("Downloading Google Speech Commands dataset...")
        download_dataset()

    # 1. Load / cache word features (101-frame clips)
    def word_loader():
        max_samples = 2000 if split == "train" else 500
        return create_multiclass_dataset(
            words=ALL_SPEECH_COMMANDS_WORDS,
            max_per_word=max_samples,
            split=split
        )
    word_feats, word_labels = get_audio_data("all_words", split, word_loader)

    # 2. Load / cache background noise clips (seq_length frames each)
    # Cache key includes seq_length so different lengths don't collide
    noise_cache_key = f"background_noise_{seq_length}"

    def noise_loader():
        from benchmarks.google_speech_commands import load_background_noise_clips
        num = 4000 if split == "train" else 500
        feats = load_background_noise_clips(num_samples=num, seq_length=seq_length, seed=42)
        return feats, np.zeros(num, dtype=np.int32)
    noise_feats, _ = get_audio_data(noise_cache_key, split, noise_loader)

    # 3. Sample indices and keyword start positions
    N_word  = word_feats.shape[0]
    N_noise = noise_feats.shape[0]
    max_start = seq_length - _KWS_KEYWORD_FRAMES   # keyword fits within the sequence
    k1, k2, k3 = jax.random.split(key, 3)

    noise_indices   = jax.random.randint(k1, (batch_size,), 0, N_noise)
    keyword_indices = jax.random.randint(k2, (batch_size,), 0, N_word)
    keyword_starts  = jax.random.randint(k3, (batch_size,), 0, max_start)

    return _sample_batch_kws_in_noise(
        noise_feats, word_feats, word_labels,
        noise_indices, keyword_indices, keyword_starts,
        num_classes=30,
    )


AUDIO_TASKS = {
    "real_audio_binary": {
        "name": "Binary Keyword Spotting (YES Detection)",
        "data_fn": lambda b, s, i, k, split='train': create_real_audio_binary_data(b, s, i, k, split=split),
        "description": "Detect 'yes' vs others using real Google Speech Commands audio",
        "default_params": {"seq_length": 101, "input_size": 13, "output_size": 2},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "real_audio_binary_quantized": {
        "name": "Binary Keyword Spotting (YES Detection)",
        "data_fn": lambda b, s, i, k, split='train': create_real_audio_binary_data(b, s, i, k, split=split),
        "description": "Detect 'yes' vs others using real Google Speech Commands audio",
        "default_params": {"seq_length": 101, "input_size": 13, "output_size": 2},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "real_audio_digits": {
        "name": "Digit Recognition (0-9 + Silence)",
        "data_fn": lambda b, s, i, k, split='train': create_real_audio_digits_data(b, s, i, k, split=split),
        "description": "11-way classification: recognize digits 0-9 + silence from real audio",
        "default_params": {"seq_length": 101, "input_size": 13, "output_size": 11},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "real_audio_all": {
        "name": "Full Speech Commands (30 Classes)",
        "data_fn": lambda b, s, i, k, split='train': create_real_audio_all_data(b, s, i, k, split=split),
        "description": "30-way classification: 20 core commands (yes, no, up, down, left, right, on, off, stop, go, 0-9) + 10 auxiliary words (bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow)",
        "default_params": {"seq_length": 101, "input_size": 13, "output_size": 30},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss": True,
        "use_mean_pooling_accuracy": True,
    },
    "kws_all_in_noise": {
        "name": "KWS All-30 in Noise (1024 frames)",
        "data_fn": lambda b, s, i, k, split='train': create_kws_all_in_noise_data(b, s, i, k, split=split),
        "description": (
            "30-way KWS: one keyword (101 frames) embedded at a random position in a "
            "1024-frame background noise sequence (~10.2 s). "
            "Noise = random crop from _background_noise_. "
            "Classify the keyword class at the last timestep only."
        ),
        "default_params": {"seq_length": 1024, "input_size": 13, "output_size": 30},
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep": True,
        "compute_accuracy_only_last_timestep": True,
    },
}
