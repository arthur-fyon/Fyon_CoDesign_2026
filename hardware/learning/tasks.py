#!/usr/bin/env python3
"""
Task definitions for keyword spotting with CMOS-SBC RNN.
Uses Google Speech Commands dataset for real audio binary classification.
"""

from typing import Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Import real audio processing
try:
    from google_speech_commands import (
        download_dataset,
        create_binary_dataset,
        DATASET_DIR
    )
    REAL_AUDIO_AVAILABLE = True
except ImportError:
    REAL_AUDIO_AVAILABLE = False
    print("Warning: google_speech_commands module not available.")
    print("Please ensure google_speech_commands.py is in the same directory.")


def create_real_audio_binary_data(
    batch_size: int,
    seq_length: int,
    input_size: int,
    key: jax.Array,
    split: str = "train",
    return_paths: bool = False
) -> Tuple[jax.Array, jax.Array]:
    """
    Load real Google Speech Commands audio for binary classification.
    Detects 'yes' keyword vs other words (no, up, down, left, right) and silence.
    
    Args:
        batch_size: Number of samples in batch
        seq_length: Sequence length (determined by audio, ~101 frames)
        input_size: Input size (13 MFCC coefficients)
        key: JAX random key for batch sampling
        split: Dataset split - 'train', 'validation', or 'test'
        return_paths: If True, also return file paths for the batch
    
    Returns:
        batch_features: (batch_size, seq_length, input_size) MFCC features
        batch_targets: (batch_size, seq_length, 2) one-hot encoded labels
        batch_paths: (optional) list of file paths
    """
    if not REAL_AUDIO_AVAILABLE:
        raise ImportError(
            "google_speech_commands module required for real audio. "
            "Please ensure google_speech_commands.py is available."
        )
    
    # Check if dataset exists, download if needed
    if not DATASET_DIR.exists():
        print("Downloading Google Speech Commands dataset...")
        download_dataset()
    
    # Cache data per split for efficiency
    cache_key = f'_cache_{split}'
    
    if not hasattr(create_real_audio_binary_data, cache_key):
        print(f"Loading real audio data ({split} set, this may take a minute)...")
        
        # Use more samples for training
        if split == "train":
            max_samples = 4000
        else:
            max_samples = 500
        
        # Load and cache the dataset
        features, labels, paths = create_binary_dataset(
            positive_word="yes",
            max_positive=max_samples,
            max_negative=max_samples,
            split=split,
            return_paths=True
        )
        setattr(create_real_audio_binary_data, cache_key, (features, labels, paths))
    
    features, labels, paths = getattr(create_real_audio_binary_data, cache_key)
    
    # Sample random batch
    num_examples = len(features)
    indices = jax.random.choice(key, num_examples, (batch_size,), replace=False)
    
    batch_features = jnp.array(features[indices])
    batch_labels = jnp.array(labels[indices])
    
    # Convert to one-hot encoding
    batch_targets = jax.nn.one_hot(batch_labels, 2)
    
    # Expand to sequence (repeat label for all timesteps)
    batch_targets = jnp.expand_dims(batch_targets, axis=1)
    batch_targets = jnp.repeat(batch_targets, batch_features.shape[1], axis=1)
    
    if return_paths:
        batch_paths = [paths[int(i)] for i in indices]
        return batch_features, batch_targets, batch_paths
    
    return batch_features, batch_targets


# Task registry
BENCHMARK_TASKS = {
    "real_audio_binary": {
        "name": "Binary Keyword Spotting (REAL AUDIO)",
        "data_fn": create_real_audio_binary_data,
        "description": "Detect 'yes' vs other words using Google Speech Commands audio",
        "default_params": {
            "seq_length": 101,   # ~1 second at 100 fps
            "input_size": 13,    # 13 MFCC coefficients
            "output_size": 2     # Binary: yes vs not-yes
        },
        "task_type": "classification",
        "hardware_specs": {
            "recommended_cells": 2,
            "state_dim": 32,
            "difficulty": "medium",
            "memory_needed": "temporal pattern recognition",
            "application": "Wake word detection",
            "data_type": "real",
            "dataset": "Google Speech Commands v0.02",
            "expected_accuracy": "90-95%"
        }
    },
}


def print_task_info():
    """Print information about available tasks."""
    print("=" * 70)
    print("KEYWORD SPOTTING TASK FOR CMOS-SBC RNN")
    print("=" * 70)
    
    if not REAL_AUDIO_AVAILABLE:
        print("\n⚠️  Real audio not available!")
        print("To enable:")
        print("  1. pip install librosa soundfile")
        print("  2. Ensure google_speech_commands.py is in the same directory")
        return
    
    for task_name, task_info in BENCHMARK_TASKS.items():
        specs = task_info["hardware_specs"]
        params = task_info["default_params"]
        
        print(f"\nTask: {task_name}")
        print(f"  Name: {task_info['name']}")
        print(f"  Description: {task_info['description']}")
        print(f"  Sequence Length: {params['seq_length']} frames (~1 second)")
        print(f"  Input Size: {params['input_size']} (MFCC coefficients)")
        print(f"  Output Size: {params['output_size']} classes")
        print(f"  Dataset: {specs.get('dataset', 'N/A')}")
        print(f"  Expected Accuracy: {specs.get('expected_accuracy', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("Quick Start:")
    print("  python main.py train --task real_audio_binary --config config.json")
    print("=" * 70)


if __name__ == "__main__":
    print_task_info()
