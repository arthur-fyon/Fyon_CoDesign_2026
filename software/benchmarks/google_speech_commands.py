#!/usr/bin/env python3
"""
Google Speech Commands dataset loader and MFCC feature extractor.
Downloads and processes real audio for keyword spotting.
Properly handles train/validation/test splits using official split files.
IMPROVED: Balances to maximum real audio count to preserve all data.
"""

from pathlib import Path
from typing import Tuple, List, Set
import tarfile
import urllib.request

import numpy as np


# Try to import audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Install with: pip install librosa")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available. Install with: pip install soundfile")


# Dataset configuration
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_DIR = Path("./data/speech_commands_v0.02")

# Audio configuration
SAMPLE_RATE = 16000  # 16kHz
AUDIO_LENGTH = 1.0   # 1 second clips
NUM_MFCC = 13        # Number of MFCC coefficients
HOP_LENGTH = 160     # 10ms hop (16000 / 160 = 100 fps)
N_FFT = 512         # FFT window size


def load_split_files() -> Tuple[Set[str], Set[str]]:
    """
    Load the official train/validation/test split files.
    
    Returns:
        validation_files: Set of filenames in validation set
        test_files: Set of filenames in test set
    """
    validation_files = set()
    test_files = set()
    
    validation_file = DATASET_DIR / "validation_list.txt"
    test_file = DATASET_DIR / "testing_list.txt"
    
    if validation_file.exists():
        with open(validation_file, 'r') as f:
            validation_files = {line.strip() for line in f if line.strip()}
    
    if test_file.exists():
        with open(test_file, 'r') as f:
            test_files = {line.strip() for line in f if line.strip()}
    
    return validation_files, test_files


def download_dataset(force_download: bool = False):
    """
    Download Google Speech Commands dataset.
    
    Args:
        force_download: If True, re-download even if exists
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required. Install with: pip install librosa soundfile")
    
    if DATASET_DIR.exists() and not force_download:
        print(f"Dataset already exists at {DATASET_DIR}")
        return
    
    print("Downloading Google Speech Commands dataset...")
    print(f"URL: {DATASET_URL}")
    print(f"Size: ~2GB (this will take a few minutes)")
    
    # Create data directory
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    tar_path = DATASET_DIR.parent / "speech_commands_v0.02.tar.gz"
    
    if not tar_path.exists() or force_download:
        print("Downloading...")
        urllib.request.urlretrieve(DATASET_URL, tar_path)
        print("Download complete!")
    
    # Extract
    print("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(DATASET_DIR)
    
    print(f"Dataset ready at {DATASET_DIR}")


def get_word_list() -> List[str]:
    """Get list of available words in the dataset."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}. Run download_dataset() first.")
    
    # Get all subdirectories (each is a word)
    words = [d.name for d in DATASET_DIR.iterdir() if d.is_dir() and not d.name.startswith('_')]
    words.sort()
    return words


def load_audio_file(filepath: str) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        filepath: Path to .wav file
    
    Returns:
        Audio signal as numpy array (1 second @ 16kHz = 16000 samples)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")
    
    # Load audio
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=AUDIO_LENGTH)
    
    # Ensure exactly 1 second (pad or trim)
    target_length = int(SAMPLE_RATE * AUDIO_LENGTH)
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        # Trim
        audio = audio[:target_length]
    
    return audio


def generate_silence_from_background(num_samples: int = 100, seed: int = 42) -> np.ndarray:
    """
    Generate silence samples by extracting 1-second clips from background noise.
    
    Args:
        num_samples: Number of silence samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        Array of silence/background noise samples (num_samples, seq_length, num_mfcc)
    """
    np.random.seed(seed)
    
    background_dir = DATASET_DIR / "_background_noise_"
    if not background_dir.exists():
        raise ValueError("Background noise directory not found")
    
    # Get all background noise files
    noise_files = list(background_dir.glob("*.wav"))
    if len(noise_files) == 0:
        raise ValueError("No background noise files found")
    
    print(f"Generating {num_samples} silence samples from {len(noise_files)} background noise files...")
    
    silence_samples = []
    target_length = int(SAMPLE_RATE * AUDIO_LENGTH)
    
    for i in range(num_samples):
        # Pick a random noise file
        noise_file = noise_files[i % len(noise_files)]
        
        # Load the full noise file
        audio, sr = librosa.load(str(noise_file), sr=SAMPLE_RATE)
        
        # Extract a random 1-second segment
        if len(audio) > target_length:
            start_idx = np.random.randint(0, len(audio) - target_length)
            audio_segment = audio[start_idx:start_idx + target_length]
        else:
            # If file is shorter than 1 second, pad it
            audio_segment = np.pad(audio, (0, max(0, target_length - len(audio))))[:target_length]
        
        # Extract MFCC
        mfcc = extract_mfcc(audio_segment)
        silence_samples.append(mfcc)
    
    return np.stack(silence_samples, axis=0)


def load_background_noise_clips(
    num_samples: int = 4000,
    seq_length: int = 1024,
    seed: int = 42
) -> np.ndarray:
    """
    Pre-generate a pool of background noise MFCC clips of the given sequence length.

    The background noise files are long recordings (several minutes).
    Each call crops a random segment of the required duration, extracts MFCC features,
    and trims/pads to exactly seq_length frames.

    Args:
        num_samples: Number of noise clips to generate
        seq_length:  Number of MFCC frames per clip (default 1024 ≈ 10.2 s)
        seed:        Random seed for reproducibility

    Returns:
        Array of noise MFCC clips (num_samples, seq_length, 13)
    """
    np.random.seed(seed)
    background_dir = DATASET_DIR / "_background_noise_"
    if not background_dir.exists():
        raise ValueError(f"Background noise directory not found at {background_dir}")
    noise_files = list(background_dir.glob("*.wav"))
    if len(noise_files) == 0:
        raise ValueError("No background noise files found")

    print(f"Generating {num_samples} noise clips (seq_length={seq_length}) "
          f"from {len(noise_files)} background files...")

    # Audio samples needed: (seq_length - 1) * hop so that MFCC gives >= seq_length frames
    audio_samples = seq_length * HOP_LENGTH
    clips = []
    for i in range(num_samples):
        noise_file = noise_files[np.random.randint(len(noise_files))]
        audio, _ = librosa.load(str(noise_file), sr=SAMPLE_RATE)
        max_start = max(1, len(audio) - audio_samples)
        start = np.random.randint(0, max_start)
        segment = audio[start:start + audio_samples]
        if len(segment) < audio_samples:
            segment = np.pad(segment, (0, audio_samples - len(segment)))
        mfcc = extract_mfcc(segment)          # (T, 13), T ≈ seq_length
        # Trim or pad to exactly seq_length frames
        if mfcc.shape[0] >= seq_length:
            mfcc = mfcc[:seq_length]
        else:
            mfcc = np.pad(mfcc, ((0, seq_length - mfcc.shape[0]), (0, 0)))
        clips.append(mfcc)

    return np.stack(clips)  # (num_samples, seq_length, 13)


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio: Audio signal (16000 samples)
    
    Returns:
        MFCC features (num_frames, num_mfcc)
        Typically (101, 13) for 1 second @ ~100fps
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=NUM_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # Transpose to (time, features)
    mfcc = mfcc.T
    
    # Normalize per coefficient (mean=0, std=1)
    mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)
    
    return mfcc


def load_word_examples(
    word: str, 
    max_examples: int = 100, 
    split: str = "train",
    exact_count: int = None,
    return_paths: bool = False
) -> np.ndarray:
    """
    Load audio examples for a specific word and extract MFCCs.
    
    Args:
        word: Word to load (e.g., 'yes', 'no', 'zero')
        max_examples: Maximum number of examples to load
        split: Which split to load - "train", "validation", or "test"
        exact_count: If specified, generate exactly this many silence samples (only for '_silence_')
        return_paths: If True, also return file paths (for export)
    
    Returns:
        features: MFCC features (num_examples, num_frames, num_mfcc)
        paths (optional): List of file paths if return_paths=True
    """
    # Special handling for silence
    if word == "_silence_":
        num_samples = exact_count if exact_count is not None else max_examples
        features = generate_silence_from_background(num_samples=num_samples)
        if return_paths:
            # For silence, return placeholder paths (background noise)
            paths = [f"_background_noise_/generated_{i}" for i in range(num_samples)]
            return features, paths
        return features
    
    word_dir = DATASET_DIR / word
    if not word_dir.exists():
        raise ValueError(f"Word '{word}' not found in dataset")
    
    # Load split files
    validation_files, test_files = load_split_files()
    
    # Get all .wav files for this word
    all_audio_files = list(word_dir.glob("*.wav"))
    
    # Filter based on split
    audio_files = []
    for audio_file in all_audio_files:
        relative_path = f"{word}/{audio_file.name}"
        
        if split == "validation":
            if relative_path in validation_files:
                audio_files.append(audio_file)
        elif split == "test":
            if relative_path in test_files:
                audio_files.append(audio_file)
        else:  # train
            if relative_path not in validation_files and relative_path not in test_files:
                audio_files.append(audio_file)
    
    # Limit to max_examples
    audio_files = audio_files[:max_examples]
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found for word '{word}' in split '{split}'")
    
    print(f"Loading {len(audio_files)} examples of '{word}' from {split} set...")
    
    # Load and process each file
    features_list = []
    paths_list = []
    for audio_file in audio_files:
        audio = load_audio_file(str(audio_file))
        mfcc = extract_mfcc(audio)
        features_list.append(mfcc)
        paths_list.append(str(audio_file))
    
    # Stack into array
    features = np.stack(features_list, axis=0)
    
    if return_paths:
        return features, paths_list
    return features


def create_binary_dataset(
    positive_word: str = "yes",
    max_positive: int = 500,
    max_negative: int = 500,
    negative_words: List[str] = None,
    split: str = "train",
    return_paths: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary classification dataset: target word vs others.
    
    Args:
        positive_word: Word to detect (e.g., 'yes')
        max_positive: Max examples of positive word
        max_negative: Max examples of negative words (silence/other)
        negative_words: List of words to use as negative. If None, use silence + random words
        split: Which split to load - "train", "validation", or "test"
        return_paths: If True, also return file paths (for export)
    
    Returns:
        features: (num_examples, num_frames, num_mfcc)
        labels: (num_examples,) - 0=negative, 1=positive
        paths (optional): List of file paths if return_paths=True
    """
    print(f"\nCreating binary dataset: '{positive_word}' vs others ({split} set)")
    
    # Load positive examples
    if return_paths:
        positive_features, positive_paths = load_word_examples(positive_word, max_positive, split=split, return_paths=True)
    else:
        positive_features = load_word_examples(positive_word, max_positive, split=split)
        positive_paths = []
    positive_labels = np.ones(len(positive_features), dtype=np.int32)
    
    print(f"Loaded {len(positive_features)} positive examples")
    
    # Load negative examples
    if negative_words is None:
        # Use silence + a few random other words
        negative_words = ['_silence_', 'no', 'up', 'down', 'left', 'right']
    
    # BALANCE: Load same total number of negatives as positives
    num_positive = len(positive_features)
    samples_per_neg_word = num_positive // len(negative_words)
    
    negative_features_list = []
    negative_paths_list = []
    for neg_word in negative_words:
        try:
            if return_paths:
                neg_features, neg_paths = load_word_examples(neg_word, samples_per_neg_word, split=split, return_paths=True)
                negative_paths_list.extend(neg_paths)
            else:
                neg_features = load_word_examples(neg_word, samples_per_neg_word, split=split)
            negative_features_list.append(neg_features)
            print(f"Loaded {len(neg_features)} examples of '{neg_word}'")
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load '{neg_word}', skipping")
            continue
    
    if len(negative_features_list) == 0:
        raise ValueError("No negative examples could be loaded")
    
    negative_features = np.concatenate(negative_features_list, axis=0)
    negative_labels = np.zeros(len(negative_features), dtype=np.int32)
    
    # Combine and shuffle
    features = np.concatenate([positive_features, negative_features], axis=0)
    labels = np.concatenate([positive_labels, negative_labels], axis=0)
    
    if return_paths:
        all_paths = list(positive_paths) + negative_paths_list
    
    # Shuffle with deterministic seed based on split (ensures reproducibility)
    # This ensures paths and features have the same order across calls
    shuffle_seed = hash(split) % (2**32)
    rng = np.random.default_rng(seed=shuffle_seed)
    indices = rng.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    if return_paths:
        all_paths = [all_paths[i] for i in indices]
    
    print(f"Dataset created: {len(features)} examples (BALANCED)")
    print(f"  Positive ('{positive_word}'): {np.sum(labels == 1)}")
    print(f"  Negative: {np.sum(labels == 0)}")
    print(f"  Feature shape: {features.shape}")
    
    if return_paths:
        return features, labels, all_paths
    return features, labels


def create_multiclass_dataset(
    words: List[str],
    max_per_word: int = 500,
    split: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create multi-class classification dataset with BALANCED classes.
    NEW: Balances to MAXIMUM real audio count to preserve all data.
    
    Args:
        words: List of words to include (e.g., ['yes', 'no', 'up', 'down'])
        max_per_word: Max examples per word
        split: Which split to load - "train", "validation", or "test"
    
    Returns:
        features: (num_examples, num_frames, num_mfcc)
        labels: (num_examples,) - class index for each word
    """
    print(f"\nCreating multi-class dataset: {words} ({split} set)")
    
    # STEP 1: Load all REAL audio words (not silence) to find max count
    real_features_list = []
    real_counts = []
    real_words = []
    has_silence = '_silence_' in words
    silence_position = words.index('_silence_') if has_silence else None
    
    for word in words:
        if word == '_silence_':
            continue  # Skip silence in first pass
        
        try:
            word_features = load_word_examples(word, max_per_word, split=split)
            real_features_list.append(word_features)
            real_counts.append(len(word_features))
            real_words.append(word)
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load '{word}', skipping")
            continue
    
    if len(real_features_list) == 0:
        raise ValueError("No examples could be loaded")
    
    # Find max count from real audio
    target_count = max(real_counts)
    print(f"\nBalancing classes to {target_count} examples each (max real audio count)...")
    
    # STEP 2: Generate silence with exact target count
    silence_features = None
    if has_silence:
        silence_features = load_word_examples('_silence_', max_per_word, split=split, 
                                              exact_count=target_count)
        print(f"Loaded {len(silence_features)} examples of '_silence_'")
    
    # STEP 3: Build final feature list in correct order
    all_features = []
    all_words_ordered = []
    
    for idx, original_word in enumerate(words):
        if original_word == '_silence_':
            all_features.append(silence_features)
            all_words_ordered.append('_silence_')
        else:
            # Find this word in real_words
            real_idx = real_words.index(original_word)
            all_features.append(real_features_list[real_idx])
            all_words_ordered.append(original_word)
    
    # STEP 4: Balance all classes to target_count
    balanced_features = []
    balanced_labels = []
    
    for class_idx, (word_features, word) in enumerate(zip(all_features, all_words_ordered)):
        if len(word_features) > target_count:
            # Randomly sample target_count examples
            indices = np.random.choice(len(word_features), target_count, replace=False)
            word_features = word_features[indices]
        # No warning needed - we keep all available data (up to target_count)
        
        word_labels = np.full(len(word_features), class_idx, dtype=np.int32)
        
        balanced_features.append(word_features)
        balanced_labels.append(word_labels)
        print(f"  Class {class_idx} ('{word}'): {len(word_features)} examples")
    
    # STEP 5: Combine and shuffle
    features = np.concatenate(balanced_features, axis=0)
    labels = np.concatenate(balanced_labels, axis=0)
    
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]
    
    print(f"Dataset created: {len(features)} examples across {len(balanced_features)} classes (BALANCED)")
    print(f"  Feature shape: {features.shape}")

    # DEBUG CHECKS:
    assert not np.any(np.isnan(features)), "NaN in features!"
    assert not np.any(np.isinf(features)), "Inf in features!"
    assert features.dtype == np.float32, f"Wrong dtype: {features.dtype}"
    assert features.shape[1] == 101, f"Wrong seq_len: {features.shape[1]}"
    assert features.shape[2] == 13, f"Wrong num_mfcc: {features.shape[2]}"
    
    print(f"Audio data stats: min={np.min(features):.3f}, max={np.max(features):.3f}, mean={np.mean(features):.3f}", flush=True)
        
    return features, labels


# ==================== EXPORT UTILITIES ====================

def get_raw_audio_from_path(filepath: str) -> np.ndarray:
    """
    Get raw audio waveform from a file path.
    
    Args:
        filepath: Path to the audio file (or placeholder for silence)
    
    Returns:
        Raw audio waveform as numpy array (16000 samples for 1 second)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")
    
    # Handle generated silence paths
    if filepath.startswith("_background_noise_/generated_"):
        # Return random background noise
        background_dir = DATASET_DIR / "_background_noise_"
        noise_files = list(background_dir.glob("*.wav"))
        if noise_files:
            # Pick first noise file and extract a segment
            audio, sr = librosa.load(str(noise_files[0]), sr=SAMPLE_RATE)
            target_length = int(SAMPLE_RATE * AUDIO_LENGTH)
            if len(audio) > target_length:
                start_idx = np.random.randint(0, len(audio) - target_length)
                return audio[start_idx:start_idx + target_length]
            else:
                return np.pad(audio, (0, max(0, target_length - len(audio))))[:target_length]
        else:
            # Return zeros if no background files
            return np.zeros(int(SAMPLE_RATE * AUDIO_LENGTH))
    
    # Load actual audio file
    return load_audio_file(filepath)


def save_audio_as_mp4(audio: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Save audio waveform as an MP4 file (audio only).
    
    Args:
        audio: Raw audio waveform
        output_path: Path to save the MP4 file
        sample_rate: Sample rate of the audio
    
    Returns:
        Path to the saved MP4 file
    """
    import subprocess
    import tempfile
    
    # First save as WAV
    wav_path = output_path.replace('.mp4', '.wav')
    
    if SOUNDFILE_AVAILABLE:
        import soundfile as sf
        sf.write(wav_path, audio, sample_rate)
    else:
        # Fallback: use scipy if available
        try:
            from scipy.io import wavfile
            # Normalize to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(wav_path, sample_rate, audio_int16)
        except ImportError:
            raise ImportError("Either soundfile or scipy is required to save audio")
    
    # Convert to MP4 using ffmpeg (if available)
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path, 
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ], capture_output=True, check=True)
        print(f"✓ Saved MP4: {output_path}")
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, PermissionError):
        # ffmpeg not available or permission issue, just keep the WAV file
        print(f"Note: ffmpeg not available, saved as WAV instead: {wav_path}")
        return wav_path


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=" * 80)
    print("Google Speech Commands Dataset Loader")
    print("=" * 80)
    
    # Check if libraries are available
    if not LIBROSA_AVAILABLE:
        print("\n❌ ERROR: Required libraries not installed!")
        print("Install with:")
        print("  pip install librosa soundfile")
        exit(1)
    
    print("\n1. Download dataset (if needed)")
    print("-" * 80)
    try:
        download_dataset()
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nManual download instructions:")
        print(f"1. Download: {DATASET_URL}")
        print(f"2. Extract to: {DATASET_DIR}")
        exit(1)
    
    print("\n2. Available words")
    print("-" * 80)
    words = get_word_list()
    print(f"Found {len(words)} words:")
    print(", ".join(words[:20]) + "...")
    
    print("\n3. Check split files")
    print("-" * 80)
    validation_files, test_files = load_split_files()
    print(f"Validation files: {len(validation_files)}")
    print(f"Test files: {len(test_files)}")
    
    print("\n4. Test new balancing strategy")
    print("-" * 80)
    print("Testing multiclass dataset with balance-to-max strategy:")
    features, labels = create_multiclass_dataset(
        words=["_silence_", "zero", "one", "two", "three"],
        max_per_word=100,
        split="validation"
    )
    
    print("\n" + "=" * 80)
    print("READY FOR TRAINING!")
    print("=" * 80)