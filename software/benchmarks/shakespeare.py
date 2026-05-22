"""Tiny Shakespeare character-level language modeling benchmark.

Task: next-character prediction on the Tiny Shakespeare dataset.
  - ~1.1M characters, 65 unique characters.
  - Split: 90% train / 5% val / 5% test (contiguous, no shuffle).
  - Input: one-hot encoded characters (vocab_size=65), shape (B, T, 65).
  - Target: one-hot encoded next characters, shape (B, T, 65).
  - Loss: cross-entropy averaged over ALL timesteps (seq-to-seq, not last-only).
  - Metric: Bits Per Character (BPC) = cross_entropy_loss / ln(2).

Length generalisation tasks (test-only):
  tinyshakespeare_512, tinyshakespeare_1024, tinyshakespeare_2048 evaluate a
  model trained on T=256 on longer sequences by feeding longer contiguous chunks
  from the test split. No positional encodings are used, so the recurrence
  extrapolates naturally.
"""

from typing import Tuple, Optional
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Global cache
# ---------------------------------------------------------------------------
_SHAKESPEARE_CACHE: Optional[dict] = None

VOCAB_SIZE = 65
_DATA_DIR = Path(__file__).parent.parent / "projects" / "anonymized_neurips2026" / "data" / "tinyshakespeare"
_DATA_PATH = _DATA_DIR / "input.txt"
_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _download_shakespeare() -> str:
    """Download Tiny Shakespeare text and cache it locally. Returns raw text."""
    import urllib.request

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not _DATA_PATH.exists():
        print(f"Downloading Tiny Shakespeare from {_DATA_URL} ...")
        try:
            req = urllib.request.Request(
                _DATA_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            _DATA_PATH.write_text(text, encoding="utf-8")
            print(f"  Saved to {_DATA_PATH}  ({len(text):,} chars)")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download Tiny Shakespeare from {_DATA_URL}. "
                f"Error: {exc}. "
                f"Please download manually and save to {_DATA_PATH}."
            ) from exc

    return _DATA_PATH.read_text(encoding="utf-8")


def get_shakespeare_data() -> dict:
    """
    Load Tiny Shakespeare and return train/val/test splits as JAX arrays.

    Returns dict with keys:
        'train', 'val', 'test'  -> jax.Array of int32 token indices, shape (N,)
        'char_to_int'           -> dict[str, int]
        'int_to_char'           -> dict[int, str]
        'vocab_size'            -> int (65)
    """
    global _SHAKESPEARE_CACHE
    if _SHAKESPEARE_CACHE is not None:
        return _SHAKESPEARE_CACHE

    text = _download_shakespeare()

    # Build vocabulary from the full text (sorted for determinism)
    chars = sorted(set(text))
    assert len(chars) == VOCAB_SIZE, (
        f"Expected {VOCAB_SIZE} unique chars, found {len(chars)}: {chars}"
    )
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    # Encode full text
    data = np.array([char_to_int[c] for c in text], dtype=np.int32)

    # Split: 90% train, 5% val, 5% test
    n = len(data)
    n_train = int(0.90 * n)
    n_val   = int(0.05 * n)
    # test = remainder (≈ 5%)

    train_data = jax.device_put(jnp.array(data[:n_train]))
    val_data   = jax.device_put(jnp.array(data[n_train : n_train + n_val]))
    test_data  = jax.device_put(jnp.array(data[n_train + n_val :]))

    print(
        f"Tiny Shakespeare loaded: "
        f"train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,} chars"
    )

    _SHAKESPEARE_CACHE = {
        "train": train_data,
        "val":   val_data,
        "test":  test_data,
        "char_to_int": char_to_int,
        "int_to_char": int_to_char,
        "vocab_size":  VOCAB_SIZE,
    }
    return _SHAKESPEARE_CACHE


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------

def create_shakespeare_batch(
    batch_size: int,
    seq_length: int,
    input_size: int,  # unused; kept for uniform data_fn signature
    key: jax.Array,
    split: str = "train",
) -> Tuple[jax.Array, jax.Array]:
    """
    Sample a batch of (input, target) pairs from the Shakespeare dataset.

    Each example is a non-overlapping chunk of length seq_length.
    The target is the same chunk shifted right by one character.

    For the train split a random offset in [0, seq_length) is applied at each
    call (using the provided JAX key) to vary chunk boundaries across epochs.
    Val and test splits use offset=0 to give reproducible results.

    Args:
        batch_size:  Number of sequences in the batch.
        seq_length:  Context window T (chunk length).
        input_size:  Ignored (kept for API compatibility).
        key:         JAX PRNG key.
        split:       'train', 'val', or 'test'.

    Returns:
        inputs:  (batch_size, seq_length, VOCAB_SIZE) one-hot float32
        targets: (batch_size, seq_length, VOCAB_SIZE) one-hot float32
    """
    data = get_shakespeare_data()[split]

    # Val/test: deterministic full-coverage sweep — return ALL non-overlapping
    # chunks in order, ignoring batch_size.  This removes sampling noise and
    # makes val and test losses directly comparable.
    if split != "train":
        num_chunks = (len(data) - 1) // seq_length
        num_chunks = max(num_chunks, 1)
        all_starts = jnp.arange(num_chunks) * seq_length          # (num_chunks,)
        offsets_2d = jnp.arange(seq_length + 1)[None, :]          # (1, T+1)
        indices = all_starts[:, None] + offsets_2d                 # (num_chunks, T+1)
        indices = jnp.clip(indices, 0, len(data) - 1)
        chunks  = data[indices]                                    # (num_chunks, T+1)
        inputs  = jax.nn.one_hot(chunks[:, :-1], VOCAB_SIZE)      # (num_chunks, T, 65)
        targets = jax.nn.one_hot(chunks[:, 1:],  VOCAB_SIZE)
        return inputs, targets

    # Train: random offset + random chunk sampling (unchanged)
    offset_key, sample_key = jax.random.split(key)
    offset = int(jax.random.randint(offset_key, shape=(), minval=0, maxval=seq_length))

    effective_data = data[offset:]
    num_chunks = (len(effective_data) - 1) // seq_length
    num_chunks = max(num_chunks, 1)

    chunk_indices = jax.random.randint(
        sample_key, shape=(batch_size,), minval=0, maxval=num_chunks
    )
    starts = chunk_indices * seq_length  # (batch_size,)

    # Use numpy for the indexing (pure Python loop avoided via vectorised gather)
    # Build (batch_size, seq_length+1) index matrix
    offsets = jnp.arange(seq_length + 1)[None, :]          # (1, T+1)
    indices = starts[:, None] + offsets                     # (batch_size, T+1)
    # Clamp to valid range
    indices = jnp.clip(indices, 0, len(effective_data) - 1)

    chunks = effective_data[indices]  # (batch_size, T+1)

    input_chars  = chunks[:, :-1]    # (batch_size, T)
    target_chars = chunks[:, 1:]     # (batch_size, T)

    inputs  = jax.nn.one_hot(input_chars,  VOCAB_SIZE)  # (B, T, 65)
    targets = jax.nn.one_hot(target_chars, VOCAB_SIZE)  # (B, T, 65)

    return inputs, targets


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def _make_task(seq_length: int) -> dict:
    """Return a task-config dict for a given sequence length."""
    return {
        "data_fn": lambda b, s, i, k, split="train": create_shakespeare_batch(
            b, s, i, k, split=split
        ),
        "default_params": {
            "seq_length": seq_length,
            "input_size":  VOCAB_SIZE,
            "output_size": VOCAB_SIZE,
        },
        "task_type": "classification",
        "should_split": True,
        # seq-to-seq: loss at every timestep, no pooling
        "compute_loss_only_last_timestep":   False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss":    False,
        "use_mean_pooling_accuracy": False,
        # flag consumed by main.py to compute BPC = loss / ln(2)
        "is_language_model": True,
        # val/test returns all chunks in one deterministic call — no loop needed
        "eval_single_pass": True,
        # callable returning the vocab dict (used by generate mode)
        "get_vocab": get_shakespeare_data,
    }


SHAKESPEARE_TASKS = {
    # Primary training & evaluation task (T = 256)
    "tinyshakespeare":      _make_task(256),
    # Length generalisation: same model evaluated on longer sequences (test-only)
    "tinyshakespeare_512":  _make_task(512),
    "tinyshakespeare_1024": _make_task(1024),
    "tinyshakespeare_2048": _make_task(2048),
}


# ===========================================================================
# Full Shakespeare — Complete Works (Project Gutenberg #100, ~5.5 M chars)
# ===========================================================================

_FULL_SHAKESPEARE_CACHE: Optional[dict] = None
_FULL_DATA_DIR  = Path(__file__).parent.parent / "projects" / "anonymized_neurips2026" / "data" / "fullshakespeare"
_FULL_DATA_PATH = _FULL_DATA_DIR / "input.txt"
_FULL_DATA_URL  = "https://www.gutenberg.org/files/100/100-0.txt"


def get_full_shakespeare_data() -> dict:
    """
    Download (once) and return the Complete Works of William Shakespeare.

    The raw Project Gutenberg file is stripped of its legal header / footer,
    filtered to printable ASCII, then split 90 / 5 / 5.

    Returns the same dict shape as get_shakespeare_data():
        'train', 'val', 'test'  -> jax.Array int32 (N,)
        'char_to_int'           -> dict[str, int]
        'int_to_char'           -> dict[int, str]
        'vocab_size'            -> int
    """
    global _FULL_SHAKESPEARE_CACHE
    if _FULL_SHAKESPEARE_CACHE is not None:
        return _FULL_SHAKESPEARE_CACHE

    _FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not _FULL_DATA_PATH.exists():
        import urllib.request
        print(f"Downloading Complete Works of Shakespeare from {_FULL_DATA_URL} ...")
        try:
            req = urllib.request.Request(_FULL_DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            _FULL_DATA_PATH.write_text(raw, encoding="utf-8")
            print(f"  Saved to {_FULL_DATA_PATH}  ({len(raw):,} chars)")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download Complete Works from {_FULL_DATA_URL}. "
                f"Error: {exc}.  Please download manually and save to {_FULL_DATA_PATH}."
            ) from exc

    raw = _FULL_DATA_PATH.read_text(encoding="utf-8", errors="replace")

    # Strip Project Gutenberg legal header and footer.
    # The markers look like: *** START OF THE PROJECT GUTENBERG EBOOK ... ***
    start_marker = "*** START OF"
    end_marker   = "*** END OF"
    start_idx = raw.find(start_marker)
    end_idx   = raw.find(end_marker)
    if start_idx != -1:
        # Skip to the end of the start-marker line
        raw = raw[raw.index("\n", start_idx) + 1:]
    if end_idx != -1:
        end_idx = raw.find(end_marker)   # recompute after slice
        raw = raw[:end_idx]

    # Keep only printable ASCII (codes 32–126 + newline/tab) for a clean, fixed vocab.
    printable = set(chr(c) for c in range(32, 127)) | {"\n", "\t"}
    text = "".join(c for c in raw if c in printable)

    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}

    data = np.array([char_to_int[c] for c in text], dtype=np.int32)

    n       = len(data)
    n_train = int(0.90 * n)
    n_val   = int(0.05 * n)

    train_data = jax.device_put(jnp.array(data[:n_train]))
    val_data   = jax.device_put(jnp.array(data[n_train : n_train + n_val]))
    test_data  = jax.device_put(jnp.array(data[n_train + n_val :]))

    print(
        f"Full Shakespeare loaded: {n:,} chars, vocab={vocab_size}  "
        f"train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,}"
    )

    _FULL_SHAKESPEARE_CACHE = {
        "train": train_data, "val": val_data, "test": test_data,
        "char_to_int": char_to_int,
        "int_to_char": int_to_char,
        "vocab_size":  vocab_size,
    }
    return _FULL_SHAKESPEARE_CACHE


def create_full_shakespeare_batch(
    batch_size: int,
    seq_length: int,
    input_size: int,   # unused; kept for API compatibility
    key: jax.Array,
    split: str = "train",
) -> Tuple[jax.Array, jax.Array]:
    """Identical sampling logic to create_shakespeare_batch but for the full corpus."""
    d = get_full_shakespeare_data()
    data      = d[split]
    vocab_size = d["vocab_size"]

    if split != "train":
        # Deterministic full-coverage sweep for val / test
        num_chunks = max((len(data) - 1) // seq_length, 1)
        all_starts = jnp.arange(num_chunks) * seq_length
        indices    = jnp.clip(all_starts[:, None] + jnp.arange(seq_length + 1)[None, :],
                              0, len(data) - 1)
        chunks  = data[indices]
        inputs  = jax.nn.one_hot(chunks[:, :-1], vocab_size)
        targets = jax.nn.one_hot(chunks[:, 1:],  vocab_size)
        return inputs, targets

    # Train: random offset + random chunk sampling
    offset_key, sample_key = jax.random.split(key)
    offset         = int(jax.random.randint(offset_key, shape=(), minval=0, maxval=seq_length))
    effective_data = data[offset:]
    num_chunks     = max((len(effective_data) - 1) // seq_length, 1)
    chunk_indices  = jax.random.randint(sample_key, shape=(batch_size,), minval=0, maxval=num_chunks)
    starts         = chunk_indices * seq_length
    indices        = jnp.clip(starts[:, None] + jnp.arange(seq_length + 1)[None, :],
                              0, len(effective_data) - 1)
    chunks  = effective_data[indices]
    inputs  = jax.nn.one_hot(chunks[:, :-1], vocab_size)
    targets = jax.nn.one_hot(chunks[:, 1:],  vocab_size)
    return inputs, targets


def _make_full_task(seq_length: int) -> dict:
    # Vocab size is determined lazily on first call to the data function;
    # we read it here only if the data is already cached (avoids import-time download).
    if _FULL_SHAKESPEARE_CACHE is not None:
        _vs = _FULL_SHAKESPEARE_CACHE["vocab_size"]
    else:
        # Sentinel: trainer reads input_size from this dict.  We force a load on
        # first access by making default_params a _LazyParams instance.
        _vs = None

    class _LazyParams(dict):
        """Resolves vocab_size on first access of 'input_size' / 'output_size'."""
        def __getitem__(self, key):
            if key in ("input_size", "output_size") and self.get("_resolved") is None:
                vs = get_full_shakespeare_data()["vocab_size"]
                super().__setitem__("input_size",  vs)
                super().__setitem__("output_size", vs)
                super().__setitem__("_resolved",   True)
            return super().__getitem__(key)

    params = _LazyParams(seq_length=seq_length)
    if _vs is not None:
        params["input_size"]  = _vs
        params["output_size"] = _vs
        params["_resolved"]   = True

    return {
        "data_fn": lambda b, s, i, k, split="train": create_full_shakespeare_batch(
            b, s, i, k, split=split
        ),
        "default_params": params,
        "task_type": "classification",
        "should_split": True,
        "compute_loss_only_last_timestep":    False,
        "compute_accuracy_only_last_timestep": False,
        "use_mean_pooling_loss":    False,
        "use_mean_pooling_accuracy": False,
        "is_language_model": True,
        "eval_single_pass":  True,
        "get_vocab": get_full_shakespeare_data,
    }


FULL_SHAKESPEARE_TASKS = {
    "fullshakespeare":      _make_full_task(512),
    "fullshakespeare_512":  _make_full_task(512),
    "fullshakespeare_1024": _make_full_task(1024),
    "fullshakespeare_2048": _make_full_task(2048),
    "fullshakespeare_4096": _make_full_task(4096),
}
