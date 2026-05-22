"""Optimizer utilities with gradient checkpointing and padding-aware pooling support."""

from typing import Dict, Tuple, Optional
import functools

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from mru.models import RNN


def create_padding_mask(inputs: Array, padding_token_idx: int = 0) -> Array:
    """
    Create padding mask from one-hot encoded inputs.
    
    Args:
        inputs: One-hot encoded inputs (batch_size, seq_length, vocab_size)
        padding_token_idx: Index of padding token in vocabulary (default: 0)
    
    Returns:
        mask: Binary mask (batch_size, seq_length) where 1=valid, 0=padding
    """
    mask = 1.0 - inputs[:, :, padding_token_idx]
    return mask


def masked_mean(values: Array, mask: Array, axis: int = 1, keepdims: bool = True) -> Array:
    """
    Compute mean over non-padded positions (optimized version).
    
    Args:
        values: Values to average (batch_size, seq_length, ...)
        mask: Binary mask (batch_size, seq_length) where 1=valid, 0=padding
        axis: Axis to average over (default: 1 for sequence dimension)
        keepdims: Whether to keep the reduced dimension
    
    Returns:
        Masked mean with same shape as values except reduced axis
    """
    # Expand mask dimensions to match values using broadcasting
    # More efficient than while loop
    for _ in range(values.ndim - mask.ndim):
        mask = mask[..., None]
    
    # Compute masked sum and count
    masked_values = values * mask
    sum_values = jnp.sum(masked_values, axis=axis, keepdims=keepdims)
    count = jnp.sum(mask, axis=axis, keepdims=keepdims)
    
    # Avoid division by zero
    count = jnp.maximum(count, 1.0)
    
    return sum_values / count


def get_last_valid_timestep(values: Array, mask: Array) -> Array:
    """
    Extract values at the last non-padded timestep (optimized version).
    
    Args:
        values: (batch_size, seq_length, ...) - sequence values
        mask: (batch_size, seq_length) - binary mask where 1=valid, 0=padding
    
    Returns:
        last_values: (batch_size, 1, ...) - values at last valid timestep
    """
    # Create position indices
    positions = jnp.arange(mask.shape[1])  # (seq_length,)
    
    # Mask positions: valid positions keep their index, padding gets -1
    # This is faster than sum for finding last valid index
    masked_positions = jnp.where(mask > 0, positions, -1)  # (batch_size, seq_length)
    
    # Find maximum position (= last valid index)
    last_indices = jnp.argmax(masked_positions, axis=1)  # (batch_size,)
    
    # Use take_along_axis for efficient gathering (faster than advanced indexing)
    last_indices_for_gather = last_indices[:, None]  # (batch_size, 1)
    
    # Expand to match all dimensions
    if values.ndim == 3:
        last_indices_for_gather = jnp.broadcast_to(
            last_indices_for_gather[..., None],
            (values.shape[0], 1, values.shape[2])
        )
    
    last_values = jnp.take_along_axis(values, last_indices_for_gather, axis=1)
    
    return last_values


def _loss_from_outputs(
    outputs: Array,
    targets: Array,
    task_type: str = "regression",
    compute_only_last_timestep: bool = False,
    use_mean_pooling: bool = False,
    mask: Optional[Array] = None,
) -> float:
    """Compute loss from already-computed model outputs (no model call).

    If targets.shape[1] < outputs.shape[1] and neither compute_only_last_timestep
    nor use_mean_pooling is set, loss is computed on the last targets.shape[1]
    timesteps of outputs. This supports selective-output tasks (e.g. selective
    copying) where targets have shape (B, K, C) with K < L.
    """
    targets_timesteps = targets.shape[1]
    targets_are_last_only = targets_timesteps == 1

    if compute_only_last_timestep:
        if mask is not None:
            outputs = get_last_valid_timestep(outputs, mask)
            if not targets_are_last_only:
                targets = get_last_valid_timestep(targets, mask)
        else:
            outputs = outputs[:, -1:]
            targets = targets if targets_are_last_only else targets[:, -1:]
    elif use_mean_pooling:
        if mask is not None:
            outputs = masked_mean(outputs, mask, axis=1, keepdims=True)
            if not targets_are_last_only:
                targets = masked_mean(targets, mask, axis=1, keepdims=True)
        else:
            outputs = jnp.mean(outputs, axis=1, keepdims=True)
            if not targets_are_last_only:
                targets = jnp.mean(targets, axis=1, keepdims=True)
    elif not targets_are_last_only and targets_timesteps < outputs.shape[1]:
        # Variable-length targets: compute loss on the last targets_timesteps outputs.
        outputs = outputs[:, -targets_timesteps:]

    if task_type == "classification":
        log_probs = jax.nn.log_softmax(outputs, axis=-1)
        return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))
    else:
        return jnp.mean(jnp.square(outputs - targets))


def compute_loss(
    model: RNN,
    inputs: Array,
    targets: Array,
    initial_state: list,
    task_type: str = "regression",
    training: bool = False,
    compute_only_last_timestep: bool = False,
    use_mean_pooling: bool = False,
    mask: Optional[Array] = None,
) -> float:
    """
    Compute loss for given inputs and targets.

    Args:
        mask: Optional padding mask (batch_size, seq_length). If provided:
              - with compute_only_last_timestep: extracts last non-padded timestep
              - with use_mean_pooling: computes mean only over non-padded positions
    """
    outputs = model(inputs, initial_state, training=training)["outputs"]
    return _loss_from_outputs(outputs, targets, task_type,
                               compute_only_last_timestep, use_mean_pooling, mask)


def compute_accuracy(
    predictions: Array,
    targets: Array,
    task_type: str = "regression",
    compute_only_last_timestep: bool = False,
    use_mean_pooling: bool = False,
    mask: Optional[Array] = None,
    compute_sequence_accuracy: bool = False,
) -> Array:
    """
    Compute accuracy metric.

    Args:
        mask: Optional padding mask (batch_size, seq_length). If provided:
              - with compute_only_last_timestep: extracts last non-padded timestep
              - with use_mean_pooling: computes mean only over non-padded positions

    If targets.shape[1] < predictions.shape[1] and neither flag is set, accuracy
    is computed on the last targets.shape[1] timesteps (variable-length targets).
    """
    targets_timesteps = targets.shape[1]
    targets_are_last_only = targets_timesteps == 1

    if compute_only_last_timestep:
        if mask is not None:
            predictions = get_last_valid_timestep(predictions, mask)
            if not targets_are_last_only:
                targets = get_last_valid_timestep(targets, mask)
        else:
            predictions = predictions[:, -1:]
            targets = targets if targets_are_last_only else targets[:, -1:]
    elif use_mean_pooling:
        if mask is not None:
            predictions = masked_mean(predictions, mask, axis=1, keepdims=True)
            if not targets_are_last_only:
                targets = masked_mean(targets, mask, axis=1, keepdims=True)
        else:
            predictions = jnp.mean(predictions, axis=1, keepdims=True)
            if not targets_are_last_only:
                targets = jnp.mean(targets, axis=1, keepdims=True)
    elif not targets_are_last_only and targets_timesteps < predictions.shape[1]:
        # Variable-length targets: evaluate accuracy on last targets_timesteps outputs.
        predictions = predictions[:, -targets_timesteps:]
    
    if task_type == "classification":
        pred_classes   = jnp.argmax(predictions, axis=-1)   # (B, T)
        target_classes = jnp.argmax(targets,     axis=-1)   # (B, T)
        correct = pred_classes == target_classes             # (B, T)
        if compute_sequence_accuracy:
            # A sequence is correct only if ALL (non-padded) timesteps are correct.
            if mask is not None:
                correct = jnp.where(mask, correct, True)    # PAD positions don't fail a seq
            return jnp.mean(jnp.all(correct, axis=-1))      # (B,) -> scalar
        return jnp.mean(correct)
    else:
        error = jnp.abs(predictions - targets)
        return -jnp.mean(error)

@functools.partial(nnx.jit, static_argnames=('task_type', 'compute_loss_only_last_timestep', 'use_mean_pooling', 'use_remat', 'use_padding_mask', 'z_penalty_coeff', 'z_penalty_type'))
def train_step_jit(
    model: RNN,
    optimizer: nnx.ModelAndOptimizer,
    inputs: Array,
    targets: Array,
    initial_state: list,
    task_type: str = "regression",
    compute_loss_only_last_timestep: bool = False,
    use_mean_pooling: bool = False,
    use_remat: bool = False,
    use_padding_mask: bool = False,
    padding_token_idx: int = 0,
    z_penalty_coeff: float = 0.0,
    z_penalty_type: str = "l1",
) -> Tuple[Array, Array, Array]:
    """JIT-compiled training step with optional gradient checkpointing and padding awareness."""
    # Create padding mask if needed
    mask = None
    if use_padding_mask:
        mask = create_padding_mask(inputs, padding_token_idx)

    def loss_fn(model):
        model_out = model(inputs, initial_state, training=True)
        outputs = model_out["outputs"]
        task_loss = _loss_from_outputs(outputs, targets, task_type,
                                       compute_loss_only_last_timestep, use_mean_pooling, mask)
        z_mean = model_out.get("z_penalty", jnp.zeros(()))
        if z_penalty_type == "l2":
            penalty = z_penalty_coeff * (z_mean ** 2)
        else:  # l1
            penalty = z_penalty_coeff * z_mean
        total_loss = task_loss + penalty
        return total_loss, z_mean

    if use_remat:
        loss_fn = jax.checkpoint(
            loss_fn,
            policy=jax.checkpoint_policies.nothing_saveable()
        )

    (loss, z_mean), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Compute gradient norm
    grad_leaves = jax.tree.leaves(grads)
    grad_norm = jnp.sqrt(
        jnp.mean(
            jnp.concatenate([jnp.ravel(jnp.square(jnp.abs(g))) for g in grad_leaves])
        )
    )

    optimizer.update(grads)

    return loss, grad_norm, z_mean


def train_step(
    model: RNN,
    optimizer: nnx.ModelAndOptimizer,
    inputs: Array,
    targets: Array,
    initial_state: list,
    task_type: str = "regression",
    compute_loss_only_last_timestep: bool = False,
    use_mean_pooling: bool = False,
    use_remat: bool = False,
    use_padding_mask: bool = False,
    padding_token_idx: int = 0,
    z_penalty_coeff: float = 0.0,
    z_penalty_type: str = "l1",
) -> Tuple[float, Dict[str, float]]:
    """Single training step."""

    loss, grad_norm, z_mean = train_step_jit(
        model, optimizer, inputs, targets, initial_state,
        task_type, compute_loss_only_last_timestep, use_mean_pooling, use_remat,
        use_padding_mask, padding_token_idx, z_penalty_coeff, z_penalty_type
    )

    return float(loss), {"grad_norm": float(grad_norm), "z_mean": float(z_mean)}


@functools.partial(nnx.jit, static_argnames=('task_type', 'compute_loss_only_last', 'compute_acc_only_last',
                                              'use_mean_pooling_loss', 'use_mean_pooling_acc',
                                              'return_infos', 'use_padding_mask', 'compute_sequence_accuracy'))
def eval_step_jit(
    model: RNN,
    inputs: Array,
    targets: Array,
    initial_state: list,
    task_type: str = "regression",
    compute_loss_only_last: bool = False,
    compute_acc_only_last: bool = False,
    use_mean_pooling_loss: bool = False,
    use_mean_pooling_acc: bool = False,
    return_infos: bool = False,
    use_padding_mask: bool = False,
    padding_token_idx: int = 0,
    compute_sequence_accuracy: bool = False,
) -> Tuple[Array, Array, dict]:
    """JIT-compiled evaluation step with padding awareness."""
    # Create padding mask if needed
    mask = None
    if use_padding_mask:
        mask = create_padding_mask(inputs, padding_token_idx)
    
    outputs = model(inputs, initial_state, training=False, return_infos=return_infos)
    predictions = outputs["outputs"]

    loss = _loss_from_outputs(
        predictions, targets, task_type,
        compute_only_last_timestep=compute_loss_only_last,
        use_mean_pooling=use_mean_pooling_loss,
        mask=mask,
    )

    accuracy = compute_accuracy(
        predictions, targets, task_type, compute_only_last_timestep=compute_acc_only_last,
        use_mean_pooling=use_mean_pooling_acc, mask=mask,
        compute_sequence_accuracy=compute_sequence_accuracy,
    )

    infos = outputs.get("recs", []) if return_infos else []
    
    return loss, accuracy, infos


@functools.partial(nnx.jit, static_argnames=('task_type', 'compute_acc_only_last', 'use_mean_pooling_acc', 'use_padding_mask', 'compute_sequence_accuracy'))
def compute_accuracy_on_batch(
    model: RNN,
    inputs: Array,
    targets: Array,
    initial_state: list,
    task_type: str = "regression",
    compute_acc_only_last: bool = False,
    use_mean_pooling_acc: bool = False,
    use_padding_mask: bool = False,
    padding_token_idx: int = 0,
    compute_sequence_accuracy: bool = False,
) -> Array:
    """JIT-compiled function to compute accuracy on a single batch."""
    mask = None
    if use_padding_mask:
        mask = create_padding_mask(inputs, padding_token_idx)

    predictions = model(inputs, initial_state, training=False)["outputs"]

    return compute_accuracy(
        predictions, targets, task_type, compute_only_last_timestep=compute_acc_only_last,
        use_mean_pooling=use_mean_pooling_acc, mask=mask,
        compute_sequence_accuracy=compute_sequence_accuracy,
    )


def get_surr_alpha_scale_factor(
    epoch: int,
    anneal_epochs: int,
    final_factor: float = 5.0,
) -> float:
    """Compute surrogate alpha scaling factor for annealing."""
    if epoch == 0 or epoch >= anneal_epochs:
        return 1.0
    return final_factor ** (1.0 / anneal_epochs)

def get_epsilon_decay_value(
    epoch: int,
    total_epochs: int,
    fraction_const: float = 0.05,
    fraction_decay: float = 0.7,
    abs_const_epochs: int = None,
    abs_decay_epochs: int = None,
) -> float:
    """
    Compute epsilon value for a 3-phase schedule:
      - constant 1.0 for const phase
      - linear decay from 1.0 to 0.0 over decay phase
      - constant 0.0 for the remaining epochs

    Phase lengths can be specified either as absolute epoch counts
    (abs_const_epochs / abs_decay_epochs) or as fractions of total_epochs
    (fraction_const / fraction_decay). Absolute counts take priority when set.

    Args:
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of training epochs
        fraction_const: Fraction of epochs to keep epsilon at 1.0 (default 0.05)
        fraction_decay: Fraction of epochs to linearly decay epsilon (default 0.7)
        abs_const_epochs: Absolute number of epochs to keep epsilon at 1.0 (overrides fraction)
        abs_decay_epochs: Absolute number of epochs for linear decay (overrides fraction)

    Returns:
        Epsilon value for the current epoch
    """
    if abs_const_epochs is not None:
        const_epochs = int(abs_const_epochs)
    else:
        if not (0.0 <= fraction_const <= 1.0):
            raise ValueError("fraction_const must be in [0.0, 1.0]")
        const_epochs = int(round(total_epochs * fraction_const))

    if abs_decay_epochs is not None:
        decay_epochs = int(abs_decay_epochs)
    else:
        if not (0.0 <= fraction_decay <= 1.0):
            raise ValueError("fraction_decay must be in [0.0, 1.0]")
        decay_epochs = int(round(total_epochs * fraction_decay))
    # Ensure we cover total_epochs exactly by assigning remainder to final phase
    final_epochs = total_epochs - const_epochs - decay_epochs

    # Phase selection
    if epoch < 0:
        return 1.0
    if epoch < const_epochs:
        return 1.0
    if decay_epochs <= 0:
        # No decay phase -> jump directly to final
        return 0.0 if epoch >= const_epochs else 1.0
    if epoch >= const_epochs + decay_epochs:
        return 0.0

    # Linear decay from 1.0 to 0.0 across decay_epochs steps (inclusive)
    step = epoch - const_epochs
    if decay_epochs == 1:
        return 0.0
    return 1.0 - (step / (decay_epochs - 1))