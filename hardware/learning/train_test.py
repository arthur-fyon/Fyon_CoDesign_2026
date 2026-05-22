#!/usr/bin/env python3
"""
Training and testing functions for CMOS-SBC RNN models.
Includes training, evaluation, visualization, and export utilities.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import json
import pickle

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from models import RNN
from tasks import BENCHMARK_TASKS
from utils import compute_loss, compute_accuracy, save_model, load_model


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_step(
    model: RNN,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    inputs: jax.Array,
    targets: jax.Array,
    initial_state: Any,
    task_type: str = "regression"
) -> Tuple[RNN, Any, float]:
    """Execute a single training step.
    
    Args:
        model: RNN model to train
        optimizer: Optax optimizer
        opt_state: Optimizer state
        inputs: Input sequences (batch_size, seq_length, input_size)
        targets: Target sequences (batch_size, seq_length, output_size)
        initial_state: Initial hidden state
        task_type: 'regression' or 'classification'
    
    Returns:
        Updated model, optimizer state, and loss value
    """
    def loss_fn(model):
        return compute_loss(model, inputs, targets, initial_state, task_type)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    param_grads = nnx.state(grads, nnx.Param)
    params = nnx.state(model, nnx.Param)
    updates, opt_state = optimizer.update(param_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)

    return model, opt_state, loss


def evaluate_model(
    model: RNN,
    task_config: Dict,
    eval_key: jax.Array,
    batch_size: int,
    num_eval_batches: int = 10,
    split: str = "validation"
) -> Dict[str, float]:
    """Evaluate model on a specific task.
    
    Args:
        model: RNN model to evaluate
        task_config: Task configuration dictionary
        eval_key: Random key for evaluation
        batch_size: Batch size for evaluation
        num_eval_batches: Number of batches to evaluate
        split: Dataset split - 'train', 'validation', or 'test'
    
    Returns:
        Dictionary with 'loss' and 'accuracy' metrics
    """
    total_loss = 0.0
    total_accuracy = 0.0

    data_fn = task_config["data_fn"]
    task_params = task_config["default_params"]
    task_type = task_config.get("task_type", "regression")
    task_name = task_config.get("name", "")

    for i in range(num_eval_batches):
        key = jax.random.fold_in(eval_key, i)

        # Generate evaluation data
        if task_type == "classification":
            if "real_audio" in task_name.lower() or "REAL AUDIO" in task_name:
                eval_inputs, eval_targets = data_fn(
                    batch_size, task_params["seq_length"], task_params["input_size"],
                    key, split=split
                )
            else:
                eval_inputs, eval_targets = data_fn(
                    batch_size, task_params["seq_length"], task_params["input_size"], key
                )
        else:
            eval_inputs, eval_targets = data_fn(
                batch_size, task_params["seq_length"], task_params["input_size"], key
            )

        initial_state = model.init_state(batch_size)
        model.eval()
        model_output = model(eval_inputs, initial_state)
        model.train()
        predictions = model_output["outputs"]

        loss = float(compute_loss(model, eval_inputs, eval_targets, initial_state, task_type))
        accuracy = compute_accuracy(predictions, eval_targets, task_type, majority_vote=True)

        total_loss += loss
        total_accuracy += accuracy

    return {
        "loss": total_loss / num_eval_batches,
        "accuracy": total_accuracy / num_eval_batches
    }


def train_model(
    task_name: str,
    config: Dict[str, Any],
    save_path: str = None,
    verbose: bool = True
) -> Tuple[RNN, List[float], List[Tuple[int, Dict[str, float]]]]:
    """Train a model on the specified task.
    
    Args:
        task_name: Name of the task (e.g., 'real_audio_binary')
        config: Training configuration dictionary
        save_path: Path to save model checkpoints
        verbose: Print training progress
    
    Returns:
        Trained model, training losses, and evaluation metrics history
    """
    if task_name not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(BENCHMARK_TASKS.keys())}")

    task_config = BENCHMARK_TASKS[task_name]
    task_params = task_config["default_params"]
    task_type = task_config.get("task_type", "regression")

    # Training hyperparameters
    num_epochs = config.get("num_epochs", 300)
    batch_size = config.get("batch_size", 16)
    learning_rate = config.get("learning_rate", 1e-3)
    eval_every = config.get("eval_every", 25)

    # Task parameters
    seq_length = task_params["seq_length"]
    input_size = task_params["input_size"]
    output_size = task_params.get("output_size", input_size)

    # Model configuration
    model_config = config.get("model", {})
    model_dim = model_config.get("model_dim", 64)
    cells = model_config.get("cells", ["cmos_sbc"])
    num_recs = model_config.get("num_recs", 2)
    positional_encodings_dims = model_config.get("positional_encodings_dims", 0)
    skip = model_config.get("skip", True)
    num_fcs = model_config.get("num_fcs", 1)

    # Cell configuration
    cell_configs = config.get("cell_configs", {
        "cmos_sbc": {
            "state_dim": 64,
            "parallel": True,
            "norm": None,
            "dropout": 0.2,
            "post_activation": "relu",
            "kernel_scale": 1.0,
            "surr_alpha": 1.0,
        }
    })

    # Initialize model
    main_key = jax.random.PRNGKey(config.get("seed", 42))
    model_key, data_key = jax.random.split(main_key)

    rngs = nnx.Rngs(int(model_key[0]))
    model = RNN(
        input_size=input_size,
        output_size=output_size,
        rngs=rngs,
        model_dim=model_dim,
        cells=cells,
        cell_configs=cell_configs,
        num_recs=num_recs,
        positional_encodings_dims=positional_encodings_dims,
        skip=skip,
        num_fcs=num_fcs,
    )

    # Initialize optimizer with cosine decay
    schedule = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=num_epochs)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )

    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)

    # Training history
    train_losses = []
    eval_metrics = []

    if verbose:
        print(f"\nTraining on: {task_config['name']}")
        print(f"Description: {task_config['description']}")
        print(f"Parameters: seq_length={seq_length}, input_size={input_size}, output_size={output_size}")
        print(f"Model: {len(cells)} cell(s), {num_recs} recurrent layers, {model_dim} hidden dims")
        print("-" * 60)

    start_time = time.time()
    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        # Generate training data
        data_key = jax.random.fold_in(data_key, epoch)

        if task_type == "classification":
            train_inputs, train_targets = task_config["data_fn"](
                batch_size, seq_length, input_size, data_key
            )
        else:
            train_inputs, train_targets = task_config["data_fn"](
                batch_size, seq_length, input_size, data_key
            )

        # Training step
        initial_state = model.init_state(batch_size)
        model, opt_state, train_loss = train_step(
            model, optimizer, opt_state, train_inputs, train_targets,
            initial_state, task_type
        )
        train_losses.append(float(train_loss))

        # Evaluation
        if (epoch + 1) % eval_every == 0:
            eval_key = jax.random.fold_in(data_key, epoch + 1000)
            metrics = evaluate_model(model, task_config, eval_key, batch_size)

            eval_metrics.append((epoch + 1, metrics))

            if metrics['loss'] < best_eval_loss:
                best_eval_loss = metrics['loss']
                if save_path:
                    save_model(model, save_path, epoch + 1, metrics, task_name, config)

            if verbose:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:4d} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Eval Loss: {metrics['loss']:.6f} | "
                      f"Eval Acc: {metrics['accuracy']:.4f} | "
                      f"Time: {elapsed:.1f}s")

    if verbose:
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training completed in {total_time:.1f}s")
        print(f"Best eval loss: {best_eval_loss:.6f}")

    return model, train_losses, eval_metrics


# =============================================================================
# TESTING AND EVALUATION FUNCTIONS
# =============================================================================

def test_model(model: RNN, task_name: str, batch_size: int = 32, num_batches: int = 50):
    """Test model on a specific task using the test split.
    
    Args:
        model: Trained RNN model
        task_name: Name of the task
        batch_size: Batch size for testing
        num_batches: Number of test batches
    
    Returns:
        Dictionary with test metrics
    """
    if task_name not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    task_config = BENCHMARK_TASKS[task_name]
    test_key = jax.random.PRNGKey(999)

    metrics = evaluate_model(model, task_config, test_key, batch_size, num_batches, split="test")

    print(f"\n{'='*60}")
    print(f"Test Results for {task_config['name']}")
    print(f"{'='*60}")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"{'='*60}")

    return metrics


def plot_confusion_matrix(
    y_true, y_pred, num_classes, task_name, model_name, split,
    save_path=None, normalize=False
):
    """Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        task_name: Task name for title
        model_name: Model name for title
        split: Data split name
        save_path: Path to save the figure
        normalize: Whether to normalize the matrix
    
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='.2f' if normalize else 'd',
        cmap='Blues', square=True, cbar=True,
        xticklabels=range(num_classes), yticklabels=range(num_classes)
    )

    title = f'Confusion Matrix - {task_name}\nModel: {model_name} | Split: {split.capitalize()}'
    if normalize:
        title += ' (Normalized)'
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

    plt.close()
    return cm


def export_confusion_matrix(
    model: RNN,
    task_key: str,
    batch_size: int = 32,
    num_batches: int = 50,
    split: str = "test",
    cm_save_dir: str = "confusion_matrices",
    model_name: str = "model"
):
    """Generate and export confusion matrix for a classification task.
    
    Args:
        model: Trained RNN model
        task_key: Task key in BENCHMARK_TASKS
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate
        split: Data split to use
        cm_save_dir: Directory to save confusion matrices
        model_name: Model name for labeling
    
    Returns:
        Dictionary with confusion matrices
    """
    if task_key not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown task: {task_key}")

    task_config = BENCHMARK_TASKS[task_key]
    task_type = task_config.get("task_type", "regression")

    if task_type != "classification":
        print(f"Task {task_key} is not a classification task. Skipping confusion matrix.")
        return None

    print(f"\nGenerating confusion matrix for {split} split...")

    # Collect predictions
    all_predictions = []
    all_targets = []

    data_fn = task_config["data_fn"]
    task_params = task_config["default_params"]
    num_classes = task_params.get("output_size", 2)

    eval_key = jax.random.PRNGKey(999)

    for i in range(num_batches):
        key = jax.random.fold_in(eval_key, i)

        if "real_audio" in task_key:
            inputs, targets = data_fn(
                batch_size, task_params["seq_length"], task_params["input_size"],
                key, split=split
            )
        else:
            inputs, targets = data_fn(
                batch_size, task_params["seq_length"], task_params["input_size"], key
            )

        initial_state = model.init_state(batch_size)
        model.eval()
        model_output = model(inputs, initial_state)
        model.train()
        predictions = model_output["outputs"]

        final_preds = jax.nn.softmax(predictions[:, -1, :], axis=-1)
        pred_classes = jnp.argmax(final_preds, axis=-1)
        true_classes = jnp.argmax(targets[:, -1, :], axis=-1)

        all_predictions.extend(pred_classes.tolist())
        all_targets.extend(true_classes.tolist())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Create save directory
    Path(cm_save_dir).mkdir(parents=True, exist_ok=True)

    # Save confusion matrices
    cm_path = Path(cm_save_dir) / f"{model_name}_{task_key}_{split}_cm.png"
    cm_norm_path = Path(cm_save_dir) / f"{model_name}_{task_key}_{split}_cm_normalized.png"

    cm = plot_confusion_matrix(
        all_targets, all_predictions, num_classes,
        task_config['name'], model_name, split,
        str(cm_path), normalize=False
    )

    cm_norm = plot_confusion_matrix(
        all_targets, all_predictions, num_classes,
        task_config['name'], model_name, split,
        str(cm_norm_path), normalize=True
    )

    # Save raw confusion matrix as CSV
    csv_path = Path(cm_save_dir) / f"{model_name}_{task_key}_{split}_cm.csv"
    np.savetxt(csv_path, cm, delimiter=',', fmt='%d')
    print(f"Raw confusion matrix saved to: {csv_path}")

    return {'confusion_matrix': cm, 'confusion_matrix_normalized': cm_norm}


# =============================================================================
# QUANTIZATION TESTING
# =============================================================================

def quantize_weights(weights: jax.Array, num_bits: int) -> jax.Array:
    """Quantize weights to a specified number of bits.
    
    Args:
        weights: Original weight array
        num_bits: Number of bits for quantization (e.g., 8, 4, 2)
    
    Returns:
        Quantized weights scaled back to original range
    """
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive")

    w_min = jnp.min(weights)
    w_max = jnp.max(weights)
    num_levels = 2 ** num_bits

    # Normalize to [0, 1], quantize, then scale back
    weights_normalized = (weights - w_min) / (w_max - w_min + 1e-8)
    weights_quantized = jnp.round(weights_normalized * (num_levels - 1)) / (num_levels - 1)
    weights_quantized = weights_quantized * (w_max - w_min) + w_min

    return weights_quantized


def test_quantized_model(
    model: RNN,
    task_name: str,
    num_bits: int = 8,
    batch_size: int = 32,
    num_batches: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """Test model with quantized weights.
    
    Args:
        model: Original trained RNN model
        task_name: Name of the task
        num_bits: Number of bits for quantization
        batch_size: Batch size for testing
        num_batches: Number of batches
        verbose: Print detailed results
    
    Returns:
        Dictionary with test metrics and compression statistics
    """
    if task_name not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    task_config = BENCHMARK_TASKS[task_name]
    test_key = jax.random.PRNGKey(999)

    # Test original model
    if verbose:
        print("=" * 60)
        print(f"Testing Model Quantization on: {task_config['name']}")
        print("=" * 60)
        print("\n1. Testing ORIGINAL model...")

    original_metrics = evaluate_model(model, task_config, test_key, batch_size, num_batches)

    if verbose:
        print(f"   Original Loss: {original_metrics['loss']:.6f}")
        print(f"   Original Accuracy: {original_metrics['accuracy']:.4f}")

    # Get original parameters
    original_params = nnx.state(model, nnx.Param)
    flat_params = jax.tree.leaves(original_params)

    # Apply quantization
    total_params = sum(p.size for p in flat_params)
    modified_flat = []

    for param_array in flat_params:
        modified_value = quantize_weights(param_array, num_bits)
        modified_flat.append(modified_value)

    # Reconstruct and update model
    modified_params = jax.tree.unflatten(jax.tree.structure(original_params), modified_flat)
    nnx.update(model, modified_params)

    # Test quantized model
    if verbose:
        print(f"\n2. Testing QUANTIZED model ({num_bits}-bit)...")

    modified_metrics = evaluate_model(model, task_config, test_key, batch_size, num_batches)

    # Restore original parameters
    nnx.update(model, original_params)

    # Calculate degradation
    loss_degradation = ((modified_metrics['loss'] - original_metrics['loss']) /
                       (original_metrics['loss'] + 1e-8) * 100)
    accuracy_degradation = ((original_metrics['accuracy'] - modified_metrics['accuracy']) /
                           (original_metrics['accuracy'] + 1e-8) * 100)

    if verbose:
        print(f"   Modified Loss: {modified_metrics['loss']:.6f}")
        print(f"   Modified Accuracy: {modified_metrics['accuracy']:.4f}")
        print("\n" + "=" * 60)
        print("COMPRESSION STATISTICS:")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Quantization: 32-bit → {num_bits}-bit")
        print(f"Theoretical size reduction: {32/num_bits:.2f}x")
        print("\n" + "=" * 60)
        print("PERFORMANCE DEGRADATION:")
        print("=" * 60)
        print(f"Loss increase: {loss_degradation:+.2f}%")
        print(f"Accuracy decrease: {accuracy_degradation:+.2f}%")
        print("=" * 60)

    return {
        "original_metrics": original_metrics,
        "modified_metrics": modified_metrics,
        "loss_degradation_percent": float(loss_degradation),
        "accuracy_degradation_percent": float(accuracy_degradation),
        "compression_stats": {
            "total_params": int(total_params),
            "num_bits": num_bits,
        }
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_network_states(model: RNN, task_name: str, save_path: str = None):
    """Visualize network states over time for multiple sequences.
    
    Args:
        model: Trained RNN model
        task_name: Task name
        save_path: Base path to save plots
    """
    if task_name not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    task_config = BENCHMARK_TASKS[task_name]
    task_params = task_config["default_params"]
    task_type = task_config.get("task_type", "regression")
    output_size = task_params["output_size"]

    # Generate 5 plots with different seeds
    num_plots = 5
    base_seed = 80

    for plot_idx in range(num_plots):
        test_key = jax.random.PRNGKey(base_seed + plot_idx)

        # Generate test data
        if task_type == "classification":
            if "real_audio" in task_name:
                test_inputs, test_targets = task_config["data_fn"](
                    1, task_params["seq_length"], task_params["input_size"],
                    test_key, split="test"
                )
            else:
                test_inputs, test_targets = task_config["data_fn"](
                    1, task_params["seq_length"], task_params["input_size"], test_key
                )
        else:
            test_inputs, test_targets = task_config["data_fn"](
                1, task_params["seq_length"], task_params["input_size"], test_key
            )

        # Forward pass
        initial_state = model.init_state(1)
        model.eval()
        model_output = model(test_inputs, initial_state)
        model.train()

        # Extract data
        predictions = model_output["outputs"][0]
        inputs = test_inputs[0]
        targets = test_targets[0]
        recs_outputs = model_output["recs"]

        # Create figure
        num_rec_layers = len(recs_outputs)
        fig_height = 6 + 2 * num_rec_layers
        fig, axes = plt.subplots(3 + num_rec_layers, 1, figsize=(12, fig_height))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        actual_seq_length = inputs.shape[0]
        time_steps = range(actual_seq_length)

        # Plot 1: Input features
        if inputs.ndim > 1 and inputs.shape[-1] > 1:
            cmap = plt.cm.get_cmap('tab20')
            for dim in range(inputs.shape[-1]):
                color = cmap(dim % 20)
                axes[0].plot(time_steps, inputs[:, dim], linewidth=1.5, alpha=0.6, color=color)
            axes[0].set_title(f'{task_config["name"]} - Input Features ({inputs.shape[-1]} dims)')
        else:
            axes[0].plot(time_steps, inputs.squeeze(), 'b-', linewidth=2)
            axes[0].set_title(f'{task_config["name"]} - Input Sequence')
        axes[0].set_ylabel('Input Value')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Predictions vs Targets
        if task_type == "classification":
            pred_classes = jnp.argmax(jax.nn.softmax(predictions, axis=-1), axis=-1)
            true_classes = jnp.argmax(targets, axis=-1)
            axes[1].plot(time_steps, true_classes, 'g-', label='True', linewidth=2, marker='o')
            axes[1].plot(time_steps, pred_classes, 'r--', label='Predicted', linewidth=2, marker='s')
            axes[1].set_ylabel('Class')
            axes[1].set_ylim(-0.5, output_size - 0.5)
        else:
            axes[1].plot(time_steps, targets.squeeze(), 'g-', label='Target', linewidth=2)
            axes[1].plot(time_steps, predictions.squeeze(), 'r--', label='Prediction', linewidth=2)
            axes[1].set_ylabel('Value')
        axes[1].set_title('Predictions vs Targets')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3+: Recurrent layer states
        for layer_idx, layer_output in enumerate(recs_outputs):
            ax_idx = 2 + layer_idx
            for cell_name, cell_output in layer_output.items():
                if 'states' in cell_output:
                    states = cell_output['states'][0]
                    for dim in range(min(5, states.shape[-1])):
                        axes[ax_idx].plot(
                            time_steps, states[:, dim],
                            label=f'{cell_name} s{dim}', alpha=0.7
                        )
            axes[ax_idx].set_title(f'Layer {layer_idx} - Hidden States')
            axes[ax_idx].set_ylabel('State')
            axes[ax_idx].legend(fontsize=8, loc='upper right')
            axes[ax_idx].grid(True, alpha=0.3)

        # Final plot: Confidence
        if task_type == "classification":
            confidences = jnp.max(jax.nn.softmax(predictions, axis=-1), axis=-1)
            axes[-1].plot(time_steps, confidences, 'purple', linewidth=2)
            axes[-1].set_title('Prediction Confidence')
            axes[-1].set_ylabel('Max Softmax')
            axes[-1].set_ylim(0, 1)
        else:
            error = jnp.abs(predictions.squeeze() - targets.squeeze())
            axes[-1].plot(time_steps, error, 'orange', linewidth=2)
            axes[-1].set_title('Absolute Error')
            axes[-1].set_ylabel('|Error|')

        axes[-1].set_xlabel('Time Step')
        axes[-1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_path:
            base_path = Path(save_path)
            base_path.parent.mkdir(parents=True, exist_ok=True)
            seed_path = str(base_path).replace('.png', f'_seed{base_seed + plot_idx}.png')
            plt.savefig(seed_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {seed_path}")

        plt.close()

    if save_path:
        print(f"\nAll {num_plots} plots saved successfully!")


# =============================================================================
# EXPORT FUNCTIONS FOR HARDWARE VALIDATION
# =============================================================================

def extract_cell_parameters(cell: Any) -> Dict[str, np.ndarray]:
    """Extract all parameters from a single CMOS-SBC cell.
    
    Returns dict with alphas, beta1, delta, initial_state, and network weights.
    """
    params = {}

    if hasattr(cell, '_alphas'):
        params['alphas'] = np.array(cell._alphas.value)
    if hasattr(cell, '_beta1'):
        params['beta1'] = np.array(cell._beta1.value)
    if hasattr(cell, '_delta'):
        params['delta'] = np.array(cell._delta.value)
    if hasattr(cell, '_initial_state'):
        params['initial_state'] = np.array(cell._initial_state.value)
    if hasattr(cell, '_cand_nn'):
        params['cand_nn_kernel'] = np.array(cell._cand_nn.kernel.value)
        params['cand_nn_bias'] = np.array(cell._cand_nn.bias.value)
    if hasattr(cell, '_post_proj_nn'):
        params['post_proj_nn_kernel'] = np.array(cell._post_proj_nn.kernel.value)
        params['post_proj_nn_bias'] = np.array(cell._post_proj_nn.bias.value)

    return params


def extract_all_model_parameters(model: RNN) -> Dict[str, Any]:
    """Extract all parameters from the entire model."""
    all_params = {}

    # Pre-projection layer
    if hasattr(model, '_pre_proj'):
        all_params['pre_proj'] = {
            'kernel': np.array(model._pre_proj.kernel.value),
            'bias': np.array(model._pre_proj.bias.value),
        }

    # Recurrent layers
    all_params['recurrent_layers'] = []
    if hasattr(model, '_recs'):
        for layer_idx, rec_layer in enumerate(model._recs):
            layer_params = {'cells': {}}
            for cell_name, cell in rec_layer.items():
                layer_params['cells'][cell_name] = extract_cell_parameters(cell)
            all_params['recurrent_layers'].append(layer_params)

    # FC layers
    all_params['fcs'] = []
    if hasattr(model, '_fcs'):
        for fc_layer in model._fcs:
            all_params['fcs'].append({
                'kernel': np.array(fc_layer.kernel.value),
                'bias': np.array(fc_layer.bias.value),
            })

    return all_params


def run_inference_with_tracking(
    model: RNN,
    inputs: jax.Array,
    initial_state: Any,
) -> Dict[str, Any]:
    """Run inference and track all internal signals for hardware validation."""
    model.eval()
    model_output = model(inputs, initial_state)
    model.train()

    predictions = model_output["outputs"]
    recs_outputs = model_output["recs"]

    tracking = {
        'inputs': np.array(inputs[0]),
        'final_outputs': np.array(predictions[0]),
        'recurrent_layers': []
    }

    for layer_idx, layer_output in enumerate(recs_outputs):
        layer_tracking = {'cells': {}}
        for cell_name, cell_output in layer_output.items():
            cell_tracking = {}
            for key in ['states', 'candidates', 'force_low', 'hold', 'force_high', 'beta1', 'beta2', 'delta']:
                if key in cell_output:
                    cell_tracking[key] = np.array(cell_output[key][0])
            layer_tracking['cells'][cell_name] = cell_tracking
        tracking['recurrent_layers'].append(layer_tracking)

    return tracking


def export_inference_data(
    model: RNN,
    task_name: str,
    seed: int = None,
    export_dir: str = "inference_exports",
    batch_size: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """Export inference data for hardware validation.
    
    Saves all model parameters and inference traces for CMOS reproduction.
    
    Args:
        model: Trained RNN model
        task_name: Task name
        seed: Random seed (None = random)
        export_dir: Directory to save exports
        batch_size: Should be 1 for single inference
        verbose: Print progress
    
    Returns:
        Export bundle dictionary
    """
    if task_name not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown task: {task_name}")

    task_config = BENCHMARK_TASKS[task_name]
    task_params = task_config["default_params"]
    task_type = task_config.get("task_type", "regression")

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = int(np.random.randint(0, 2**31))

    if verbose:
        print(f"\n{'='*60}")
        print("EXPORTING INFERENCE DATA FOR HARDWARE VALIDATION")
        print(f"{'='*60}")
        print(f"Task: {task_config['name']}")
        print(f"Seed: {seed}")

    # Generate inference data
    eval_key = jax.random.PRNGKey(seed)
    audio_path = None

    if task_type == "classification" and "real_audio" in task_name:
        result = task_config["data_fn"](
            batch_size, task_params["seq_length"], task_params["input_size"],
            eval_key, split="test", return_paths=True
        )
        inputs, targets, paths = result
        audio_path = paths[0] if paths else None
    else:
        inputs, targets = task_config["data_fn"](
            batch_size, task_params["seq_length"], task_params["input_size"], eval_key
        )

    if verbose:
        print(f"Input shape: {inputs.shape}")
        if audio_path:
            print(f"Audio source: {audio_path}")

    # Run inference
    initial_state = model.init_state(batch_size)
    tracking = run_inference_with_tracking(model, inputs, initial_state)
    all_params = extract_all_model_parameters(model)

    # Create export bundle
    export_bundle = {
        'metadata': {
            'task_name': task_name,
            'task_type': task_type,
            'seed': seed,
            'task_params': task_params,
            'timestamp': str(np.datetime64('now')),
            'audio_source_path': audio_path,
        },
        'model_parameters': all_params,
        'inference_data': {
            'inputs': tracking['inputs'],
            'outputs': tracking['final_outputs'],
            'targets': np.array(targets[0]),
            'recurrent_tracking': tracking['recurrent_layers'],
        }
    }

    # Save as pickle
    pickle_path = export_path / f"inference_{seed}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(export_bundle, f)

    if verbose:
        print(f"✓ Saved: {pickle_path}")

    # Save metadata as JSON
    json_metadata = {
        'metadata': export_bundle['metadata'],
        'model_parameters_shapes': {
            'pre_proj': {k: list(v.shape) for k, v in all_params.get('pre_proj', {}).items()},
            'recurrent_layers': [
                {
                    'cells': {
                        cell_name: {k: list(v.shape) for k, v in cell_params.items()}
                        for cell_name, cell_params in layer['cells'].items()
                    }
                }
                for layer in all_params['recurrent_layers']
            ],
            'fcs': [
                {k: list(v.shape) for k, v in fc.items()}
                for fc in all_params['fcs']
            ]
        },
    }

    json_path = export_path / f"inference_{seed}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(json_metadata, f, indent=2, default=str)

    if verbose:
        print(f"✓ Saved: {json_path}")
        print(f"\n{'='*60}")
        print("EXPORT COMPLETE!")
        print(f"{'='*60}")

    return export_bundle
