"""Model compression utilities: quantization and pruning."""

from typing import Dict, Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from mru.models import RNN
from .trainer import evaluate_model


def quantize_weights(weights: Array, num_bits: int) -> Array:
    """Quantize weights to num_bits precision."""
    if num_bits >= 32:
        return weights
    
    # Get value range
    w_min = jnp.min(weights)
    w_max = jnp.max(weights)
    
    # Quantization levels
    levels = 2 ** num_bits
    scale = (w_max - w_min) / (levels - 1)
    
    # Quantize and dequantize
    quantized = jnp.round((weights - w_min) / scale)
    dequantized = quantized * scale + w_min
    
    return dequantized


def prune_weights(weights: Array, prune_percentage: float, rng_key: jax.Array) -> Array:
    """Prune smallest magnitude weights (or random if magnitudes equal)."""
    if prune_percentage <= 0:
        return weights
    
    # Number of weights to prune
    num_weights = weights.size
    num_prune = int(num_weights * prune_percentage / 100.0)
    
    # Get absolute values
    abs_weights = jnp.abs(weights.flatten())
    
    # Find threshold (use percentile)
    threshold = jnp.percentile(abs_weights, prune_percentage)
    
    # Create mask
    mask = jnp.abs(weights) > threshold
    
    return weights * mask


def test_compressed_model(
    model: RNN,
    task_name: str,
    task_config: Dict,
    num_bits: int = None,
    prune_percentage: float = 0.0,
    batch_size: int = 32,
    num_batches: int = 50,
    verbose: bool = True,
    wandb_log: bool = False,
) -> Dict[str, Any]:
    """Test model with quantization and/or pruning."""
    
    test_key = jax.random.PRNGKey(999)
    
    if verbose:
        print("="*80)
        print(f"Testing compression on: {task_config['name']}")
        print("="*80)
    
    # Original performance
    original_metrics = evaluate_model(model, task_config, test_key, batch_size, num_batches)
    
    if verbose:
        print(f"Original: loss={original_metrics['loss']:.6f}, acc={original_metrics['accuracy']:.4f}")
    
    # Get original parameters
    original_params = nnx.state(model, nnx.Param)
    
    # Apply compression
    prune_key = jax.random.PRNGKey(54321)
    flat_params = jax.tree.leaves(original_params)
    
    total_params = 0
    zero_params = 0
    modified_flat = []
    
    for i, param_array in enumerate(flat_params):
        total_params += param_array.size
        modified = param_array
        
        # Prune
        if prune_percentage > 0:
            param_key = jax.random.fold_in(prune_key, i)
            modified = prune_weights(modified, prune_percentage, param_key)
        
        # Quantize
        if num_bits is not None:
            modified = quantize_weights(modified, num_bits)
        
        zero_params += jnp.sum(jnp.abs(modified) == 0)
        modified_flat.append(modified)
    
    # Update model
    modified_params = jax.tree.unflatten(jax.tree.structure(original_params), modified_flat)
    nnx.update(model, modified_params)
    
    # Test compressed model
    if verbose:
        print(f"Compression: quant={num_bits}-bit, prune={prune_percentage}%")
    
    modified_metrics = evaluate_model(model, task_config, test_key, batch_size, num_batches)
    
    # Restore original
    nnx.update(model, original_params)
    
    # Compute degradation
    loss_deg = ((modified_metrics['loss'] - original_metrics['loss']) / 
                (original_metrics['loss'] + 1e-8) * 100)
    acc_deg = ((original_metrics['accuracy'] - modified_metrics['accuracy']) / 
               (original_metrics['accuracy'] + 1e-8) * 100)
    
    if verbose:
        print(f"Compressed: loss={modified_metrics['loss']:.6f}, acc={modified_metrics['accuracy']:.4f}")
        print(f"Degradation: loss={loss_deg:+.2f}%, acc={acc_deg:+.2f}%")
        print(f"Sparsity: {100*zero_params/total_params:.2f}%")
    
    result = {
        "original_metrics": original_metrics,
        "compressed_metrics": modified_metrics,
        "loss_degradation_pct": float(loss_deg),
        "accuracy_degradation_pct": float(acc_deg),
        "total_params": int(total_params),
        "zero_params": int(zero_params),
        "sparsity_pct": float(100 * zero_params / total_params),
        "num_bits": num_bits,
        "prune_percentage": prune_percentage,
    }
    
    # Log to wandb
    if wandb_log:
        try:
            import wandb
            # prune_percentage + "_" + 
            string_key = f"{num_bits}_{prune_percentage}"
            wandb.log({
                f"compression{string_key}/num_bits": num_bits or 32,
                f"compression{string_key}/prune_percentage": prune_percentage,
                f"compression{string_key}/sparsity_pct": result["sparsity_pct"],
                f"compression{string_key}/original_accuracy": original_metrics['accuracy'],
                f"compression{string_key}/compressed_accuracy": modified_metrics['accuracy'],
                f"compression{string_key}/accuracy_degradation_pct": result["accuracy_degradation_pct"],
                f"compression{string_key}/loss_degradation_pct": result["loss_degradation_pct"],
            })
        except:
            pass
    
    return result