#!/usr/bin/env python3
"""
Utility functions for model saving, loading, and evaluation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pickle

import jax
import jax.numpy as jnp
from flax import nnx

from models import RNN


def compute_loss(model: RNN, inputs: jax.Array, targets: jax.Array, initial_state: Any, task_type: str = "regression") -> jax.Array:
    """Compute loss between model outputs and targets.
    
    Args:
        model: The RNN model
        inputs: Input sequences
        targets: Target sequences
        initial_state: Initial state for the model
        task_type: "regression" or "classification"
    """
    model_output = model(inputs, initial_state)
    predictions = model_output["outputs"]

    if task_type == "classification":
        # Only compute loss on the final time step
        # final_predictions = predictions[:, -1, :]  # (batch_size, output_size)
        # final_targets = targets[:, -1, :]          # (batch_size, output_size)

        # Cross-entropy loss for classification (final step only)
        log_probs = jax.nn.log_softmax(predictions, axis=-1)
        loss = -jnp.mean(jnp.sum(targets * log_probs, axis=-1))
        
    else:
        # MSE loss for regression (all steps)
        loss = jnp.mean((predictions - targets) ** 2)

    return loss


def compute_accuracy(predictions: jax.Array, targets: jax.Array, task_type: str = "regression", 
                    threshold: float = 0.1, online_mode: bool = False, 
                    transient_frames: int = 10, min_detection_frames: int = 3, majority_vote: bool = False) -> float:
    """Compute accuracy based on task type.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: "regression" or "classification"
        threshold: Threshold for regression accuracy
        online_mode: If True, use strict frame-by-frame accuracy for online tasks
        transient_frames: Number of initial frames to ignore (for online mode)
        min_detection_frames: Minimum consecutive frames to count as valid detection (for online mode)
    
    Returns:
        Accuracy as float between 0 and 1
    """
    if task_type == "classification":
        if online_mode:
            # ONLINE ACCURACY: Trigger-based detection per sequence with temporal filtering
            # Success (100%): Keyword detected during its occurrence (with margin)
            # Fail (0%): Any sustained false positive OR keyword missed
            # NEW: Requires min_detection_frames consecutive detections to count
            
            # Get predicted classes at each timestep
            pred_classes = jnp.argmax(predictions, axis=-1)  # (batch_size, seq_length)
            true_classes = jnp.argmax(targets, axis=-1)      # (batch_size, seq_length)
            
            batch_size, seq_length = pred_classes.shape
            
            # Ignore transient frames at the beginning
            if transient_frames > 0:
                pred_classes = pred_classes[:, transient_frames:]
                true_classes = true_classes[:, transient_frames:]
                seq_length = pred_classes.shape[1]
            
            correct_sequences = []
            
            for b in range(batch_size):
                # Find where the keyword actually occurs (true_classes > 0)
                keyword_frames = jnp.where(true_classes[b] > 0)[0]
                background_frames = jnp.where(true_classes[b] == 0)[0]
                
                # === TEMPORAL FILTERING: Remove isolated spikes ===
                # Only count detections that last at least min_detection_frames
                is_detection = pred_classes[b] > 0
                filtered_detections = jnp.zeros(seq_length, dtype=jnp.bool_)
                
                # Scan through and mark valid detection regions
                i = 0
                while i < seq_length:
                    if is_detection[i]:
                        # Count consecutive detections starting from i
                        consecutive_count = 0
                        j = i
                        while j < seq_length and is_detection[j]:
                            consecutive_count += 1
                            j += 1
                        
                        # If this run is long enough, mark all frames as valid detections
                        if consecutive_count >= min_detection_frames:
                            filtered_detections = filtered_detections.at[i:j].set(True)
                        
                        i = j  # Skip to end of this run
                    else:
                        i += 1
                
                # Check for false positives (using filtered detections)
                has_false_positive = jnp.any(filtered_detections[background_frames])
                
                if len(keyword_frames) > 0:
                    # Keyword is present in this sequence
                    # Check if model detected it during keyword period (with small margin)
                    keyword_start = keyword_frames[0]
                    keyword_end = keyword_frames[-1]
                    
                    # Allow detection slightly after keyword (margin of ~10 frames = 0.1s)
                    detection_window_start = keyword_start
                    detection_window_end = min(keyword_end + 10, seq_length - 1)
                    
                    # Check if there was at least one FILTERED detection in the window
                    detected_in_window = jnp.any(
                        filtered_detections[detection_window_start:detection_window_end+1]
                    )
                    
                    # Success: detected in window AND no (sustained) false positives
                    sequence_correct = detected_in_window & (~has_false_positive)
                else:
                    # No keyword in sequence (all background)
                    # Success: no (sustained) detections at all
                    sequence_correct = ~has_false_positive
                
                correct_sequences.append(sequence_correct)
            
            # Accuracy = percentage of correct sequences
            accuracy = jnp.mean(jnp.array(correct_sequences))
            return float(accuracy)
        elif majority_vote:
            # MAJORITY VOTE ACCURACY: Count predictions across entire sequence
            # The class predicted most often wins for each sequence
            pred_classes = jnp.argmax(predictions, axis=-1)  # (batch_size, seq_length)
            true_classes = jnp.argmax(targets, axis=-1)      # (batch_size, seq_length)
            
            # Get the final true class (constant across sequence for classification)
            final_true_classes = true_classes[:, -1]  # (batch_size,)
            
            # Count votes for each class per sequence
            batch_size = pred_classes.shape[0]
            num_classes = predictions.shape[-1]
            
            # One-hot encode predictions and sum across time
            pred_one_hot = jax.nn.one_hot(pred_classes, num_classes)  # (batch, seq, num_classes)
            class_counts = jnp.sum(pred_one_hot, axis=1)  # (batch, num_classes)
            
            # Majority class per sequence
            majority_pred = jnp.argmax(class_counts, axis=-1)  # (batch_size,)
            
            correct = jnp.mean(majority_pred == final_true_classes)
            return float(correct)
        else:
            # STANDARD ACCURACY: Only final timestep (for regular tasks)
            final_predictions = predictions[:, -1, :]  # (batch_size, num_classes)
            final_targets = targets[:, -1, :]          # (batch_size, num_classes)

            pred_classes = jnp.argmax(final_predictions, axis=-1)
            true_classes = jnp.argmax(final_targets, axis=-1)
            correct = jnp.mean(pred_classes == true_classes)
            return float(correct)
    else:
        # Regression accuracy: percentage of outputs within threshold
        errors = jnp.abs(predictions - targets)
        correct = jnp.mean(errors < threshold)
        return float(correct)


def save_model(model: RNN, path: str, epoch: int, metrics: Dict[str, float],
               task_name: str, config: Dict[str, Any]):
    """Save model checkpoint with configuration."""
    from tasks import BENCHMARK_TASKS
    
    save_dir = Path(path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model state and configuration
    model_state = nnx.state(model)

    # Extract model parameters from config
    task_config = BENCHMARK_TASKS[task_name]
    task_params = task_config["default_params"]

    model_info = {
        'model_state': model_state,
        'epoch': epoch,
        'metrics': metrics,
        'task_name': task_name,
        'config': config,
        'model_params': {
            'input_size': task_params["input_size"],
            'output_size': task_params["output_size"],
            'model_dim': config.get("model", {}).get("model_dim", 64),
            'cells': config.get("model", {}).get("cells", ["sbc"]),
            'cell_configs': config.get("cell_configs", {}),
            'num_recs': config.get("model", {}).get("num_recs", 2),
            'positional_encodings_dims': config.get("model", {}).get("positional_encodings_dims", 2),
            'skip': config.get("model", {}).get("skip", True),
            'num_fcs': config.get("model", {}).get("num_fcs", 1),
        }
    }

    with open(path, 'wb') as f:
        pickle.dump(model_info, f)

    print(f"Saved model checkpoint: {path}")


def load_model(path: str) -> Tuple[RNN, Dict[str, Any]]:
    """Load model from checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    # Extract model parameters
    model_params = checkpoint['model_params']

    # Create dummy rngs for loading (will be replaced by loaded state)
    dummy_rngs = nnx.Rngs(0)

    # Recreate model with correct architecture
    model = RNN(
        input_size=model_params['input_size'],
        output_size=model_params['output_size'],
        rngs=dummy_rngs,
        model_dim=model_params['model_dim'],
        cells=model_params['cells'],
        cell_configs=model_params['cell_configs'],
        num_recs=model_params['num_recs'],
        positional_encodings_dims=model_params['positional_encodings_dims'],
        skip=model_params['skip'],
        num_fcs=model_params['num_fcs'],
    )

    # Load the saved state
    nnx.update(model, checkpoint['model_state'])

    return model, checkpoint