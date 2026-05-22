"""Robustness testing: evaluate model under multiplicative parameter noise."""

import numpy as np
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from mru.models import RNN
from .trainer import evaluate_model


def robustness_test(
    model: RNN,
    task_config: Dict,
    eval_key: jax.Array,
    noise_levels: Sequence[float] = (0.05, 0.10, 0.15),
    n_trials: int = 10,
    batch_size: int = 32,
    num_eval_batches: int = 100,
) -> Dict[float, Dict]:
    """
    Evaluate robustness to parameter mismatch via multiplicative Gaussian noise.

    For each noise level p (e.g. 0.05 = 5%):
        sigma_rel = p / 3
        noisy_param = nominal * (1 + N(0, sigma_rel))

    This matches the ±p = 3σ convention: 99.7% of samples are within ±p of the
    nominal value.  Applies independently to every scalar in every parameter
    tensor (weights, biases, and cell-specific parameters such as alpha, beta1,
    delta/width for CMOS_BMRU; log_nu, log_theta, B, C for LRU; etc.).

    Args:
        model:             Trained model.
        task_config:       Task configuration dict from BENCHMARK_TASKS.
        eval_key:          JAX random key used for test-data generation.
        noise_levels:      Noise levels as fractions (e.g. [0.05, 0.10, 0.15]).
        n_trials:          Number of independent noisy evaluations per noise level.
        batch_size:        Batch size for evaluation.
        num_eval_batches:  Number of mini-batches for each evaluation.

    Returns:
        Dict mapping each noise_level (float) -> {
            'mean_accuracy', 'std_accuracy',
            'mean_loss',     'std_loss',
            'accuracies',    'losses'   (per-trial lists)
        }
    """
    original_state = nnx.state(model)

    results: Dict[float, Dict] = {}

    for sigma_pct in noise_levels:
        sigma_rel = float(sigma_pct) / 3.0
        trial_accs: List[float] = []
        trial_losses: List[float] = []

        for trial in range(n_trials):
            # Derive a reproducible, unique seed for this (noise_level, trial).
            fold_key = jax.random.fold_in(eval_key, int(sigma_pct * 10000) * n_trials + trial)
            noise_seed = int(jax.random.randint(fold_key, (), 0, 2**31 - 1))
            rng = np.random.default_rng(seed=noise_seed)

            def _add_noise(x):
                """Apply element-wise multiplicative Gaussian noise to numeric leaves.
                PRNG keys, integers, and other non-numeric leaves are returned as-is."""
                if not (jnp.issubdtype(x.dtype, jnp.floating)
                        or jnp.issubdtype(x.dtype, jnp.complexfloating)):
                    return x
                x_np = np.array(x)
                noise = rng.normal(0.0, sigma_rel, size=x_np.shape)
                return jnp.array(x_np * (1.0 + noise), dtype=x.dtype)

            noisy_state = jax.tree.map(_add_noise, original_state)
            nnx.update(model, noisy_state)

            metrics = evaluate_model(
                model, task_config, eval_key,
                batch_size=batch_size,
                num_eval_batches=num_eval_batches,
                wandb_log=False,
                prefix="test",
            )
            trial_accs.append(metrics["accuracy"])
            trial_losses.append(metrics["loss"])

        # Restore nominal parameters before the next noise level.
        nnx.update(model, original_state)

        mean_acc  = float(np.mean(trial_accs))
        std_acc   = float(np.std(trial_accs))
        mean_loss = float(np.mean(trial_losses))
        std_loss  = float(np.std(trial_losses))

        print(
            f"  σ={sigma_pct * 100:.0f}%: "
            f"acc={mean_acc:.4f}±{std_acc:.4f}  "
            f"loss={mean_loss:.6f}±{std_loss:.6f}"
        )

        results[sigma_pct] = {
            "mean_accuracy": mean_acc,
            "std_accuracy":  std_acc,
            "mean_loss":     mean_loss,
            "std_loss":      std_loss,
            "accuracies":    trial_accs,
            "losses":        trial_losses,
        }

    return results
