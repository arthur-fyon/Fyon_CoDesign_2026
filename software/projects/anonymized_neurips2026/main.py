#!/usr/bin/env python3
"""
Main entry point for training, testing, and overfitting tests.
"""

import argparse
import json
import math
from pathlib import Path
import wandb

from benchmarks.registry import BENCHMARK_TASKS
from mru.training.trainer import train_model, evaluate_model
from mru.training.overfitting import run_overfitting_test
from mru.training.compression import test_compressed_model
from mru.training.checkpointing import save_model, load_model


# -------------------------
# LM text generation
# -------------------------
def generate_lm_samples(
    model,
    int_to_char: dict,
    char_to_int: dict,
    vocab_size: int,
    prompt: str = "\n",
    num_chars: int = 500,
    temperature: float = 1.0,
    num_samples: int = 3,
    key=None,
) -> list[str]:
    """
    Generate text autoregressively using stateful token-by-token generation (Option B).

    Strategy:
      1. Process the full prompt in a single forward pass to get the final recurrent
         states and conv buffers (O(T) via parallel scan, no padding needed).
      2. Generate one token at a time via model.step_generate(), carrying state
         forward explicitly (O(1) per step, zero padding).

    This is strictly correct for all cell types including CMRU/aCMRU: the prompt is
    processed exactly as during training (no padding contamination), and each generated
    token sees a clean state derived only from real tokens.
    """
    import jax
    import jax.numpy as jnp

    if key is None:
        key = jax.random.PRNGKey(42)

    samples = []
    for _ in range(num_samples):
        key, sample_key = jax.random.split(key)

        # Encode prompt (skip chars not in vocabulary)
        ctx_ids = [char_to_int[c] for c in prompt if c in char_to_int]
        if not ctx_ids:
            ctx_ids = [0]

        # --- Phase 1: process prompt in one shot to get carry state ---
        prompt_oh = jax.nn.one_hot(
            jnp.array(ctx_ids, dtype=jnp.int32), vocab_size
        )[None, :, :]  # (1, T, vocab_size)

        initial_state = model.init_state(1)
        result = model(prompt_oh, initial_state, training=False, return_carry=True)
        rec_states = result["final_rec_states"]
        conv_buffers = result["final_conv_buffers"]

        # Use the logit from Phase 1 directly for the first generated token.
        # This avoids re-feeding the last prompt token (which the carry state
        # already encodes), which would corrupt the conv buffer and cause both
        # cells in hybrid models to diverge from the start.
        logits = result["outputs"][0, -1, :]  # (vocab_size,) — prediction after last prompt token

        # --- Phase 2: generate token by token ---
        generated: list[str] = []
        for _ in range(num_chars):
            sample_key, step_key = jax.random.split(sample_key)
            next_idx = int(jax.random.categorical(step_key, logits / temperature))
            generated.append(int_to_char[next_idx])

            token_oh = jax.nn.one_hot(
                jnp.array([[next_idx]], dtype=jnp.int32), vocab_size
            ).astype(jnp.float32)  # (1, 1, vocab_size)

            logits, rec_states, conv_buffers = model.step_generate(
                token_oh, rec_states, conv_buffers
            )

        samples.append("".join(generated))

    return samples

import os

# -------------------------
# Metadata extraction (unified)
# -------------------------
def extract_arch_metadata(config: dict, task: str) -> dict:
    """Extract and normalize model architecture metadata including new gMRU parameters."""

    model_cfg = config.get("model", {})
    optimizer_cfg = config.get("optimizer", {})

    cells = model_cfg.get("cells", ["sbc"])
    cell_type = "_".join(cells) if isinstance(cells, list) else str(cells)
    seed = config.get("seed", 42)
    num_recs = model_cfg.get("num_recs", "x")
    model_dim = model_cfg.get("model_dim", "x")

    # Get first cell name for cell-specific config lookup
    first_cell = cells[0] if isinstance(cells, list) else cell_type
    cell_config = config.get("cell_configs", {}).get(first_cell, {})

    # Architecture fields
    state_dim = model_cfg.get("state_dim")
    if not state_dim:
        state_dim = cell_config.get("state_dim", "none")

    # Epsilon (from cell config)
    epsilon = cell_config.get("epsilon", "none")
    epsilon_decay = config.get("epsilon_decay", False)
    if epsilon_decay:
        epsilon = "decaying"

    # Activation (from cell config, for gating function in MLP-based cells)
    activation = cell_config.get("activation", "none")

    # Normalization
    norm = model_cfg.get("norm", "batch")

    # MLP parameters
    mlp_activation = model_cfg.get("mlp_activation", "gelu")
    mlp_hidden_dim = model_cfg.get("mlp_hidden_dim", model_dim)
    mlp_dropout = model_cfg.get("mlp_dropout", 0.0)

    # Training hyperparameters
    num_epochs = config.get("num_epochs", "default")
    learning_rate = config.get("learning_rate", "default")
    weight_decay = optimizer_cfg.get("weight_decay", "default")
    batch_size = config.get("batch_size", "default")

    # Cell-specific extras (HP-tuned params not covered by seeder)
    gate_bias_init = cell_config.get("gate_bias_init", None)   # minGRU
    r_min = cell_config.get("r_min", None)                     # LRU
    r_max = cell_config.get("r_max", None)                     # LRU
    max_phase = cell_config.get("max_phase", None)             # LRU

    skip_connections = model_cfg.get("skip", True)
    aggregate = model_cfg.get("aggregate", "sum")

    # Comprehensive tags
    tags = [
        cell_type,
        f"seed_{seed}",
        f"recs_{num_recs}",
        f"dim_{model_dim}",
        f"state_{state_dim}",
        f"norm_{norm}",
        task,
    ]

    # Add epsilon tag
    if epsilon == "learnable":
        tags.append("epsilon_learnable")
    elif epsilon != "none":
        tags.append(f"epsilon_{epsilon}")

    # Add activation tag
    if activation != "none":
        tags.append(f"activation_{activation}")

    # Add mlp tags
    if mlp_activation != "gelu":
        tags.append(f"mlp_{mlp_activation}")

    if mlp_dropout > 0.0:
        tags.append(f"dropout_{mlp_dropout}")

    # Add training hyperparameter tags if non-default
    if learning_rate != "default":
        tags.append(f"lr_{learning_rate}")

    if weight_decay != "default":
        tags.append(f"wd_{weight_decay}")

    if batch_size != "default":
        tags.append(f"bs_{batch_size}")

    # Cell-specific extra tags
    if gate_bias_init is not None:
        tags.append(f"gbias_{gate_bias_init:.4g}")
    if r_min is not None:
        tags.append(f"rmin_{r_min:.4g}")
    if r_max is not None:
        tags.append(f"rmax_{r_max:.4g}")
    if max_phase is not None:
        tags.append(f"phase_{max_phase:.4g}")

    # Group (for aggregating across seeds)
    group_parts = [
        cell_type,
        f"r{num_recs}",
        f"d{model_dim}",
        f"norm{norm}",
        task,
    ]
    
    # Add state_dim to group if different
    if state_dim != "none" and state_dim != model_dim:
        group_parts.append(f"s{state_dim}")
    
    # Add epsilon to group
    if epsilon == "learnable":
        group_parts.append("eps_learn")
    elif epsilon != "none":
        group_parts.append(f"e{epsilon}")
    
    # Add activation to group
    if activation != "none":
        group_parts.append(f"act{activation}")
    
    # Add mlp parameters to group
    if mlp_activation != "gelu":
        group_parts.append(f"mlp{mlp_activation}")
    
    if mlp_dropout > 0.0:
        group_parts.append(f"drop{mlp_dropout}")
    
    # Add training params to group if non-default
    if learning_rate != "default":
        group_parts.append(f"lr{learning_rate}")
    
    if weight_decay != "default":
        group_parts.append(f"wd{weight_decay}")
    
    group_seed = "_".join(group_parts)

    # Flattened config for wandb
    wandb_config = {
        # Model architecture
        "model_type": cell_type,
        "cells": cells,
        "num_recs": num_recs,
        "model_dim": model_dim,
        "state_dim": state_dim,
        "epsilon": epsilon,
        "activation": activation,

        # Architecture parameters
        "norm": norm,
        "mlp_hidden_dim": mlp_hidden_dim,
        "mlp_activation": mlp_activation,
        "mlp_dropout": mlp_dropout,
        "skip_connections": skip_connections,
        "aggregate": aggregate,

        # Training hyperparameters
        "seed": seed,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,

        # Cell-specific extras
        "gate_bias_init": gate_bias_init,
        "r_min": r_min,
        "r_max": r_max,
        "max_phase": max_phase,

        # Task + grouping
        "task_name": task,
        "group_seed": group_seed,

        # Full config for reference
        "full_config": config,
    }

    return {
        "cell_type": cell_type,
        "tags": tags,
        "group_seed": group_seed,
        "wandb_config": wandb_config,
    }


# -------------------------
# Deterministic sidecar helpers (independent of model_path)
# -------------------------
def sidecar_dir() -> Path:
    """Directory where deterministic run-id sidecars are stored."""
    d = Path("models") / "wandb_ids"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sidecar_path_for_run(run_name: str) -> Path:
    """Return the deterministic sidecar path for a given run_name."""
    return sidecar_dir() / f"{run_name}_wandb_id.txt"


def load_wandb_id_for_run(run_name: str) -> str:
    """Load wandb run id for a run_name, or None if not present."""
    p = sidecar_path_for_run(run_name)
    if p.exists():
        try:
            return p.read_text().strip()
        except Exception:
            return None
    return None


def save_wandb_id_for_run(run_name: str, run_id: str):
    """Save wandb run id for a run_name."""
    p = sidecar_path_for_run(run_name)
    try:
        p.write_text(str(run_id))
        print(f"✓ Saved WandB run ID to sidecar: {p}")
    except Exception as e:
        print(f"⚠️ Failed to save wandb sidecar {p}: {e}")


# -------------------------
# Map CLI mode -> job_type used in old script
# -------------------------
def job_type_for_mode(mode: str) -> str:
    mapping = {
        "train": "training",
        "test": "testing",
        "plot_test": "plotting",
        "overfit": "overfitting",
        "run_sanity_check": "sanity_check",
        "compress": "compressing",
        "benchmark": "benchmarking",
        "robustness": "robustness_testing",
        "generate": "generation",
    }
    return mapping.get(mode, mode)


# -------------------------
# WandB init/resume (single run per run_name)
# -------------------------
def _setup_wandb_metrics():
    """Define custom step metrics for different test types to avoid step conflicts."""
    if wandb.run is None:
        return
    
    # Training and validation use train_epoch
    wandb.define_metric("train_epoch")
    wandb.define_metric("train/*", step_metric="train_epoch")
    wandb.define_metric("val/*", step_metric="train_epoch")
    wandb.define_metric("final/*", step_metric="train_epoch")
    wandb.define_metric("time/*", step_metric="train_epoch")
    
    # Overfitting test uses overfit_step
    wandb.define_metric("overfit_step")
    wandb.define_metric("overfit/*", step_metric="overfit_step")
    
    # Sanity check uses sanity_run (single value)
    wandb.define_metric("sanity_run")
    wandb.define_metric("sanity_check/*", step_metric="sanity_run")
    
    # Benchmark uses benchmark_run (single value)
    wandb.define_metric("benchmark_run")
    wandb.define_metric("benchmark/*", step_metric="benchmark_run")

    # Compression uses compress_run (single value)
    wandb.define_metric("compress_run")
    wandb.define_metric("compress/*", step_metric="compress_run")

    # Robustness uses robustness_run (single value)
    wandb.define_metric("robustness_run")
    wandb.define_metric("robustness/*", step_metric="robustness_run")


def init_or_resume_wandb_for_run(run_name: str, wandb_config: dict, tags: list, group: str, project: str = "DOUBLE_BLIND_REVIEW", entity: str = None, job_type: str = None):
    """
    Initialize or resume a wandb run for the deterministic run_name.
    Uses resume='allow' with the saved run id if present.
    If no saved id exists, creates a new run and saves the id.
    Returns wandb.run.id
    """

    run_id = load_wandb_id_for_run(run_name)

    init_kwargs = dict(
        project=project,
        entity=entity,
        name=run_name,
        config=wandb_config,
        tags=tags,
        group=group,
    )

    if job_type:
        init_kwargs["job_type"] = job_type

    # If run_id exists, pass it and allow resume. This is the old behavior.
    if run_id:
        try:
            # resume='allow' lets wandb resume that id, or create a new run if it cannot.
            wandb.init(id=run_id, resume="allow", **init_kwargs)
            print(f"✓ Attempted resume of WandB run {run_id} for {run_name}")
            print(f"✓ View run at: {wandb.run.get_url()}")
            # Save (in case the run id changed during resume)
            save_wandb_id_for_run(run_name, wandb.run.id)
            _setup_wandb_metrics()
            return wandb.run.id
        except Exception as e:
            # If initialization with id fails for some reason, fall back to fresh init
            print(f"⚠️ wandb.init resume attempt failed for id={run_id}: {e}")
            print("→ Creating a new WandB run instead (fallback).")

    # No existing id, or resume failed -> create a new run
    wandb.init(**init_kwargs)
    sid = wandb.run.id
    print(f"✓ Created new WandB run {sid} ({run_name})")
    print(f"✓ View run at: {wandb.run.get_url()}")
    save_wandb_id_for_run(run_name, sid)
    _setup_wandb_metrics()
    return sid


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="gMRU Model Training and Testing (Unified WandB)")
    parser.add_argument(
        "mode",
        choices=["train", "test", "compress", "robustness", "generate"],
        help="Operation mode",
    )
    parser.add_argument("--task", type=str, default="copy_first_discrete", help="Task name")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    parser.add_argument("--model_path", type=str, help="Path to save/load model (optional)")
    parser.add_argument("--plot_path", type=str, help="Path to save plots")
    parser.add_argument("--num_test", type=int, default=5, help="Samples to plot")
    parser.add_argument("--quantize_bits", type=int, help="Quantization bits")
    parser.add_argument("--prune_percent", type=float, default=0.0, help="Prune percentage")
    parser.add_argument("--num_warmup", type=int, default=5, help="Benchmark warmup iterations")
    parser.add_argument("--num_iterations", type=int, default=50, help="Benchmark timed iterations")
    parser.add_argument("--test_seed", type=int, default=999, help="Random seed for testing")
    parser.add_argument("--wandb_project", type=str, help="WandB project name (overrides config)")
    parser.add_argument(
        "--noise_levels", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        help="Noise levels for robustness test as fractions (e.g. 0.05 = 5%%). "
             "Convention: ±noise_level corresponds to 3σ of the Gaussian.",
    )
    parser.add_argument(
        "--noise_n_trials", type=int, default=10,
        help="Number of independent noisy evaluations per noise level (default: 10).",
    )
    # --- LM generation args ---
    parser.add_argument("--gen_prompt", type=str, default="\n",
        help="Seed text for generation (default: newline).")
    parser.add_argument("--gen_num_chars", type=int, default=500,
        help="Number of characters to generate per sample (default: 500).")
    parser.add_argument("--gen_temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8). Lower = more conservative.")
    parser.add_argument("--gen_num_samples", type=int, default=3,
        help="Number of independent samples to generate (default: 3).")

    args = parser.parse_args()

    # Load config JSON
    with open(args.config, "r") as f:
        config = json.load(f)

    # Extract metadata; run_name is derived directly from the config filename
    # to guarantee uniqueness and full traceability to the config on disk.
    meta = extract_arch_metadata(config, args.task)
    config_stem = Path(args.config).stem
    run_name = config_stem[len("config_"):] if config_stem.startswith("config_") else config_stem
    tags = meta["tags"]
    group_seed = meta["group_seed"]
    wandb_cfg = meta["wandb_config"]

    print(f"Run name: {run_name}")
    

    # WandB settings from config (optional)
    wb_conf = config.get("wandb", {})
    project = wb_conf.get("project", "DOUBLE_BLIND_REVIEW")
    if args.wandb_project:
        project = args.wandb_project
    entity = wb_conf.get("entity", None)

    # Modes that should participate in the unified run
    wandb_participating = {"train", "test", "plot_test", "overfit", "compress", "run_sanity_check", "benchmark", "robustness", "generate"}

    wandb_run_id = None
    if args.mode in wandb_participating:
        # initialize/resume a single wandb run for this run_name
        job_type = job_type_for_mode(args.mode)
        wandb_run_id = init_or_resume_wandb_for_run(
            run_name=run_name,
            wandb_config=wandb_cfg,
            tags=tags,
            group=group_seed,
            project=project,
            entity=entity,
            job_type=job_type,
        )

    # Validate task
    if args.task not in BENCHMARK_TASKS:
        print(f"Unknown task: {args.task}")
        print(f"Available: {list(BENCHMARK_TASKS.keys())}")
        if args.mode in wandb_participating:
            wandb.finish()
        return

    task_config = BENCHMARK_TASKS[args.task]

    # -----------------------
    # Modes
    # -----------------------
    if args.mode == "train":
        print(f"[TRAIN] Task={args.task}")
        save_strategy = "best_loss" if task_config.get("is_language_model", False) else "best_accuracy"
        model, losses, metrics = train_model(
            args.task, task_config, config, wandb_log=True,
            save_best_model=True, model_save_path=args.model_path, model_save_strategy=save_strategy,
        )

        if args.model_path:
            save_model(
                model,
                args.model_path,
                epoch=len(losses),
                task_name=args.task,
                config=config,
                metrics=metrics,
                wandb_id=wandb_run_id,
            )
            # keep sidecar (it is deterministic) but also save an auxiliary copy next to model if desired
            try:
                # auxiliary copy next to model (optional, mirrors old behavior)
                model_sidecar = str(Path(args.model_path).with_suffix("") ) + "_wandb_id.txt"
                with open(model_sidecar, "w") as f:
                    f.write(wandb_run_id)
                print(f"✓ (aux) Saved WandB run ID to: {model_sidecar}")
            except Exception:
                pass

        # Auto-generate text samples from the best checkpoint when training an LM
        if task_config.get("is_language_model", False) and args.model_path:
            best_pkl = str(Path(args.model_path).with_suffix("")) + "_best.pkl"
            if Path(best_pkl).exists():
                print(f"\n=== LM generation from best checkpoint ===")
                try:
                    import jax
                    best_model, _ = load_model(best_pkl)
                    shk = task_config["get_vocab"]()
                    samples = generate_lm_samples(
                        best_model,
                        shk["int_to_char"], shk["char_to_int"], shk["vocab_size"],
                        prompt="\n", num_chars=500, temperature=0.8,
                        num_samples=3, key=jax.random.PRNGKey(0),
                    )
                    gen_prompt = "\n"
                    for i, s in enumerate(samples, 1):
                        print(f"\n--- Sample {i} ---")
                        print(f"[PROMPT]    {repr(gen_prompt)}")
                        print(f"[GENERATED]\n{s}")
                    try:
                        table = wandb.Table(columns=["sample", "prompt", "generated"])
                        for i, s in enumerate(samples, 1):
                            table.add_data(i, repr(gen_prompt), s)
                        wandb.log({"generated_samples": table})
                    except Exception:
                        pass
                except Exception as e:
                    print(f"  Warning: generation failed: {e}")

    elif args.mode == "test":
        if not args.model_path:
            print("Error: --model_path required for test mode")
            if args.mode in wandb_participating:
                wandb.finish()
            return

        print(f"[TEST] Task={args.task}")
        model, checkpoint = load_model(args.model_path)

        import jax
        metrics = evaluate_model(
            model, task_config, jax.random.PRNGKey(args.test_seed),
            batch_size=32, num_eval_batches=100,
            wandb_log=True,
            prefix="test",
            wandb_prefix="test" if args.test_seed == 999 else f"test_seed{args.test_seed}",
        )

        print(f"Test loss: {metrics['loss']:.6f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if task_config.get("is_language_model", False):
            bpc = metrics["loss"] / math.log(2)
            print(f"BPC: {bpc:.4f}")
            wandb.log({"test/bpc": bpc, "benchmark_run": 0})
    elif args.mode == "compress":
        if not args.model_path:
            print("Error: --model_path required for compress mode")
            if args.mode in wandb_participating:
                wandb.finish()
            return

        print(f"[COMPRESS] Task={args.task}")
        model, checkpoint = load_model(args.model_path)

        result = test_compressed_model(
            model, args.task, task_config,
            num_bits=args.quantize_bits,
            prune_percentage=args.prune_percent,
            batch_size=32,
            num_batches=50,
            verbose=True,
            wandb_log=True,
        )

        print(f"\nCompression results:")
        print(f"  Original accuracy: {result['original_metrics']['accuracy']:.4f}")
        print(f"  Compressed accuracy: {result['compressed_metrics']['accuracy']:.4f}")
        print(f"  Degradation: {result['accuracy_degradation_pct']:.2f}%")

    elif args.mode == "robustness":
        if not args.model_path:
            print("Error: --model_path required for robustness mode")
            if args.mode in wandb_participating:
                wandb.finish()
            return

        print(f"[ROBUSTNESS] Task={args.task}")
        print(f"  Noise levels: {[f'{p*100:.0f}%' for p in args.noise_levels]}")
        print(f"  Trials per level: {args.noise_n_trials}")

        model, checkpoint = load_model(args.model_path)

        import jax
        from mru.training.robustness import robustness_test

        eval_key = jax.random.PRNGKey(args.test_seed)

        # Baseline: nominal parameters
        metrics_nominal = evaluate_model(
            model, task_config, eval_key,
            batch_size=32, num_eval_batches=100,
            wandb_log=False,
            prefix="test",
        )
        print(f"Nominal accuracy: {metrics_nominal['accuracy']:.4f}  "
              f"loss: {metrics_nominal['loss']:.6f}")

        # Robustness sweep
        results = robustness_test(
            model, task_config,
            eval_key=eval_key,
            noise_levels=args.noise_levels,
            n_trials=args.noise_n_trials,
            batch_size=32,
            num_eval_batches=100,
        )

        # Log everything to wandb in a single step
        log_dict = {
            "robustness/nominal_accuracy": metrics_nominal["accuracy"],
            "robustness/nominal_loss":     metrics_nominal["loss"],
            "robustness_run": 0,
        }
        for sigma_pct, res in results.items():
            key_prefix = f"robustness/sigma{int(sigma_pct * 100)}"
            log_dict[f"{key_prefix}_mean_acc"]  = res["mean_accuracy"]
            log_dict[f"{key_prefix}_std_acc"]   = res["std_accuracy"]
            log_dict[f"{key_prefix}_mean_loss"] = res["mean_loss"]
            log_dict[f"{key_prefix}_std_loss"]  = res["std_loss"]
        wandb.log(log_dict)

        # Save results to JSON file next to the model (for offline plotting)
        import json as _json
        arch = config.get("model", {}).get("cells", ["unknown"])[0]
        seed = config.get("seed", None)
        json_payload = {
            "model_path":        args.model_path,
            "config":            args.config,
            "task":              args.task,
            "arch":              arch,
            "seed":              seed,
            "noise_levels":      list(args.noise_levels),
            "n_trials":          args.noise_n_trials,
            "nominal_accuracy":  metrics_nominal["accuracy"],
            "nominal_loss":      metrics_nominal["loss"],
            "results": {
                str(k): {
                    "mean_accuracy": v["mean_accuracy"],
                    "std_accuracy":  v["std_accuracy"],
                    "mean_loss":     v["mean_loss"],
                    "std_loss":      v["std_loss"],
                    "accuracies":    v["accuracies"],
                    "losses":        v["losses"],
                }
                for k, v in results.items()
            },
        }
        json_path = Path(args.model_path).with_suffix("").with_suffix("") \
            .with_name(Path(args.model_path).stem.replace("_best", "") + "_robustness.json")
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as _f:
                _json.dump(json_payload, _f, indent=2)
            print(f"Robustness results saved to: {json_path}")
        except Exception as _e:
            print(f"WARNING: could not save robustness JSON: {_e}")

        print("\nRobustness summary:")
        print(f"  Nominal:  acc={metrics_nominal['accuracy']:.4f}")
        for sigma_pct, res in results.items():
            print(f"  {sigma_pct*100:.0f}%:     "
                  f"acc={res['mean_accuracy']:.4f}±{res['std_accuracy']:.4f}")

    elif args.mode == "generate":
        if not args.model_path:
            print("Error: --model_path required for generate mode")
            wandb.finish()
            return
        if not task_config.get("is_language_model", False):
            print("Error: generate mode is only supported for language model tasks")
            wandb.finish()
            return

        print(f"[GENERATE] Task={args.task}, model={args.model_path}")
        print(f"  prompt={repr(args.gen_prompt)}, chars={args.gen_num_chars}, "
              f"T={args.gen_temperature}, samples={args.gen_num_samples}")

        model, _ = load_model(args.model_path)

        import jax
        shk = task_config["get_vocab"]()

        samples = generate_lm_samples(
            model,
            shk["int_to_char"], shk["char_to_int"], shk["vocab_size"],
            prompt=args.gen_prompt,
            num_chars=args.gen_num_chars,
            temperature=args.gen_temperature,
            num_samples=args.gen_num_samples,
            key=jax.random.PRNGKey(args.test_seed),
        )

        for i, s in enumerate(samples, 1):
            print(f"\n{'='*60}")
            print(f"Sample {i}/{args.gen_num_samples}  (T={args.gen_temperature})")
            print(f"{'='*60}")
            print(f"[PROMPT]    {repr(args.gen_prompt)}")
            print(f"[GENERATED]\n{s}")

        try:
            table = wandb.Table(columns=["sample", "temperature", "prompt", "generated"])
            for i, s in enumerate(samples, 1):
                table.add_data(i, args.gen_temperature, repr(args.gen_prompt), s)
            wandb.log({"generated_samples": table, "benchmark_run": 0})
        except Exception:
            pass

    # Finish WandB only if participated
    if args.mode in wandb_participating:
        wandb.finish()


if __name__ == "__main__":
    import jax
    print(jax.devices(), flush=True)
    main()