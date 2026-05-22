"""Training loop."""

from typing import Dict, List, Tuple, Callable, Optional
import math
import time
from pathlib import Path

import jax
from flax import nnx
import optax

from mru.models import RNN
from .optimizer import train_step, eval_step_jit, compute_accuracy_on_batch, get_surr_alpha_scale_factor, get_epsilon_decay_value
from .checkpointing import save_model


def count_parameters(model: RNN) -> Dict[str, int]:
    """Count total and trainable parameters in the model."""
    state = nnx.state(model)
    params = jax.tree.leaves(state)
    
    total_params = sum(p.size for p in params)
    trainable_params = total_params  # All params are trainable in this setup
    
    return {
        "total": total_params,
        "trainable": trainable_params,
    }


def create_optimizer(
    config: Dict,
    model: Optional[RNN] = None,
) -> Tuple[optax.GradientTransformation, Callable[[int], float]]:
    """
    Creates optimizer and returns LR schedule function.
    
    If model is provided and optimizer uses weight decay (AdamW, SGD with weight_decay),
    weight decay will be disabled for recurrent parameters while keeping the same learning rate.
    
    Args:
        config: Training configuration
        model: Optional model to inspect for recurrent parameters
    
    Returns:
        Tuple of (optimizer, lr_schedule_function)
    """
    optimizer_config = config.get("optimizer", {"type": "adam"})
    opt_type = optimizer_config.get("type", "adam").lower()
    learning_rate = config.get("learning_rate", 1e-3)
    num_epochs = config.get("num_epochs", 300)
    
    gradient_clip_norm = config.get("gradient_clip_norm", -1.0)
    use_mixed_precision = config.get("use_mixed_precision", False)

    lr_schedule_config = config.get("lr_schedule", {})
    warmup_epochs = lr_schedule_config.get("warmup_epochs", 0)
    min_lr_factor = lr_schedule_config.get("min_lr_factor", 0.0)

    # Support warmup_epochs as percentage (0-1) or absolute number
    if 0 < warmup_epochs < 1:
        # Treat as percentage
        warmup_steps = int(warmup_epochs * num_epochs)
        print(f"  Warmup: {warmup_epochs*100:.1f}% of training ({warmup_steps} epochs)")
    else:
        # Treat as absolute number
        warmup_steps = int(warmup_epochs)
        if warmup_steps > 0:
            print(f"  Warmup: {warmup_steps} epochs ({warmup_steps/num_epochs*100:.1f}%)")
    
    total_steps = num_epochs

    if warmup_steps > 0:
        warmup = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps)
        cosine = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=max(total_steps - warmup_steps, 1), alpha=min_lr_factor)
        schedule_fn = optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])
    else:
        schedule_fn = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=max(total_steps, 1), alpha=min_lr_factor)

    opt_kwargs = {k: v for k, v in optimizer_config.items() if k != "type"}

    # Determine if we're using an optimizer with weight decay
    has_weight_decay = opt_type in ["adamw", "sgd"] and opt_kwargs.get("weight_decay", 0.0) > 0.0

    # Flags to control which params are excluded from weight decay
    exclude_rec_from_weight_decay = config.get("exclude_rec_from_weight_decay", False)
    exclude_theta_from_weight_decay = config.get("exclude_theta_from_weight_decay", False)

    # Collect parameter names to exclude from weight decay
    no_wd_param_names: set = set()
    if model is not None and has_weight_decay:
        for rec_layer in model._recs:
            for cell_name, cell in rec_layer.items():
                if exclude_rec_from_weight_decay and hasattr(cell, 'rec_params'):
                    no_wd_param_names.update(cell.rec_params())
                if exclude_theta_from_weight_decay and hasattr(cell, 'no_weight_decay_params'):
                    no_wd_param_names.update(cell.no_weight_decay_params())

    # Create weight decay mask if there are params to exclude
    if model is not None and has_weight_decay and no_wd_param_names:
        # Create mask function that returns True for parameters that SHOULD have weight decay
        def weight_decay_mask(path):
            """Return True if parameter should have weight decay, False otherwise."""
            for name in no_wd_param_names:
                if name in path:  # Exact match in path tuple
                    return False
            return True

        # Create optimizer with masked weight decay
        weight_decay_value = opt_kwargs.pop("weight_decay", 0.01)

        if opt_type == "adamw":
            base_adam = optax.adam(schedule_fn, **opt_kwargs)
            masked_weight_decay = optax.add_decayed_weights(
                weight_decay_value,
                mask=weight_decay_mask
            )
            base_optimizer = optax.chain(masked_weight_decay, base_adam)
        elif opt_type == "sgd":
            base_sgd = optax.sgd(schedule_fn, **opt_kwargs)
            masked_weight_decay = optax.add_decayed_weights(
                weight_decay_value,
                mask=weight_decay_mask
            )
            base_optimizer = optax.chain(masked_weight_decay, base_sgd)

        print(f"  Weight decay: excluded {len(no_wd_param_names)} param type(s): {sorted(no_wd_param_names)}", flush=True)
    else:
        # Standard optimizer creation (no model provided, no weight decay, or masking disabled)
        if opt_type == "adam":
            base_optimizer = optax.adam(schedule_fn, **opt_kwargs)
        elif opt_type == "adamw":
            base_optimizer = optax.adamw(schedule_fn, **opt_kwargs)
        elif opt_type == "sgd":
            base_optimizer = optax.sgd(schedule_fn, **opt_kwargs)
        elif opt_type == "rmsprop":
            base_optimizer = optax.rmsprop(schedule_fn, **opt_kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        if has_weight_decay:
            print(f"  Weight decay: enabled for all parameters")
    
    transformations = []
    
    if gradient_clip_norm > 0:
        transformations.append(optax.clip_by_global_norm(gradient_clip_norm))
        print(f"  Gradient clipping enabled: max_norm={gradient_clip_norm}")
    
    if use_mixed_precision:
        transformations.append(optax.apply_if_finite(base_optimizer, max_consecutive_errors=5))
        print(f"  Mixed precision training enabled with dynamic loss scaling")
    else:
        transformations.append(base_optimizer)
    
    if len(transformations) > 1:
        optimizer_tx = optax.chain(*transformations)
    else:
        optimizer_tx = transformations[0]

    return optimizer_tx, schedule_fn


def evaluate_model(
    model: RNN,
    task_config: Dict,
    eval_key: jax.Array,
    batch_size: int,
    num_eval_batches: int = 100,
    return_infos: bool = False,
    wandb_log: bool = False,
    epoch: int = 0,
    prefix: str = "val",
    wandb_prefix: str = None,
) -> Dict[str, float]:
    """Evaluate model on task."""
    
    total_loss = 0.0
    total_accuracy = 0.0
    infos_dict = None
    
    data_fn = task_config["data_fn"]
    task_params = task_config["default_params"]
    task_type = task_config.get("task_type", "regression")
    compute_loss_only_last = task_config.get("compute_loss_only_last_timestep", False)
    compute_acc_only_last = task_config.get("compute_accuracy_only_last_timestep", False)
    use_mean_pooling_loss = task_config.get("use_mean_pooling_loss", False)
    use_mean_pooling_acc = task_config.get("use_mean_pooling_accuracy", False)
    compute_seq_acc = task_config.get("compute_sequence_accuracy", False)

    # Padding configuration
    is_padded = task_config.get("is_padded", False)
    padding_value = task_config.get("padding_value", 0)

    kwargs = {}
    if "should_split" in task_config:
        kwargs["split"] = prefix
    if "delay_steps" in task_params:
        kwargs["delay_steps"] = task_params["delay_steps"]
    
    # Tasks that return the full split in a single call (e.g. Shakespeare val/test
    # deterministic sweep) set eval_single_pass=True to avoid redundant iterations.
    eval_single_pass = task_config.get("eval_single_pass", False) and prefix != "train"
    actual_eval_batches = 1 if eval_single_pass else num_eval_batches

    for i in range(actual_eval_batches):
        key = jax.random.fold_in(eval_key, i)

        inputs, targets = data_fn(
            batch_size, task_params["seq_length"], task_params["input_size"], key, **kwargs
        )

        # Use actual returned batch size (val/test may return all chunks at once)
        actual_batch_size = inputs.shape[0]
        initial_state = model.init_state(actual_batch_size)
        
        loss, accuracy, infos = eval_step_jit(
            model, inputs, targets, initial_state,
            task_type, compute_loss_only_last, compute_acc_only_last,
            use_mean_pooling_loss, use_mean_pooling_acc,
            return_infos=(return_infos and i == 0),
            use_padding_mask=is_padded,
            padding_token_idx=padding_value,
            compute_sequence_accuracy=compute_seq_acc,
        )
        
        if return_infos and i == 0 and infos:
            infos_dict = {}
            for rec_idx, rec_info in enumerate(infos):
                for cell_name, cell_data in rec_info.items():
                    for key_name, value in cell_data.items():
                        if value is not None:
                            flat_value = value.reshape(-1)
                            new_key = f"{prefix}/{key_name}_rec{rec_idx}_{cell_name}"
                            infos_dict[new_key] = flat_value
        
        total_loss += float(loss)
        total_accuracy += float(accuracy)

    result = {
        "loss": total_loss / actual_eval_batches,
        "accuracy": total_accuracy / actual_eval_batches,
    }
    
    if wandb_prefix is not None:
        prefix = wandb_prefix

    if wandb_log:
        try:
            import wandb
            log_dict = {
                "epoch": epoch,
                f"{prefix}/loss": result["loss"],
                f"{prefix}/accuracy": result["accuracy"],
            }
            if infos_dict:
                log_dict.update(infos_dict)
            wandb.log(log_dict)
        except Exception as e:
            print(f"  Warning: wandb logging failed: {e}")

    return result


def train_model(
    task_name: str,
    task_config: Dict,
    config: Dict,
    wandb_log: bool = True,
    save_best_model: bool = True,
    model_save_path: Optional[str] = None,
    model_save_strategy: str = "best_loss",
    epoch_callback: Optional[Callable[["RNN", int], None]] = None,
) -> Tuple[RNN, List[float], Dict]:
    """Train model on task.
    
    Args:
        task_name: Name of the task
        task_config: Task configuration dictionary
        config: Training configuration dictionary
        wandb_log: Whether to log metrics to wandb
        save_best_model: Whether to save models during training
        model_save_path: Path to save the model
        model_save_strategy: Strategy for saving models. Options:
            - "best_loss": Save model with best validation loss (default)
            - "best_accuracy": Save model with best validation accuracy
            - "last": Save only the final model after training
            - "both": Save both best_loss and best_accuracy models
    
    Returns:
        Tuple of (trained_model, loss_history, final_metrics)
    """
    
    # Validate model save strategy
    valid_strategies = ["best_loss", "best_accuracy", "last", "both"]
    if model_save_strategy not in valid_strategies:
        raise ValueError(f"Invalid model_save_strategy '{model_save_strategy}'. Must be one of {valid_strategies}")
    
    task_params = task_config["default_params"]
    seq_length = task_params["seq_length"]
    input_size = task_params["input_size"]
    output_size = task_params.get("output_size", input_size)
    task_type = task_config.get("task_type", "regression")
    
    num_epochs = config.get("num_epochs", 300)
    batch_size = config.get("batch_size", 16)
    eval_every = config.get("eval_every", 10)
    eval_num_batches = config.get("eval_num_batches", 20)
    final_eval_num_batches = config.get("final_eval_num_batches", 100)

    train_log_every = max(1, eval_every // 4)

    model_config = config.get("model", {})
    model_dim = model_config.get("model_dim", 64)
    cells = model_config.get("cells", ["sbc"])
    num_recs = model_config.get("num_recs", 2)
    cell_configs = config.get("cell_configs", {})
    
    # Gradient checkpointing config
    use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
    
    # Padding configuration
    is_padded = task_config.get("is_padded", False)
    padding_value = task_config.get("padding_value", 0)
    
    anneal_epochs = config.get("anneal_epochs", 0)
    anneal_final_factor = config.get("anneal_final_factor", 1.0)
    if anneal_final_factor <= 0.0 or anneal_final_factor == 1.0:
        anneal_final_factor = 1.0
        anneal_epochs = 0
    if anneal_epochs < 0: 
        anneal_epochs = num_epochs
    _cumulated_surr_alpha_scale = 1.0
    
    # Epsilon decay configuration (separate from alpha annealing)
    epsilon_decay = config.get("epsilon_decay", False)
    epsilon_const_epochs = config.get("epsilon_const_epochs", None)   # absolute epochs at eps=1
    epsilon_decay_epochs = config.get("epsilon_decay_epochs", None)   # absolute epochs for decay

    # z=1 activation penalty configuration
    z_penalty_coeff = config.get("z_penalty_coeff", 0.0)
    z_penalty_type = config.get("z_penalty_type", "l1")
    
    print(f"Training on {task_name}")
    print(f"  Model: {cells}, {num_recs} layers, dim={model_dim}")
    print(f"  Training: {num_epochs} epochs, batch={batch_size}")
    print(f"  Model save strategy: {model_save_strategy}")
    print(f"  JIT compilation: Enabled (first epoch will compile)")
    
    gradient_clip_norm = config.get("gradient_clip_norm", -1.0)
    use_mixed_precision = config.get("use_mixed_precision", False)
    use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
    if gradient_clip_norm > 0:
        print(f"  Gradient clipping: enabled (max_norm={gradient_clip_norm})")
    else:
        print(f"  Gradient clipping: disabled")
    if use_mixed_precision:
        print(f"  Mixed precision: enabled")
    else:
        print(f"  Mixed precision: disabled")
    if use_gradient_checkpointing:
        print(f"  Gradient checkpointing: enabled (saves memory for long sequences)")
    else:
        print(f"  Gradient checkpointing: disabled")
    if is_padded:
        print(f"  Padding-aware pooling: enabled (padding token idx={padding_value})")
    else:
        print(f"  Padding-aware pooling: disabled")
    
    seed = config.get("seed", 42)
    main_key = jax.random.PRNGKey(seed)
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
        positional_encodings_dims=model_config.get("positional_encodings_dims", 0),
        skip=model_config.get("skip", True),
        aggregate=model_config.get("aggregate", "sum"),
        norm=model_config.get("norm", "batch"),
        post_rec_norm=model_config.get("post_rec_norm", "partial"),
        mlp_hidden_dim=model_config.get("mlp_hidden_dim", model_dim),
        mlp_activation=model_config.get("mlp_activation", "gelu"),
        mlp_dropout=model_config.get("mlp_dropout", 0.1),
        mlp_ff_expansion=model_config.get("mlp_ff_expansion", 4),
        use_gating=model_config.get("use_gating", True),
        conv_kernel_size=model_config.get("conv_kernel_size", 0),
    )
    
    optimizer_tx, lr_fn = create_optimizer(config, model)
    optimizer = nnx.ModelAndOptimizer(model, optimizer_tx)
    
    # Count and display parameters
    param_counts = count_parameters(model)
    print(f"  Model parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")
    print(f"  Cell params at init:")
    model.print_cell_param_stats(prefix="    ")
    
    if wandb_log:
        try:
            import wandb
            wandb.config.update({
                "total_parameters": param_counts['total'],
                "trainable_parameters": param_counts['trainable'],
            })
        except Exception as e:
            print(f"  Warning: wandb config update failed: {e}")
    
    is_language_model = task_config.get("is_language_model", False)

    losses = []
    compute_loss_only_last = task_config.get("compute_loss_only_last_timestep", False)
    compute_acc_only_last = task_config.get("compute_accuracy_only_last_timestep", False)
    use_mean_pooling_loss = task_config.get("use_mean_pooling_loss", False)
    use_mean_pooling_acc = task_config.get("use_mean_pooling_accuracy", False)
    compute_seq_acc = task_config.get("compute_sequence_accuracy", False)
    
    start_time = time.time()
    best_val_loss = float('inf')
    best_val_acc = float('-inf')

    # Early stopping (patience-based)
    es_patience = config.get("early_stop_patience", 20)
    es_min_delta = config.get("early_stop_min_delta", 1e-6)
    es_perfect_acc_threshold = config.get("early_stop_perfect_acc_threshold", 0.9999)
    es_perfect_acc_patience = config.get("early_stop_perfect_acc_patience", 5)
    es_no_improvement_count = 0
    es_best_val = float('inf')
    es_perfect_acc_count = 0

    print(f"  Early stopping: patience={es_patience} eval steps (min_delta={es_min_delta:.0e}), "
          f"perfect_acc={es_perfect_acc_threshold} for {es_perfect_acc_patience} steps")

    if epsilon_decay:
        _c = epsilon_const_epochs if epsilon_const_epochs is not None else int(round(num_epochs * 0.05))
        _d = epsilon_decay_epochs if epsilon_decay_epochs is not None else int(round(num_epochs * 0.7))
        print(f"  Epsilon decay: enabled (hold {_c} epochs, decay {_d} epochs → 0.0)")
    if z_penalty_coeff > 0.0:
        print(f"  z=1 penalty: enabled (coeff={z_penalty_coeff}, type={z_penalty_type})")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        if anneal_epochs > 0:
            scale = get_surr_alpha_scale_factor(epoch, anneal_epochs, anneal_final_factor)
            model.scale_surr_alpha_all(scale)
            _cumulated_surr_alpha_scale *= scale

        if epsilon_decay:
            epsilon_value = get_epsilon_decay_value(
                epoch, num_epochs,
                abs_const_epochs=epsilon_const_epochs,
                abs_decay_epochs=epsilon_decay_epochs,
            )
            model.set_epsilon_all(epsilon_value)

        epoch_key = jax.random.fold_in(data_key, epoch)

        kwargs = {}
        if "delay_steps" in task_params:
            kwargs["delay_steps"] = task_params["delay_steps"]
        if "should_split" in task_config:
            kwargs["split"] = "train"

        inputs, targets = task_config["data_fn"](
            batch_size, seq_length, input_size, epoch_key,
            **kwargs
        )

        initial_state = model.init_state(batch_size)

        loss, metrics = train_step(
            model, optimizer, inputs, targets, initial_state,
            task_type, compute_loss_only_last, use_mean_pooling_loss,
            use_gradient_checkpointing, is_padded, padding_value,
            z_penalty_coeff=z_penalty_coeff, z_penalty_type=z_penalty_type,
        )

        losses.append(loss)
        current_lr = lr_fn(epoch)

        do_log = wandb_log and (epoch % train_log_every == 0 or epoch == num_epochs - 1)
        do_eval = (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1

        if do_eval:
            train_accuracy = float(compute_accuracy_on_batch(
                model, inputs, targets, initial_state,
                task_type, compute_acc_only_last, use_mean_pooling_acc,
                is_padded, padding_value,
                compute_sequence_accuracy=compute_seq_acc,
            ))

            eval_metrics = evaluate_model(
                model, task_config, epoch_key, batch_size, num_eval_batches=eval_num_batches,
                return_infos=False, wandb_log=False, epoch=epoch, prefix="val",
            )

            val_loss = eval_metrics['loss']
            val_acc = eval_metrics['accuracy']

            if epoch_callback is not None:
                try:
                    epoch_callback(model, epoch)
                except Exception as e:
                    print(f"  Warning: epoch_callback failed at epoch {epoch}: {e}")

            elapsed = time.time() - start_time
            epoch_time = time.time() - epoch_start

            if is_language_model:
                val_bpc = val_loss / math.log(2)
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"loss={loss:.6f}, train_acc={train_accuracy:.4f}, "
                      f"val_loss={val_loss:.6f}, val_bpc={val_bpc:.4f}, "
                      f"lr={current_lr:.6f}, "
                      f"epoch_time={epoch_time:.2f}s, total_time={elapsed:.1f}s", flush=True)
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"loss={loss:.6f}, train_acc={train_accuracy:.4f}, "
                      f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}, "
                      f"lr={current_lr:.6f}, "
                      f"epoch_time={epoch_time:.2f}s, total_time={elapsed:.1f}s")
            model.print_cell_param_stats(prefix="  ")

            # Only track best/save when epsilon has decayed (or decay is disabled)
            if not epsilon_decay or (epsilon_decay and epsilon_value < 1e-5):
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    if save_best_model and model_save_path and model_save_strategy in ["best_loss", "both"]:
                        best_model_path = str(Path(model_save_path).with_suffix("")) + "_best.pkl"
                        save_model(model, best_model_path, epoch=epoch, task_name=task_name,
                                   config=config, metrics=eval_metrics, wandb_id=None)
                        print(f"  Saved best loss model to {best_model_path}")

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    if save_best_model and model_save_path and model_save_strategy in ["best_accuracy", "both"]:
                        best_model_path = str(Path(model_save_path).with_suffix("")) + "_best.pkl"
                        save_model(model, best_model_path, epoch=epoch, task_name=task_name,
                                   config=config, metrics=eval_metrics, wandb_id=None)
                        print(f"  Saved best accuracy model to {best_model_path}")

                # Early stopping
                if val_loss < es_best_val - es_min_delta:
                    es_best_val = val_loss
                    es_no_improvement_count = 0
                else:
                    es_no_improvement_count += 1

                if val_acc >= es_perfect_acc_threshold:
                    es_perfect_acc_count += 1
                else:
                    es_perfect_acc_count = 0

                if es_no_improvement_count >= es_patience:
                    print(f"\nEarly stopping: no val_loss improvement for {es_patience} eval steps.")
                    break
                if es_perfect_acc_count >= es_perfect_acc_patience:
                    print(f"\nEarly stopping: val_acc >= {es_perfect_acc_threshold} for {es_perfect_acc_patience} eval steps.")
                    break

        # Build single wandb log dict per epoch
        if do_log or do_eval:
            epoch_log = {
                "epoch": epoch,
                "train/loss": loss,
                "train/grad_norm": metrics["grad_norm"],
                "train/lr": current_lr,
            }
            if wandb_log and do_eval:
                for layer_idx, rec_layer in enumerate(model._recs):
                    for cell_name, cell_obj in rec_layer.items():
                        if hasattr(cell_obj, "get_param_images"):
                            epoch_log.update(cell_obj.get_param_images(
                                prefix=f"params/layer{layer_idx}/{cell_name}"
                            ))
            if z_penalty_coeff > 0.0:
                epoch_log["train/z_mean"] = metrics["z_mean"]
                epoch_log["train/z_penalty"] = z_penalty_coeff * metrics["z_mean"]
            if anneal_epochs > 0:
                epoch_log["train/cumulated_surr_alpha_scale"] = _cumulated_surr_alpha_scale
            if epsilon_decay:
                epoch_log["train/epsilon"] = epsilon_value
            if do_eval:
                eval_update = {
                    "train/accuracy": train_accuracy,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/best_loss": best_val_loss,
                    "val/best_accuracy": best_val_acc,
                    "time/epoch": epoch_time,
                    "time/total": elapsed,
                    "early_stop/no_improvement_count": es_no_improvement_count,
                    "early_stop/perfect_acc_count": es_perfect_acc_count,
                }
                if is_language_model:
                    eval_update["val/bpc"] = val_loss / math.log(2)
                    eval_update["val/best_bpc"] = best_val_loss / math.log(2)
                epoch_log.update(eval_update)
            if wandb_log:
                try:
                    import wandb
                    wandb.log(epoch_log)
                except Exception as e:
                    print(f"  Warning: wandb logging failed: {e}")
    
    final_metrics = evaluate_model(
        model, task_config, data_key, batch_size, num_eval_batches=final_eval_num_batches,
        return_infos=False, wandb_log=False, epoch=epoch,
    )
    
    total_time = time.time() - start_time
    if is_language_model:
        final_bpc = final_metrics['loss'] / math.log(2)
        print(f"Training complete: final_loss={final_metrics['loss']:.6f}, "
              f"final_bpc={final_bpc:.4f}, total_time={total_time:.1f}s", flush=True)
    else:
        print(f"Training complete: final_loss={final_metrics['loss']:.6f}, "
              f"final_acc={final_metrics['accuracy']:.4f}, total_time={total_time:.1f}s")
    
    # Save final model if strategy is "last"
    if save_best_model and model_save_path and model_save_strategy == "last":
        final_model_path = str(Path(model_save_path).with_suffix("")) + "_best.pkl"
        save_model(
            model,
            final_model_path,
            epoch=epoch,
            task_name=task_name,
            config=config,
            metrics=final_metrics,
            wandb_id=None,
        )
        print(f"  Saved final model to {final_model_path}", flush=True)
    
    if wandb_log:
        try:
            import wandb
            final_log = {
                "epoch": epoch,
                "final/loss": final_metrics['loss'],
                "final/accuracy": final_metrics['accuracy'],
                "time/total_training": total_time,
            }
            if is_language_model:
                final_log["final/bpc"] = final_metrics['loss'] / math.log(2)
            wandb.log(final_log)
            wandb.finish()
        except Exception as e:
            print(f"  Warning: wandb final logging failed: {e}")

    return model, losses, final_metrics