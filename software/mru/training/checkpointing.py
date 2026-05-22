"""Model checkpointing utilities."""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

from flax import nnx
from mru.models import RNN


def save_model(
    model: RNN,
    path: str,
    epoch: int = None,
    task_name: str = None,
    config: Dict = None,
    metrics: Dict = None,
    wandb_id: str = None,
):
    """Save model and metadata to file."""
    
    # Get model state
    state = nnx.state(model)
    
    # Build checkpoint
    checkpoint = {
        "state": state,
        "epoch": epoch,
        "task_name": task_name,
        "config": config,
        "metrics": metrics,
        "wandb_id": wandb_id,
    }
    
    # Save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    
    print(f"Model saved to {path}")


def load_model(path: str) -> Tuple[RNN, Dict[str, Any]]:
    """
    Load model and metadata from file.
    
    Returns:
        model: Loaded RNN model
        checkpoint: Dict with epoch, task_name, config, metrics
    """
    
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded from {path}", flush=True)

    # Extract config
    config = checkpoint.get("config", {})

    # Recreate model
    model_config = config.get("model", {})

    # Infer input/output sizes from the saved state (avoids needing to store
    # them separately; _pre_proj maps input_size → model_dim and
    # _output_proj maps model_dim → output_size).
    try:
        state = checkpoint["state"]
        input_size  = state._pre_proj.kernel.value.shape[0]
        output_size = state._output_proj.kernel.value.shape[1]
    except Exception:
        # Fall back to explicit checkpoint fields for old checkpoints
        input_size  = checkpoint.get("input_size", 65)
        output_size = checkpoint.get("output_size", 65)

    from flax import nnx
    rngs = nnx.Rngs(config.get("seed", 42))

    model = RNN(
        input_size=input_size,
        output_size=output_size,
        rngs=rngs,
        model_dim=model_config.get("model_dim", 64),
        cells=model_config.get("cells", ["sbc"]),
        cell_configs=config.get("cell_configs", {}),
        num_recs=model_config.get("num_recs", 2),
        positional_encodings_dims=model_config.get("positional_encodings_dims", 0),
        skip=model_config.get("skip", True),
        aggregate=model_config.get("aggregate", "sum"),
        norm=model_config.get("norm", "batch"),
        post_rec_norm=model_config.get("post_rec_norm", "partial"),
        mlp_hidden_dim=model_config.get("mlp_hidden_dim"),
        mlp_activation=model_config.get("mlp_activation", "gelu"),
        mlp_dropout=model_config.get("mlp_dropout", 0.1),
        mlp_ff_expansion=model_config.get("mlp_ff_expansion", 4),
        rec_dropout=model_config.get("rec_dropout", 0.0),
        use_gating=model_config.get("use_gating", True),
        conv_kernel_size=model_config.get("conv_kernel_size", 0),
    )
    
    # Load state
    nnx.update(model, checkpoint["state"])
    
    print(f"Model loaded from {path}", flush=True)
    
    return model, checkpoint