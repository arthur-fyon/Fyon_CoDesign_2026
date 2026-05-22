#!/usr/bin/env python3
"""
Enhanced config variant generator with full optional parameter support.

Usage:
    python seeder.py <input_json> <output_folder> <task> <seed> \\
                     [num_recs] [model_dim] [norm] [state_dim] [epsilon] \\
                     [activation] [num_epochs] [learning_rate] [weight_decay] [mlp_dropout] [bidirectional]

Required Arguments:
    input_json: Path to base config JSON
    output_folder: Directory to save variant config
    task: Task identifier for filename
    seed: Random seed

Optional Arguments (use "none" or omit to keep base config default):
    num_recs: Number of recurrent layers
    model_dim: Model dimension
    norm: Normalization type ("batch", "layer", "rms", "none")
    state_dim: RNN state dimension
    epsilon: Leaky update coefficient - float or "learnable"
    activation: Activation function for MLP gates ("relu", "gelu", "tanh")
    num_epochs: Training epochs
    learning_rate: Learning rate
    weight_decay: Weight decay coefficient
    mlp_dropout: MLP dropout rate
    bidirectional: Bidirectional RNN ("true" or "false")
    batch_size: Training batch size
"""

import json
import sys
from pathlib import Path
from typing import Optional, Union


def parse_optional_arg(arg: Optional[str], arg_type: type, allow_learnable: bool = False, allow_none: bool = True) -> Optional[Union[str, float, int, bool, bool]]:
    """
    Parse optional argument, returning None if not provided or "none".
    
    Args:
        arg: Command line argument
        arg_type: Target type (int, float, str, bool)
        allow_learnable: If True, allow "learnable" as a string value
        allow_none: If True, allow "none" as a string value different to returning None
    """

    if arg is None or (not allow_none and arg.lower() == "none"):
        return None
    elif arg == "none" and allow_none:
        return "none"
    
    if allow_learnable and arg.lower() == "learnable":
        return "learnable"

    if allow_learnable and arg.lower() == "decaying":
        return "decaying"
    
    # Handle boolean type
    if arg_type == bool:
        if arg.lower() in ("true", "1", "yes"):
            return True
        elif arg.lower() in ("false", "0", "no"):
            return False
        else:
            return None
    
    try:
        return arg_type(arg)
    except (ValueError, TypeError):
        return None


def create_variant(
    input_file: str,
    output_folder: str,
    task: str,
    seed: int,
    num_recs: Optional[int] = None,
    model_dim: Optional[int] = None,
    norm: Optional[str] = None,
    state_dim: Optional[int] = None,
    epsilon: Optional[Union[float, str]] = None,
    activation: Optional[str] = None,
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    mlp_dropout: Optional[float] = None,
    bidirectional: Optional[bool] = None,
    batch_size: Optional[int] = None,
) -> None:
    """
    Create a single config variant with specified hyperparameters.
    Only modifies parameters that are explicitly provided.
    
    Args:
        input_file: Path to base config JSON
        output_folder: Directory to save variant config
        task: Task identifier for filename
        seed: Random seed (always applied)
        num_recs: Number of recurrent layers (None = keep base config)
        model_dim: Model dimension (None = keep base config)
        norm: Normalization type (None = keep base config)
        state_dim: RNN state dimension (None = keep base config or sync with model_dim)
        epsilon: Leaky update coefficient - float or "learnable" (None = keep base config)
        activation: Activation function for MLP gates (None = keep base config)
        num_epochs: Training epochs (None = keep base config)
        learning_rate: Learning rate (None = keep base config)
        weight_decay: Weight decay coefficient (None = keep base config)
        mlp_dropout: MLP dropout rate (None = keep base config)
        bidirectional: Bidirectional RNN (None = keep base config)
    """
    input_path = Path(input_file)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    with open(input_path, 'r') as f:
        config = json.load(f)
    
    # Track what we modified for filename
    modifications = {}
    
    # Always update seed
    config['seed'] = seed
    modifications['seed'] = seed
    
    # Update model settings (only if provided)
    if num_recs is not None:
        config['model']['num_recs'] = num_recs
        modifications['num_recs'] = num_recs
    
    if model_dim is not None:
        config['model']['model_dim'] = model_dim
        config['model']['mlp_hidden_dim'] = model_dim  # Sync with model_dim
        modifications['model_dim'] = model_dim
    

    if norm is not None:
        config['model']['norm'] = norm
        modifications['norm'] = norm

    # Update training settings (only if provided)
    if num_epochs is not None:
        config['num_epochs'] = num_epochs
        modifications['num_epochs'] = num_epochs
    
    if learning_rate is not None:
        config['learning_rate'] = learning_rate
        modifications['learning_rate'] = learning_rate
    
    if weight_decay is not None:
        config['optimizer']['weight_decay'] = weight_decay
        modifications['weight_decay'] = weight_decay
    
    if mlp_dropout is not None:
        config['model']['mlp_dropout'] = mlp_dropout
        modifications['mlp_dropout'] = mlp_dropout
    
    if activation is not None:
        config['model']['mlp_activation'] = activation
        modifications['activation'] = activation

    # Handle state_dim logic
    actual_state_dim = None
    if state_dim is not None:
        # Explicitly provided state_dim
        actual_state_dim = state_dim
        modifications['state_dim'] = state_dim
    elif model_dim is not None:
        # model_dim was changed, sync state_dim with it
        actual_state_dim = model_dim
        modifications['state_dim'] = model_dim
    
    # Update cell-specific configs
    for cell_name in config['cell_configs']:
        cell_config = config['cell_configs'][cell_name]
        
        # Update state_dim (only if we determined a value)
        if actual_state_dim is not None:
            cell_config['state_dim'] = actual_state_dim
            if 'state_dim' not in config['model']:
                config['model']['state_dim'] = actual_state_dim
        
        # Update epsilon (only if provided)
        if epsilon is not None:
            epsilon_decay = False
            # Save original value for filename tracking
            epsilon_for_filename = epsilon
            
            # if epsilon is not "learnable", ensure it's a float, written as a float (so 0 is 0.0, 1 is 1.0, etc.)
            if epsilon != "learnable" and epsilon != "decaying":
                cell_config['epsilon'] = float(epsilon)
            elif epsilon == "decaying":
                cell_config['epsilon'] = 1.0
                epsilon = 1.0
                epsilon_decay = True
                config['epsilon_decay'] = True
            else:
                cell_config['epsilon'] = epsilon
            modifications['epsilon'] = epsilon_for_filename
            modifications['epsilon_decay'] = epsilon_decay
        
        # Update bidirectional (only if provided)
        if bidirectional is not None:
            cell_config['bidirectional'] = bidirectional
            modifications['bidirectional'] = bidirectional

    if batch_size is not None:
        config['batch_size'] = batch_size
        modifications['batch_size'] = batch_size

    # Build descriptive filename
    filename_parts = [input_path.stem, task, f"seed{seed}"]
    
    # Add modifications to filename (in consistent order)
    if 'num_recs' in modifications:
        filename_parts.append(f"recs{modifications['num_recs']}")
    
    if 'model_dim' in modifications:
        filename_parts.append(f"dim{modifications['model_dim']}")
    
    if 'norm' in modifications:
        filename_parts.append(f"norm{modifications['norm']}")
    
    if 'state_dim' in modifications and modifications.get('state_dim') != modifications.get('model_dim'):
        filename_parts.append(f"state{modifications['state_dim']}")
    
    if 'epsilon' in modifications:
        if modifications['epsilon'] == "learnable":
            filename_parts.append("eps_learn")
        elif modifications['epsilon'] == "decaying":
            filename_parts.append("eps_decay")
        else:
            filename_parts.append(f"eps{modifications['epsilon']}")
    
    if 'activation' in modifications:
        filename_parts.append(f"act{modifications['activation']}")
    
    if 'bidirectional' in modifications:
        if modifications['bidirectional']:
            filename_parts.append("bidir")
    
    if 'num_epochs' in modifications:
        filename_parts.append(f"epochs{modifications['num_epochs']}")
    
    if 'learning_rate' in modifications:
        filename_parts.append(f"lr{modifications['learning_rate']}")
    
    if 'weight_decay' in modifications:
        filename_parts.append(f"wd{modifications['weight_decay']}")
    
    if 'mlp_dropout' in modifications:
        filename_parts.append(f"drop{modifications['mlp_dropout']}")

    if 'batch_size' in modifications:
        filename_parts.append(f"bs{modifications['batch_size']}")

    output_filename = "_".join(filename_parts) + ".json"
    output_path = output_dir / output_filename
    
    # Save config
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        print("\nExamples:")
        print("  # Minimal usage (only change seed):")
        print("  python seeder.py config.json variants/ task1 42")
        print()
        print("  # Override num_recs and model_dim:")
        print("  python seeder.py config.json variants/ task1 42 2 64")
        print()
        print("  # Add norm:")
        print("  python seeder.py config.json variants/ task1 42 2 64 batch")
        print()
        print("  # Set learnable epsilon (keep other defaults):")
        print("  python seeder.py config.json variants/ task1 42 none none none none learnable")
        print()
        print("  # Enable bidirectional:")
        print("  python seeder.py config.json variants/ task1 42 none none none none none none none none none none true")
        print()
        print("  # Full configuration:")
        print("  python seeder.py config.json variants/ task1 42 2 64 batch 128 learnable relu 100000 0.001 0.0001 0.25 true")
        print()
        print("  # Skip intermediate args with 'none':")
        print("  python seeder.py config.json variants/ task1 42 none none none none 0.0 none none 0.0005")
        print()
        print("  # Change only epsilon and learning rate:")
        print("  python seeder.py config.json variants/ task1 42 none none none none learnable none none 0.0005")
        sys.exit(1)
    
    # Required arguments
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    task = sys.argv[3]
    seed = int(sys.argv[4])
    
    # Optional arguments with parsing
    # allow_none=False: "none" means "keep base config default" (don't override)
    # allow_none=True only for 'norm', where "none" is a valid value (no normalization)
    num_recs = parse_optional_arg(sys.argv[5] if len(sys.argv) > 5 else None, int, allow_none=False)
    model_dim = parse_optional_arg(sys.argv[6] if len(sys.argv) > 6 else None, int, allow_none=False)
    norm = parse_optional_arg(sys.argv[7] if len(sys.argv) > 7 else None, str, allow_none=True)
    state_dim = parse_optional_arg(sys.argv[8] if len(sys.argv) > 8 else None, int, allow_none=False)
    epsilon = parse_optional_arg(sys.argv[9] if len(sys.argv) > 9 else None, float, allow_learnable=True, allow_none=False)
    activation = parse_optional_arg(sys.argv[10] if len(sys.argv) > 10 else None, str, allow_none=False)
    num_epochs = parse_optional_arg(sys.argv[11] if len(sys.argv) > 11 else None, int, allow_none=False)
    learning_rate = parse_optional_arg(sys.argv[12] if len(sys.argv) > 12 else None, float, allow_none=False)
    weight_decay = parse_optional_arg(sys.argv[13] if len(sys.argv) > 13 else None, float, allow_none=False)
    mlp_dropout = parse_optional_arg(sys.argv[14] if len(sys.argv) > 14 else None, float, allow_none=False)
    bidirectional = parse_optional_arg(sys.argv[15] if len(sys.argv) > 15 else None, bool, allow_none=False)
    batch_size = parse_optional_arg(sys.argv[16] if len(sys.argv) > 16 else None, int, allow_none=False)

    create_variant(
        input_file=input_file,
        output_folder=output_folder,
        task=task,
        seed=seed,
        num_recs=num_recs,
        model_dim=model_dim,
        norm=norm,
        state_dim=state_dim,
        epsilon=epsilon,
        activation=activation,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mlp_dropout=mlp_dropout,
        bidirectional=bidirectional,
        batch_size=batch_size,
    )