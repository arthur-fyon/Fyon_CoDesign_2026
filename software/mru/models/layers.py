"""Helper layers for RNN models."""

from typing import Callable, Optional

import jax.numpy as jnp
from flax import nnx
from jax import Array


class Identity(nnx.Module):
    """Identity layer - returns input unchanged."""
    
    def __call__(self, x: Array) -> Array:
        return x


class ScaleProj(nnx.Module):
    
    _scale: nnx.Param[Array]
    _dropout: Optional[nnx.Dropout]
    
    def __init__(self, dim: int, dropout: float = 0.0, scaling_factor: float = 1.0, rngs: nnx.Rngs = None):
        super().__init__()
        self._scale = nnx.Param(scaling_factor * jnp.ones((dim,)))
        self._dropout = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0.0 else None
    
    def __call__(self, x: Array, y: Array, training: bool = False) -> Array:
        return self._scale * x + y


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization."""
    
    _scale: nnx.Param[Array]
    _eps: float
    
    def __init__(self, num_features: int, eps: float = 1e-6, rngs: nnx.Rngs = None):
        super().__init__()
        self._eps = eps
        self._scale = nnx.Param(jnp.ones((num_features,)))
    
    def __call__(self, x: Array) -> Array:
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self._eps)
        return self._scale * (x / rms)

class MLP(nnx.Module):
    """
    Transformer/SSM-style MLP: Linear -> activation/GLU -> Linear.
    Supports relu, gelu, glu activations.
    """

    _proj_in: nnx.Linear
    _proj_out: nnx.Linear
    _activation: Callable[[Array], Array]
    _dropout: Optional[nnx.Dropout]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        ff_expansion: int = 4,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()

        d_ff = ff_expansion * hidden_dim

        # Select activation
        if activation == "relu":
            self._activation = nnx.relu
            mul = 1

        elif activation == "gelu":
            self._activation = nnx.gelu
            mul = 1

        elif activation == "glu":
            # Use the built-in nnx.glu (expects 2*d_ff input)
            self._activation = nnx.glu
            mul = 2

        else:
            raise ValueError(f"Unknown activation: '{activation}'")

        # Projections
        self._proj_in = nnx.Linear(input_dim, d_ff * mul, rngs=rngs)
        self._proj_out = nnx.Linear(d_ff, output_dim, rngs=rngs)

        self._dropout = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else None
        self._d_ff = d_ff

    def __call__(self, x: Array, training: bool = False) -> Array:
        h = self._proj_in(x)

        h = self._activation(h)

        if self._dropout is not None:
            h = self._dropout(h, deterministic=not training)

        return self._proj_out(h)



def near_identity_linear(model_dim: int, rng: nnx.Rngs, eps: float = 1e-1) -> nnx.Linear:
    """Create a linear layer near identity (for stable initialization)."""
    import jax
    
    key = rng()
    W = jnp.eye(model_dim) + eps * jax.random.normal(key, (model_dim, model_dim))
    
    linear = nnx.Linear(model_dim, model_dim, use_bias=False, rngs=rng)
    linear.kernel.value = W
    
    return linear