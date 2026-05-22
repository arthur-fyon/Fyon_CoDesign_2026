"""Positional encoding modules."""

import jax.numpy as jnp
from flax import nnx
from jax import Array


class PositionalEncoder(nnx.Module):
    """Sinusoidal positional encoding."""
    
    _num_dims: int
    _n: int
    _max_seq_len: int
    
    def __init__(
        self,
        num_dims: int = 64,
        n: int = 10000,
        max_seq_len: int = 2048,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        assert num_dims % 2 == 0, "num_dims must be even"
        self._num_dims = num_dims
        self._n = n
        self._max_seq_len = max_seq_len
    
    @property
    def num_dims(self) -> int:
        return self._num_dims
    
    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len
    
    def __call__(self, inputs: Array) -> Array:
        batch_size, seq_len, *_ = inputs.shape
        
        # Sinusoidal encoding
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self._num_dims, 2) * (-jnp.log(self._n) / self._num_dims)
        )
        
        sinusoid = jnp.zeros((seq_len, self._num_dims))
        sinusoid = sinusoid.at[:, 0::2].set(jnp.sin(position * div_term))
        sinusoid = sinusoid.at[:, 1::2].set(jnp.cos(position * div_term))
        
        # Expand batch dimension
        return jnp.tile(sinusoid[None, :, :], (batch_size, 1, 1))

class LearnablePositionalEncoder(nnx.Module):
    """Sinusoidal base + learnable refinement with modulo indexing."""
    
    _num_dims: int
    _n: int
    _max_seq_len: int
    learnable_offset: nnx.Param[Array]
    
    def __init__(
        self,
        num_dims: int = 64,
        n: int = 10000,
        max_seq_len: int = 2048,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        assert num_dims % 2 == 0, "num_dims must be even"
        
        self._num_dims = num_dims
        self._n = n
        self._max_seq_len = max_seq_len
        
        # Learnable offset per position
        self.learnable_offset = nnx.Param(jnp.zeros((max_seq_len, num_dims)))
    
    @property
    def num_dims(self) -> int:
        return self._num_dims
    
    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len
    
    def __call__(self, inputs: Array) -> Array:
        batch_size, seq_len, *_ = inputs.shape
        
        # Sinusoidal base
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self._num_dims, 2) * (-jnp.log(self._n) / self._num_dims)
        )
        
        sinusoid = jnp.zeros((seq_len, self._num_dims))
        sinusoid = sinusoid.at[:, 0::2].set(jnp.sin(position * div_term))
        sinusoid = sinusoid.at[:, 1::2].set(jnp.cos(position * div_term))
        
        # Learnable refinement (modulo indexing for arbitrary length)
        idx = jnp.arange(seq_len) % self._max_seq_len
        learnable = self.learnable_offset.value[idx, :]
        
        # Combine
        encoding = sinusoid + learnable
        
        # Expand batch dimension
        return jnp.tile(encoding[None, :, :], (batch_size, 1, 1))