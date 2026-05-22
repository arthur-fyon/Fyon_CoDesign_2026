import jax.numpy as jnp
from flax import nnx
from jax import Array


class PositionalEncoder(nnx.Module):
    """
    Positional encoder module.
    """

    _num_dims: int
    _n: int

    def __init__(
        self,
        num_dims: int = 16,
        n: int = 10000,
    ) -> None:
        super().__init__()

        assert num_dims % 2 == 0, "num_dims must be even"

        self._num_dims = num_dims
        self._n = n

    @property
    def num_dims(self) -> int:
        return self._num_dims

    def __call__(self, inputs: Array) -> Array:

        batch_size, seq_length, *_ = inputs.shape

        time_positions = jnp.reshape(jnp.arange(seq_length), (1, -1, 1))
        time_positions = jnp.tile(time_positions, (batch_size, 1, self.num_dims // 2))

        encoding_positions = jnp.reshape(jnp.arange(self.num_dims // 2), (1, 1, -1))
        encoding_positions = jnp.tile(encoding_positions, (batch_size, seq_length, 1))

        sines = jnp.sin(
            time_positions / (self._n ** (2.0 * encoding_positions / self.num_dims))
        )

        cosines = jnp.cos(
            time_positions / (self._n ** (2.0 * encoding_positions / self.num_dims))
        )

        encodings = jnp.concat([sines, cosines], axis=-1)

        return encodings
