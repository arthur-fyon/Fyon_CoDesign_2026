from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .base import BaseCell, cell


def associative_operator(
    el_i: tuple[Array, Array],
    el_j: tuple[Array, Array],
):
    a_i, bu_i = el_i
    a_j, bu_j = el_j

    return a_i * a_j, a_j * bu_i + bu_j


@cell
class LRU(BaseCell):
    """
    Linear Recurrent Unit (LRU) - Diagonal state-space model.
    
    Key characteristics:
    - Complex-valued diagonal state-space model
    - Efficient parallel scan implementation
    - No gating mechanism (unlike SBC variants)
    
    Update rule:
    -----------
    x_t = diag · x_{t-1} + B · u_t        # State update (complex)
    y_t = Re(C · x_t) + D · u_t           # Output (real)
    
    Note: Normalization, dropout, and activation are handled at the model level,
    not within the cell (consistent with SBC variants).
    """
    
    _r_min: float
    _r_max: float
    _max_phase: float
    _post_proj: nnx.Linear

    _log_nu: nnx.Param
    _log_theta: nnx.Param
    _log_gamma: nnx.Param
    _B: nnx.Param
    _C: nnx.Param
    _D: nnx.Param

    def __init__(
        self,
        input_size: int,
        state_dim: int,
        output_size: int,
        rngs: nnx.Rngs,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LRU cell.
        
        Args:
        -----
        input_size: int
            Size of inputs (model_dim)
        state_dim: int
            Size of recurrent state (complex-valued)
        output_size: int
            Size of output (projection dimension)
        rngs: nnx.Rngs
            Random number generator
        r_min: float
            Minimum eigenvalue radius (default: 0.0)
        r_max: float
            Maximum eigenvalue radius (default: 1.0)
        max_phase: float
            Maximum phase angle (default: 6.28 ~ 2pi)
        **kwargs: Additional arguments (for compatibility, ignored)
            
        Note: Unlike the original LRU implementation, normalization, dropout,
        and activation are handled at the model level in the RNN architecture.
        """
        super().__init__(input_size, state_dim, output_size, rngs)

        self._r_min = r_min
        self._r_max = r_max
        self._max_phase = max_phase

        # Initialize state-space matrices
        u1 = jax.random.uniform(rngs(), (state_dim,))
        u2 = jax.random.uniform(rngs(), (state_dim,))

        log_nu = jnp.log(-0.5 * jnp.log(r_min**2 + u1 * (r_max**2 - r_min**2)))
        log_theta = jnp.log(u2 * max_phase)

        diag = jnp.exp(-jnp.exp(log_nu) + 1j * jnp.exp(log_theta))
        log_gamma = jnp.log(jnp.sqrt(1.0 - jnp.abs(diag) ** 2))

        initializer = nnx.initializers.xavier_uniform()

        B_re = initializer(rngs(), (input_size, state_dim))
        B_im = initializer(rngs(), (input_size, state_dim))
        C_re = initializer(rngs(), (state_dim, state_dim))
        C_im = initializer(rngs(), (state_dim, state_dim))
        D = initializer(rngs(), (input_size, state_dim))

        self._log_nu = nnx.Param(log_nu)
        self._log_theta = nnx.Param(log_theta)
        self._log_gamma = nnx.Param(log_gamma)
        self._B = nnx.Param(jax.lax.complex(B_re, B_im))
        self._C = nnx.Param(jax.lax.complex(C_re, C_im))
        self._D = nnx.Param(D)

        self._post_proj = nnx.Linear(state_dim, output_size, rngs=rngs)

    @staticmethod
    def rec_params() -> Sequence[str]:
        return [
            "_log_nu",
            "_log_theta",
            "_log_gamma",
            "_B",
        ]

    def init_state(self, batch_size: int) -> Array:
        return jnp.zeros((batch_size, self._state_dim), dtype=jnp.complex64)

    def __call__(
        self,
        inputs: Array,
        initial_state: Array,
        training: bool = False,
        return_infos: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through LRU cell.
        
        Args:
        -----
        inputs: Array [B, L, input_size]
            Input tensor
        initial_state: Array [B, state_dim]
            Initial state (complex-valued)
        training: bool
            Whether in training mode (for interface compatibility, not used in LRU)
        return_infos: bool
            Whether to return diagnostic information
            
        Returns:
        --------
        dict containing:
        - outputs: Array [B, L, output_size]
        - states: Complex states [B, L, state_dim] (if return_infos)
        """

        batch_size, seq_length, *_ = inputs.shape

        # Compute state-space parameters
        diag = jnp.exp(
            -jnp.exp(self._log_nu.value) + 1j * jnp.exp(self._log_theta.value)
        )
        B_norm = self._B.value * jnp.expand_dims(jnp.exp(self._log_gamma.value), axis=0)

        repeated_diag = jnp.repeat(diag[None, ...], seq_length, axis=0)
        repeated_diag = jnp.repeat(repeated_diag[None, ...], batch_size, axis=0)

        Bus = inputs @ B_norm

        # Add initial state in the elements
        init_diag = jnp.ones((batch_size, 1, self._state_dim), dtype=jnp.complex64)
        init_bu = jnp.expand_dims(initial_state, axis=1)
        repeated_diag = jnp.concat([init_diag, repeated_diag], axis=1)
        Bus = jnp.concat([init_bu, Bus], axis=1)

        # Do the scan
        els = (repeated_diag, Bus)
        _, states = jax.lax.associative_scan(associative_operator, els, axis=1)

        # Remove initial state
        states = states[:, 1:]

        # Compute outputs of state-space
        x = (states @ self._C.value).real + inputs @ self._D.value

        # Output projection (normalization, activation, dropout handled at model level)
        outputs = self._post_proj(x)

        result = {"outputs": outputs, "final_state": states[:, -1, :]}

        if return_infos:
            result["states"] = states

        return result