"""
minGRU (Minimal Gated Recurrent Unit) - Simplified parallelizable GRU.

Based on "Were RNNs All We Needed?" (Feng et al., 2024)
https://arxiv.org/abs/2410.01201

Key characteristics:
- Removes hidden state dependencies from gates (enables parallel training)
- No range restriction (no tanh)
- Significantly fewer parameters than traditional GRU
- Fully parallelizable via parallel scan algorithm

Update rule:
-----------
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
z_t = σ(Linear(x_t))
h̃_t = Linear(x_t)
"""

from math import sqrt
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .base import cell, BaseCell


@cell
class minGRU(BaseCell):
    """
    Minimal Gated Recurrent Unit (minGRU).
    """
    
    _linear_z: nnx.Linear  # Update gate projection
    _linear_h: nnx.Linear  # Candidate hidden state projection
    _out_proj: nnx.Linear  # Output projection
    _initial_state: nnx.Param[Array]
    _parallel: bool

    def __init__(
        self,
        input_size: int,
        state_dim: int,
        output_size: int,
        rngs: nnx.Rngs,
        parallel: bool = True,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__(input_size, state_dim, output_size, rngs)

        self._parallel = parallel

        self._linear_z = nnx.Linear(input_size, state_dim, rngs=rngs)
        if gate_bias_init != 0.0:
            self._linear_z.bias.value = jnp.full_like(self._linear_z.bias.value, gate_bias_init)
        self._linear_h = nnx.Linear(input_size, state_dim, rngs=rngs)

        # Output projection scaled down for variance control
        self._out_proj = nnx.Linear(state_dim, output_size, rngs=rngs)
        self._out_proj.kernel.value = self._out_proj.kernel.value / sqrt(state_dim)

        self._initial_state = nnx.Param(self._create_initial_state(rngs))
    
    def _create_initial_state(self, rngs: nnx.Rngs) -> Array:
        """Create initial hidden state with random uniform values in [-1, 1]."""
        return jax.random.uniform(rngs(), (1, self._state_dim), minval=-1.0, maxval=1.0)

    @property
    def parallel(self) -> bool:
        """Whether parallel scan is enabled."""
        return self._parallel
    
    @staticmethod
    def rec_params() -> Sequence[str]:
        return ["_linear_z", "_linear_h", "_initial_state"]

    def init_state(self, batch_size: int) -> Array:
        return jnp.repeat(self._initial_state.value, batch_size, axis=0)
    
    def __call__(
        self,
        inputs: Array,
        initial_state: Array,
        training: bool = False,
        return_infos: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through minGRU cell.
        
        Args:
        -----
        inputs: Array [B, L, input_size]
            Input tensor
        initial_state: Array [B, state_dim]
            Initial state
        training: bool
            Whether in training mode (for interface compatibility)
        return_infos: bool
            Whether to return diagnostic information
            
        Returns:
        --------
        dict containing:
        - outputs: Array [B, L, output_size]
        - states: Hidden states [B, L, state_dim] (if return_infos)
        - gates: Update gates [B, L, state_dim] (if return_infos)
        """
        # Compute states
        states = self._forward_pass(inputs, initial_state)

        # Project to output dimension
        outputs = self._out_proj(states)

        result = {"outputs": outputs, "final_state": states[:, -1, :]}
        
        if return_infos:
            z = jax.nn.sigmoid(self._linear_z(inputs))
            result["states"] = states
            result["gates"] = z
        
        return result
    
    def _forward_pass(
        self,
        inputs: Array,
        initial_state: Array,
    ) -> Array:
        # Compute update gate: z_t = σ(Linear(x_t))
        z = jax.nn.sigmoid(self._linear_z(inputs))
        
        # Compute candidate: h̃_t = Linear(x_t)
        h_tilde = self._linear_h(inputs)
        
        if self._parallel:
            return self._parallel_scan(z, h_tilde, initial_state)
        else:
            return self._sequential_scan(z, h_tilde, initial_state)
    
    @staticmethod
    def _parallel_scan(
        z: Array,
        h_tilde: Array,
        h_0: Array,
    ) -> Array:
        """
        Parallel scan for h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        as a linear recurrence: h_t = a_t ⊙ h_{t-1} + b_t
        where a_t = (1 - z_t)  and  b_t = z_t ⊙ h̃_t
        """
        # Coefficients: a_t = (1 - z_t)
        coeffs = 1.0 - z
        
        # Values: b_t = z_t ⊙ h̃_t
        values = z * h_tilde
        
        # Prepare for associative scan
        h_0_expanded = jnp.expand_dims(h_0, axis=1)
        initial_coeffs = jnp.ones_like(h_0_expanded)
        
        # Concatenate initial values
        coeffs_with_init = jnp.concat([initial_coeffs, coeffs], axis=1)
        values_with_init = jnp.concat([h_0_expanded, values], axis=1)
        
        # Associative scan
        def scan_fn(a: tuple[Array, Array], b: tuple[Array, Array]) -> tuple[Array, Array]:
            """(a, b) ⊕ (c, d) = (a⊙c, c⊙b + d)"""
            return (a[0] * b[0], b[0] * a[1] + b[1])
        
        result = jax.lax.associative_scan(
            scan_fn,
            (coeffs_with_init, values_with_init),
            axis=1
        )
        
        # Remove initial state from result
        return result[1][:, 1:]
    
    @staticmethod
    def _sequential_scan(
        z: Array,
        h_tilde: Array,
        h_0: Array,
    ) -> Array:
        """Sequential scan: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t"""
        def step_fn(h_prev: Array, inputs: tuple[Array, Array]) -> tuple[Array, Array]:
            z_t, h_tilde_t = inputs
            h_t = (1.0 - z_t) * h_prev + z_t * h_tilde_t
            return h_t, h_t
        
        # Transpose to time-major for scan
        z_time_major = z.transpose(1, 0, 2)  # [L, B, D]
        h_tilde_time_major = h_tilde.transpose(1, 0, 2)  # [L, B, D]
        
        _, states = jax.lax.scan(
            step_fn,
            h_0,
            (z_time_major, h_tilde_time_major)
        )
        
        # Transpose back to batch-major
        return states.transpose(1, 0, 2)  # [B, L, D]
    
    def sanity_check(
        self,
        inputs: Array,
        initial_state: Array,
    ) -> bool:
        """Verify parallel and sequential implementations match."""
        # Compute gates and candidates
        z = jax.nn.sigmoid(self._linear_z(inputs))
        h_tilde = self._linear_h(inputs)
        
        # Compare implementations
        states_parallel = self._parallel_scan(z, h_tilde, initial_state)
        states_sequential = self._sequential_scan(z, h_tilde, initial_state)
        
        return jnp.allclose(states_parallel, states_sequential, rtol=1e-4, atol=1e-5)