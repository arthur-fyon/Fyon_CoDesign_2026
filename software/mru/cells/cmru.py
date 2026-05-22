from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .base import BaseCell, EpsilonMixin, cell
from .utils.memory import to_cpu
from .utils.surrogate import heaviside, sign


@cell
class CMRU(BaseCell, EpsilonMixin):
    """
    CMRU with global alpha scaling.

    Note: cmru corresponds to epsilon=1.
    The name bmru referred to the epsilon=0 (non-leaky) configuration.

    Update rule:
    ------------
    c_t = cand_nn(x_t)                    # Candidate
    beta_t = beta_nn(x_t)                 # Threshold
    z_t = H(|c_t| - |beta_t|)            # Gate
    s_t = z_t * (sign(c_t) + eps*s_{t-1}) + (1-z_t) * s_{t-1}  # Leaky update
    o_t = out_proj(s_t * alpha)           # Scaled output projection

    Alpha is global (not input-dependent) and applied after the scan.

    Mathematical formulation:
    -------------------------
    s_t = a_t * s_{t-1} + b_t
    where a_t = (1-z_t) + z_t * eps  and  b_t = z_t * sign(c_t)
    """

    _state_dim: int
    _cand_nn: nnx.Linear
    _beta_nn: nnx.Linear
    _alphas: nnx.Param[Array]
    _initial_state: nnx.Param[Array]
    _parallel: bool
    _surr_alpha: Array
    _output_size: int
    _out_proj: nnx.Linear

    def __init__(
        self,
        input_size: int,
        state_dim: int,
        output_size: int,
        rngs: nnx.Rngs,
        parallel: bool = True,
        surr_alpha: float = 1.0,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(
            input_size=input_size,
            state_dim=state_dim,
            output_size=output_size,
            rngs=rngs,
        )

        self._parallel = parallel
        self._init_epsilon(epsilon)

        self._cand_nn = nnx.Linear(input_size, state_dim, rngs=rngs)

        # Positive threshold bias so candidates start near the decision boundary
        self._beta_nn = nnx.Linear(input_size, state_dim, rngs=rngs)
        self._beta_nn.bias.value = jnp.full(state_dim, 1.0)

        self._alphas = nnx.Param(jnp.ones((1, 1, state_dim)))
        self._out_proj = nnx.Linear(state_dim, output_size, rngs=rngs)
        self._initial_state = nnx.Param(self._create_initial_state(rngs))
        self._surr_alpha = jnp.array(surr_alpha)

    @property
    def parallel(self) -> bool:
        return self._parallel

    @staticmethod
    def rec_params() -> Sequence[str]:
        return ["_cand_nn", "_initial_state"]

    def _create_initial_state(self, rngs: nnx.Rngs) -> Array:
        """Random binary {-1, +1} initial state."""
        s = jax.random.randint(rngs(), shape=(1, self._state_dim), minval=0, maxval=2)
        return s * 2.0 - 1.0

    def init_state(self, batch_size: int) -> Array:
        return jnp.repeat(self._initial_state.value, batch_size, axis=0)

    def _compute_gate_values(self, inputs: Array) -> Tuple[Array, Array, Array]:
        cands = self._cand_nn(inputs)
        betas = self._beta_nn(inputs)
        zs = heaviside(jnp.abs(cands) - jnp.abs(betas), self._surr_alpha)
        update_values = sign(cands, self._surr_alpha)
        return cands, zs, update_values

    def _apply_update_rule(self, zs: Array, update_values: Array, initial_state: Array) -> Array:
        epsilon = self._get_epsilon()
        if self._parallel:
            return self._parallel_scan(zs, update_values, initial_state, epsilon)
        return self._sequential_scan(zs, update_values, initial_state, epsilon)

    @staticmethod
    def _parallel_scan(
        zs: Array,
        update_values: Array,
        initial_state: Array,
        epsilon: Array,
    ) -> Array:
        """
        Parallel scan for the leaky update: s_t = a_t * s_{t-1} + b_t
        where a_t = (1 - z_t) + z_t * eps  and  b_t = z_t * v_t
        """
        leaky_a = (1 - zs) + zs * epsilon
        initial_state = jnp.expand_dims(initial_state, axis=1)

        first = jnp.concat([jnp.full_like(initial_state, 1.0), leaky_a], axis=1)
        second = jnp.concat([initial_state, zs * update_values], axis=1)

        def scan_fn(a, b):
            return (a[0] * b[0], b[0] * a[1] + b[1])

        result = jax.lax.associative_scan(scan_fn, (first, second), axis=1)
        return result[1][:, 1:]

    @staticmethod
    def _sequential_scan(
        zs: Array,
        update_values: Array,
        initial_state: Array,
        epsilon: Array,
    ) -> Array:
        def step_fn(carry, inputs):
            z, v = inputs
            new_state = z * (v + epsilon * carry) + (1.0 - z) * carry
            return new_state, new_state

        _, states = jax.lax.scan(
            step_fn, initial_state, (zs.transpose(1, 0, 2), update_values.transpose(1, 0, 2))
        )
        return states.transpose(1, 0, 2)

    def __call__(
        self,
        inputs: Array,
        initial_state: Array,
        training: bool = False,
        return_infos: bool = False,
    ) -> dict[str, Array]:
        cands, zs, update_values = self._compute_gate_values(inputs)
        states_raw = self._apply_update_rule(zs, update_values, initial_state)

        # Global alpha applied after scan
        states = states_raw * self._alphas.value
        outputs = self._out_proj(states)

        # final_state carries the raw (pre-alpha) state for stateful generation
        result = {"outputs": outputs, "z_mean": jnp.mean(zs), "final_state": states_raw[:, -1, :]}

        if return_infos:
            betas = self._beta_nn(inputs)
            result["states"] = to_cpu(states_raw * self._alphas.value)
            result["updates"] = to_cpu(jnp.abs(cands) > jnp.abs(betas))
            result["surr_alpha"] = to_cpu(self._surr_alpha)
            result["epsilon"] = to_cpu(self._get_epsilon())

        return result

    def sanity_check(self, inputs: Array, initial_state: Array) -> bool:
        """Verify parallel and sequential scans match."""
        _, zs, update_values = self._compute_gate_values(inputs)
        epsilon = self._get_epsilon()
        s_par = self._parallel_scan(zs, update_values, initial_state, epsilon)
        s_seq = self._sequential_scan(zs, update_values, initial_state, epsilon)
        return jnp.allclose(s_par, s_seq, rtol=1e-5, atol=1e-6)

    def set_surr_alpha(self, new_surr_alpha: float) -> None:
        self._surr_alpha = jnp.array(new_surr_alpha)

    def scale_surr_alpha(self, scale: float) -> None:
        self._surr_alpha = jnp.array(self._surr_alpha * scale)
