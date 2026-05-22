from typing import Sequence, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .base import BaseCell, EpsilonMixin, cell
from .utils.surrogate import heaviside


@cell
class CMOS_BMRU(BaseCell, EpsilonMixin):
    """
    CMOS BMRU with Schmitt Trigger behavior and additive integration.

    The state update follows a 3-region Schmitt trigger:
      - cands < β₁:          state = 0 + ε · old_state  (force LOW)
      - β₁ ≤ cands ≤ β₂:    state = old_state           (HOLD / hysteresis)
      - cands > β₂:          state = 1 + ε · old_state  (force HIGH)

    β₂ > β₁ is guaranteed by parameterizing as β₁ and δ (hysteresis width),
    where β₂ = β₁ + δ.

    Initialization places most candidates in the HOLD region initially.
    """

    _state_dim: int
    _cand_nn: nnx.Linear
    _post_proj_nn: nnx.Linear
    _initial_state: nnx.Param[Array]
    _alphas: nnx.Param[Array]
    _beta1: nnx.Param[Array]
    _delta: nnx.Param[Array]
    _parallel: bool
    _surr_alpha: Array

    def __init__(
        self,
        input_size: int,
        state_dim: int,
        output_size: int,
        rngs: nnx.Rngs,
        parallel: bool = True,
        surr_alpha: float = 1.0,
        beta1_init_scale: float = 0.3,
        delta_init_scale: float = 0.5,
        epsilon: float = 0.0,
        hold_init_target: float = 0.5,
    ) -> None:
        """
        Initialize CMOS BMRU.

        Args:
        -----
        beta1_init_scale: float
            Controls where β₁ is placed relative to hold_init_target.
            β₁ ≈ hold_init_target * beta1_init_scale.
        delta_init_scale: float
            Scales the hysteresis width δ so β₂ = β₁ + δ brackets hold_init_target.
        hold_init_target: float
            Target candidate value for the center of the HOLD region at init.
        """
        super().__init__(
            input_size=input_size,
            state_dim=state_dim,
            output_size=output_size,
            rngs=rngs,
        )
        self._parallel = parallel
        self._init_epsilon(epsilon)

        # Positive-only candidates
        self._cand_nn = nnx.Linear(input_size, state_dim, rngs=rngs)
        self._cand_nn.kernel.value = jnp.abs(self._cand_nn.kernel.value)
        self._cand_nn.bias.value = jnp.full(state_dim, hold_init_target)

        # Schmitt trigger thresholds: β₁ below target, β₂ = β₁ + δ above target
        beta1_center = hold_init_target * beta1_init_scale
        beta1_noise = jax.random.uniform(rngs(), (1, 1, state_dim), minval=0.8, maxval=1.2)
        beta1 = beta1_center * beta1_noise

        desired_delta = 2.0 * (hold_init_target - beta1_center)
        delta_noise = jax.random.uniform(rngs(), (1, 1, state_dim), minval=0.8, maxval=1.2)
        delta = desired_delta * delta_noise * delta_init_scale

        self._beta1 = nnx.Param(jnp.abs(beta1))
        self._delta = nnx.Param(jnp.abs(delta))

        self._post_proj_nn = nnx.Linear(state_dim, output_size, rngs=rngs)
        self._post_proj_nn.kernel.value = jnp.abs(self._post_proj_nn.kernel.value)
        self._post_proj_nn.bias.value = jnp.abs(self._post_proj_nn.bias.value)

        initial_state = jax.random.uniform(rngs(), shape=(1, state_dim), minval=0.0, maxval=1.0)
        self._initial_state = nnx.Param(initial_state)

        initializer = nnx.initializers.variance_scaling(scale=2.0, mode="fan_out", distribution="uniform")
        self._alphas = nnx.Param(jnp.abs(initializer(key=rngs(), shape=(1, 1, state_dim))))

        self._surr_alpha = jnp.array(surr_alpha)

    @property
    def parallel(self) -> bool:
        return self._parallel

    @staticmethod
    def rec_params() -> Sequence[str]:
        return ["_cand_nn", "_initial_state"]

    def init_state(self, batch_size: int) -> Array:
        initial_state = jnp.repeat(self._initial_state.value, batch_size, axis=0)
        binary_state = heaviside(jnp.abs(initial_state) - 0.5, self._surr_alpha)
        return binary_state * jnp.abs(self._alphas.value[0])

    @staticmethod
    def _parallel_scan(
        hold: Array,
        force_high: Array,
        initial_state: Array,
        epsilon: Array,
    ) -> Array:
        """
        Parallel Schmitt trigger update: s_t = a_t · s_{t-1} + b_t
        where a_t = hold_t + (1 - hold_t) · ε  and  b_t = (1 - hold_t) · force_high_t
        """
        initial_state = jnp.expand_dims(initial_state, axis=1)
        comp_hold = 1 - hold
        leaky_a = hold + comp_hold * epsilon
        leaky_b = comp_hold * force_high

        first = jnp.concat([jnp.full_like(initial_state, 1.0), leaky_a], axis=1)
        second = jnp.concat([initial_state, leaky_b], axis=1)

        def scan_fn(a, b):
            return (a[0] * b[0], b[0] * a[1] + b[1])

        result = jax.lax.associative_scan(scan_fn, (first, second), axis=1)
        return result[1][:, 1:]

    @staticmethod
    def _sequential_scan(
        hold: Array,
        force_high: Array,
        initial_state: Array,
        epsilon: Array,
    ) -> Array:
        def step_fn(carry, inputs):
            h, fh = inputs
            new_state = h * carry + (1 - h) * (fh + epsilon * carry)
            return new_state, new_state

        _, states = jax.lax.scan(
            step_fn, initial_state, (hold.transpose(1, 0, 2), force_high.transpose(1, 0, 2))
        )
        return states.transpose(1, 0, 2)

    def __call__(
        self,
        inputs: Array,
        initial_state: Array,
        training: bool = False,
        return_infos: bool = False,
    ) -> dict[str, Array]:
        x = jnp.maximum(inputs, 0.0)
        cands = jnp.maximum(self._cand_nn(x), 0.0)

        beta1 = jnp.abs(self._beta1.value)
        delta = jnp.abs(self._delta.value)
        beta2 = beta1 + delta

        batch_size, seq_len, _ = cands.shape
        beta1_exp = jnp.broadcast_to(beta1, (batch_size, seq_len, self._state_dim))
        beta2_exp = jnp.broadcast_to(beta2, (batch_size, seq_len, self._state_dim))

        above_beta1 = heaviside(cands - beta1_exp, self._surr_alpha)
        above_beta2 = heaviside(cands - beta2_exp, self._surr_alpha)

        force_high = above_beta2
        hold = above_beta1 * (1 - above_beta2)

        epsilon = self._get_epsilon()
        if self._parallel:
            states = self._parallel_scan(hold, force_high, initial_state, epsilon)
        else:
            states = self._sequential_scan(hold, force_high, initial_state, epsilon)

        states = jnp.maximum(states * jnp.abs(self._alphas.value), 0.0)
        outputs = jnp.maximum(self._post_proj_nn(states), 0.0)

        result = {"outputs": outputs}

        if return_infos:
            result["states"] = states
            result["force_low"] = 1 - above_beta1
            result["hold"] = hold
            result["force_high"] = force_high
            result["beta1"] = beta1_exp
            result["beta2"] = beta2_exp
            result["delta"] = jnp.broadcast_to(delta, (batch_size, seq_len, self._state_dim))
            result["candidates"] = cands
            result["epsilon"] = self._get_epsilon()

        return result

    def sanity_check(self, inputs: Array, initial_state: Array) -> bool:
        """Verify parallel and sequential scans match."""
        x = jnp.maximum(inputs, 0.0)
        cands = jnp.maximum(self._cand_nn(x), 0.0)

        beta1 = jnp.abs(self._beta1.value)
        beta2 = beta1 + jnp.abs(self._delta.value)
        batch_size, seq_len, _ = cands.shape
        beta1_exp = jnp.broadcast_to(beta1, (batch_size, seq_len, self._state_dim))
        beta2_exp = jnp.broadcast_to(beta2, (batch_size, seq_len, self._state_dim))

        above_beta1 = heaviside(cands - beta1_exp, self._surr_alpha)
        above_beta2 = heaviside(cands - beta2_exp, self._surr_alpha)
        force_high = above_beta2
        hold = above_beta1 * (1 - above_beta2)

        epsilon = self._get_epsilon()
        s_par = self._parallel_scan(hold, force_high, initial_state, epsilon)
        s_seq = self._sequential_scan(hold, force_high, initial_state, epsilon)
        return jnp.allclose(s_par, s_seq, rtol=1e-5, atol=1e-6)

    def set_surr_alpha(self, new_surr_alpha: float) -> None:
        self._surr_alpha = jnp.array(new_surr_alpha)

    def scale_surr_alpha(self, scale: float) -> None:
        self._surr_alpha = jnp.array(self._surr_alpha * scale)
