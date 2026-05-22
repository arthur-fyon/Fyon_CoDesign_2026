from math import sqrt
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .base import BaseCell, cell
from .utils.surrogate import heaviside, sign

# =============================================================================
# PARALLEL SCAN OPERATIONS
# =============================================================================

@jax.jit
def parallel_step_cmos_sbc(
    a: tuple[Array, Array],
    b: tuple[Array, Array],
) -> tuple[Array, Array]:
    """
    Associative operation for parallel scan.
    
    State equation: state[t] = hold[t] * state[t-1] + force_high[t]
    
    This is associative with composition:
        (hold_a, val_a) ∘ (hold_b, val_b) = (hold_a * hold_b, hold_b * val_a + val_b)
    """
    return (
        a[0] * b[0],
        b[0] * a[1] + b[1],
    )


@jax.jit
def parallel_cmos_sbc(
    hold: Array,
    force_high: Array,
    alphas: Array,
    initial_state: Array,
) -> Array:
    """
    Parallel Schmitt trigger state update.
    
    For each timestep:
      - If cands < beta1: state = 0 (force LOW)
      - If beta1 <= cands <= beta2: state = old_state (HOLD / hysteresis)
      - If cands > beta2: state = 1 (force HIGH)
    
    State equation: state[t] = hold[t] * state[t-1] + force_high[t]
    
    When hold=0 and force_high=0 (force_low region), state becomes 0.
    When hold=0 and force_high=1, state becomes 1.
    When hold=1, state keeps previous value.
    """
    # Initial state setup
    initial_state = jnp.expand_dims(initial_state, axis=1)
    initial_hold = jnp.full_like(initial_state, 1.0)

    # Create sequences for scan
    # first = hold coefficients (multiply previous state)
    # second = force_high values (add to result)
    first = jnp.concat([initial_hold, hold], axis=1)
    second = jnp.concat([initial_state, force_high], axis=1)

    # Call associative scan
    scan: tuple[Array, Array] = jax.lax.associative_scan(
        parallel_step_cmos_sbc,
        (first, second),
        axis=1,
    )

    return scan[1][:, 1:] * alphas


def sequential_cmos_sbc(
    hold: Array,
    force_high: Array,
    alphas: Array,
    initial_state: Array,
) -> Array:
    """
    Sequential Schmitt trigger state update.
    
    For each timestep:
      - If cands < beta1: state = 0 (force LOW)
      - If beta1 <= cands <= beta2: state = old_state (HOLD / hysteresis)
      - If cands > beta2: state = 1 (force HIGH)
    
    State equation: state[t] = hold[t] * state[t-1] + force_high[t]
    """
    state = initial_state
    states: list[Array] = list()
    
    for t in range(hold.shape[1]):
        h = hold[:, t]
        fh = force_high[:, t]
        
        # Schmitt trigger update:
        # - force_high=1 -> state = 1
        # - hold=1 -> state = old_state
        # - both=0 (force_low) -> state = 0
        state = h * state + fh
        states.append(state)

    return jnp.stack(states, axis=1) * alphas


@cell
class CMOS_SBC(BaseCell):
    """
    CMOS SBC with Schmitt Trigger behavior.
    
    The state update follows a 3-region Schmitt trigger:
      - cands < beta1: state = 0 (force LOW)
      - beta1 <= cands <= beta2: state = old_state (HOLD / hysteresis)
      - cands > beta2: state = 1 (force HIGH)
    
    beta2 > beta1 is guaranteed by parameterizing as beta1 and delta (hysteresis width),
    where beta2 = beta1 + delta.
    """
    
    _state_dim: int
    _cand_nn: nnx.Linear
    _post_proj_nn: nnx.Linear
    _initial_state: nnx.Variable[Array]
    _alphas: nnx.Param[Array]
    _beta1: nnx.Param[Array]   # Lower threshold
    _delta: nnx.Param[Array]   # Hysteresis width (beta2 = beta1 + delta)
    _parallel: bool
    _kernel_scale: float
    _surr_alpha: nnx.Variable[Array]

    _norm: Optional[nnx.BatchNorm | nnx.LayerNorm]
    _dropout: Optional[nnx.Dropout]
    _post_activation: Optional[Callable[[Array], Array]]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rngs: nnx.Rngs,
        state_dim: int = 64,
        parallel: bool = True,
        norm: str | None = "batch",
        dropout: float = 0.1,
        post_activation: str | None = "relu",
        kernel_scale: float = 1.0,
        surr_alpha: float = 1.0,
        beta1_init_scale: float = 0.3,
        delta_init_scale: float = 0.5,
    ) -> None:
        """
        Initialize a new CMOS SBC with Schmitt Trigger behavior.

        Args:
        -----
        input_size: int
            Size of inputs
        output_size: int
            Size of outputs
        rngs: nnx.Rngs
            Random number generator
        state_dim: int
            Size of states
        parallel: bool
            Whether to use parallel scan or not
        norm: str | None
            Normalization to apply ('batch', 'layer', or None)
        dropout: float
            Dropout ratio
        post_activation: str | None
            Post-activation ('relu', 'gelu', 'glu', or None)
        kernel_scale: float
            Scaling of kernel weights for candidates
        surr_alpha: float
            Alpha parameter for surrogate gradient
        beta1_init_scale: float
            Initialization scale for lower threshold beta1
        delta_init_scale: float
            Initialization scale for hysteresis width delta
        """

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            rngs=rngs,
        )

        self._state_dim = state_dim
        self._parallel = parallel
        self._kernel_scale = kernel_scale

        # Optional normalization before the FCs
        self._norm = None
        if norm is not None:
            if norm == "layer":
                self._norm = nnx.LayerNorm(input_size, rngs=rngs)
            elif norm == "batch":
                self._norm = nnx.BatchNorm(input_size, rngs=rngs)
            else:
                raise ValueError(f"Unknown normalization: '{norm}'.")

        # FC to compute positive candidates only
        self._cand_nn = nnx.Linear(input_size, state_dim, rngs=rngs)
        self._cand_nn.kernel.value = jnp.abs(self._cand_nn.kernel.value) / sqrt(kernel_scale)
        self._cand_nn.bias.value = jnp.abs(self._cand_nn.bias.value)

        # Schmitt trigger thresholds:
        # beta1 = lower threshold (learned, positive via abs())
        # delta = hysteresis width (learned, positive via abs())
        # beta2 = beta1 + delta (computed, guaranteed > beta1)
        beta1 = jnp.abs(jax.random.normal(rngs(), (1, 1, state_dim))) * beta1_init_scale
        delta = jnp.abs(jax.random.normal(rngs(), (1, 1, state_dim))) * delta_init_scale
        self._beta1 = nnx.Param(beta1)
        self._delta = nnx.Param(delta)

        # FC to project to output size
        if post_activation == "glu":
            self._post_proj_nn = nnx.Linear(state_dim, output_size * 2, rngs=rngs)
        else:
            self._post_proj_nn = nnx.Linear(state_dim, output_size, rngs=rngs)

        # Ensure positive weights and biases in projection layer
        self._post_proj_nn.kernel.value = jnp.abs(self._post_proj_nn.kernel.value)
        self._post_proj_nn.bias.value = jnp.abs(self._post_proj_nn.bias.value)

        # Optional activation
        self._post_activation = None
        if post_activation is not None:
            if post_activation == "relu":
                self._post_activation = nnx.relu
            elif post_activation == "gelu":
                self._post_activation = lambda x: nnx.relu(nnx.gelu(x))
            elif post_activation == "glu":
                self._post_activation = lambda x: nnx.relu(nnx.glu(x))
            else:
                raise ValueError(f"Unknown activation: '{post_activation}'.")

        # Optional dropout
        self._dropout = None
        if dropout > 0.0:
            self._dropout = nnx.Dropout(rate=dropout, rngs=rngs)

        # Initial state - positive values in [0, 1]
        initial_state = jax.random.uniform(
            key=rngs(),
            shape=(1, state_dim),
            minval=0.0,
            maxval=1.0,
        )
        self._initial_state = nnx.Variable(initial_state)

        # Positive alphas only
        initializer = nnx.initializers.variance_scaling(
            scale=2.0,
            mode="fan_out",
            distribution="uniform",
        )
        alphas = jnp.abs(initializer(key=rngs(), shape=(1, 1, state_dim)))
        self._alphas = nnx.Param(alphas)

        # Surrogate alpha
        self._surr_alpha = nnx.Variable(jnp.array(surr_alpha))

    @property
    def parallel(self) -> bool:
        return self._parallel

    @staticmethod
    def rec_params() -> Sequence[str]:
        return ["_cand_nn"]

    def init_state(self, batch_size: int) -> Array:
        # Ensure positive initial state
        initial_state = jnp.repeat(self._initial_state.value, batch_size, axis=0)
        # Binarize at runtime: if |value| > 0.5 -> 1, else -> 0
        # Uses surrogate gradient so learning still works
        binary_state = heaviside(jnp.abs(initial_state) - 0.5, self._surr_alpha.value)
        return binary_state * jnp.abs(self._alphas.value[0])

    def __call__(
        self,
        inputs: Array,
        initial_state: Array,
    ) -> dict[str, Array]:

        x = inputs

        # Ensure positive inputs (project to first quadrant)
        x = jnp.maximum(x, 0.0)

        # Optional normalisation
        if self._norm is not None:
            x = self._norm(x)
            x = jnp.maximum(x, 0.0)

        # Compute positive candidates only
        cands = self._cand_nn(x)
        cands = jnp.maximum(cands, 0.0)

        # Get Schmitt trigger thresholds
        # beta1 = lower threshold (positive)
        # beta2 = beta1 + delta (upper threshold, guaranteed > beta1)
        beta1 = jnp.abs(self._beta1.value)
        delta = jnp.abs(self._delta.value)
        beta2 = beta1 + delta  # Ensures beta2 > beta1

        # Broadcast thresholds to match batch and sequence dimensions
        batch_size, seq_len, _ = cands.shape
        beta1_expanded = jnp.broadcast_to(beta1, (batch_size, seq_len, self._state_dim))
        beta2_expanded = jnp.broadcast_to(beta2, (batch_size, seq_len, self._state_dim))

        # =====================================================================
        # SCHMITT TRIGGER LOGIC
        # =====================================================================
        # Three regions:
        #   cands < beta1        -> force_low = 1  -> state = 0
        #   beta1 <= cands <= beta2   -> hold = 1       -> state = old_state  
        #   cands > beta2        -> force_high = 1 -> state = 1
        # =====================================================================
        
        # Region detection with surrogate gradients for training
        above_beta1 = heaviside(cands - beta1_expanded, self._surr_alpha.value)  # 1 if cands ≥ beta1
        above_beta2 = heaviside(cands - beta2_expanded, self._surr_alpha.value)  # 1 if cands > beta2
        
        # Compute the three regions (mutually exclusive, sum to 1)
        force_low = 1 - above_beta1                    # 1 if cands < beta1
        force_high = above_beta2                       # 1 if cands > beta2
        hold = above_beta1 * (1 - above_beta2)         # 1 if beta1 <= cands <= beta2

        # Ensure positive alphas
        alphas = jnp.abs(self._alphas.value)

        # Schmitt trigger state update
        # state[t] = hold[t] * state[t-1] + force_high[t]
        # - When force_high=1: state = 1
        # - When hold=1: state = old_state (hysteresis)
        # - When both=0 (force_low): state = 0
        if self.parallel:
            x = parallel_cmos_sbc(hold, force_high, alphas, initial_state)
        else:
            x = sequential_cmos_sbc(hold, force_high, alphas, initial_state)

        states = x

        # Ensure states remain positive
        states = jnp.maximum(states, 0.0)

        # Projection to output size
        x = self._post_proj_nn(states)

        # Optional post activation
        if self._post_activation is not None:
            x = self._post_activation(x)

        # Ensure final outputs are positive
        x = jnp.maximum(x, 0.0)

        # Optional dropout
        if self._dropout is not None:
            x = self._dropout(x)

        outputs = x

        return {
            "outputs": outputs,
            "states": states,
            "force_low": force_low,
            "hold": hold,
            "force_high": force_high,
            "beta1": beta1_expanded,
            "beta2": beta2_expanded,
            "delta": jnp.broadcast_to(delta, (batch_size, seq_len, self._state_dim)),
            "candidates": cands,
        }