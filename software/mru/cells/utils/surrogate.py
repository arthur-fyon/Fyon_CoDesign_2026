from math import pi
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


@jax.custom_vjp
def _sign(x: Array, alpha: Array) -> Array:
    return 2 * jnp.heaviside(x, 1.0) - 1.0


def _sign_fwd(x: Array, alpha: Array) -> tuple[Array, tuple[Array, Array]]:
    return 2 * jnp.heaviside(x, 1.0) - 1.0, (x, alpha)


def _sign_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, None]:
    x, alpha = res
    derivative = 2*alpha / (1 + (pi * x * alpha) ** 2)
    return (g * derivative, None)


@jax.custom_vjp
def _heaviside(x: Array, alpha: Array) -> Array:
    return jnp.heaviside(x, 1.0)


def _heaviside_fwd(x: Array, alpha: Array) -> tuple[Array, tuple[Array, Array]]:
    return jnp.heaviside(x, 1.0), (x, alpha)


def _heaviside_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, None]:
    x, alpha = res
    derivative = alpha / (1 + (pi * x * alpha) ** 2)
    return (g * derivative, None)


_sign.defvjp(_sign_fwd, _sign_bwd)
_heaviside.defvjp(_heaviside_fwd, _heaviside_bwd)

sign: Callable[[Array, Array], Array] = jax.jit(_sign)
heaviside: Callable[[Array, Array], Array] = jax.jit(_heaviside)
