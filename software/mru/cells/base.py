"""Base class and registry for recurrent cells."""

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar

import jax.numpy as jnp
from flax import nnx
from jax import Array

PyTree = Any

Cell = TypeVar("Cell", bound="BaseCell")
Cells: dict[str, type["BaseCell"]] = dict()


def cell(cls: type[Cell]) -> type[Cell]:
    """Register a cell class in the global Cells registry."""
    Cells[cls.__name__.lower()] = cls
    return cls


class HyperParam(nnx.Variable):
    """Non-trainable mutable hyperparameter.

    Wrapping schedule values (e.g. epsilon) in this Variable subclass instead
    of storing plain jnp.arrays makes them part of NNX's *dynamic state*
    rather than the static graphdef.  This means:
      - JIT sees the updated value on every call without retracing.
      - The optimizer ignores it (only nnx.Param instances are differentiated).
      - Checkpointing includes the current value via nnx.state().
    """
    pass


class EpsilonMixin:
    """
    Mixin for cells with a leaky epsilon parameter.

    Epsilon is a fixed scalar controlled externally (e.g., via decay schedule).
    Subclasses call _init_epsilon() in their __init__ and use _get_epsilon()
    in their forward pass.
    """

    _epsilon: HyperParam

    def _init_epsilon(self, epsilon: float) -> None:
        self._epsilon = HyperParam(jnp.array(float(epsilon)))

    def _get_epsilon(self) -> Array:
        return self._epsilon.value

    def set_epsilon(self, new_epsilon: float) -> None:
        """Set epsilon to a new fixed value."""
        self._epsilon.value = jnp.array(float(new_epsilon))


class BaseCell(nnx.Module, ABC):
    """
    Abstract base class for recurrent cells.
    
    All cells must implement:
    - __init__: Initialize the cell
    - init_state: Create initial state for a batch
    - __call__: Forward pass through the cell
    """
    
    _state_dim: int
    _input_size: int
    _output_size: int
    
    def __init__(
        self,
        input_size: int,
        state_dim: int,
        output_size: int,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> None:
        """
        Initialize base cell.
        
        Args:
        -----
        input_size: int
            Input dimension
        state_dim: int
            Hidden state dimension
        output_size: int
            Output dimension
        rngs: nnx.Rngs
            Random number generator
        **kwargs: Additional cell-specific arguments
        """
        super().__init__()
        
        self._input_size = input_size
        self._state_dim = state_dim
        self._output_size = output_size
    
    @property
    def state_dim(self) -> int:
        """Hidden state dimension."""
        return self._state_dim
    
    @property
    def input_size(self) -> int:
        """Input dimension."""
        return self._input_size
    
    @property
    def output_size(self) -> int:
        """Output dimension."""
        return self._output_size
    
    @staticmethod
    def rec_params() -> Sequence[str]:
        """
        Return names of recurrent parameters (for differential learning rates).

        Override in subclasses to specify which parameters should have
        different learning rates in the optimizer.
        """
        return []

    @staticmethod
    def no_weight_decay_params() -> Sequence[str]:
        """
        Return names of parameters that should be excluded from weight decay.

        Override in subclasses to specify which parameters must not be
        regularised by weight decay (e.g. angular/phase parameters whose
        natural range is not centred on zero).
        """
        return []
    
    @abstractmethod
    def init_state(self, batch_size: int) -> PyTree:
        """
        Initialize hidden state for a batch.
        
        Args:
        -----
        batch_size: int
            Batch size
            
        Returns:
        --------
        PyTree: Initial state (typically Array of shape (batch_size, state_dim))
        """
        raise NotImplementedError()
    
    @abstractmethod
    def __call__(
        self,
        inputs: Array,
        initial_state: PyTree,
        training: bool = False,
        return_infos: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass through the cell.
        
        Args:
        -----
        inputs: Array
            Input tensor of shape (batch_size, seq_length, input_size)
        initial_state: PyTree
            Initial state from init_state()
        training: bool
            Whether in training mode (affects dropout, etc.)
        return_infos: bool
            Whether to return diagnostic information
            
        Returns:
        --------
        dict containing at minimum:
        - outputs: Array of shape (batch_size, seq_length, output_size)
        
        When return_infos=True, may also contain:
        - states: Hidden states over time
        - updates: Update masks
        - Any other cell-specific diagnostics
        """
        raise NotImplementedError()