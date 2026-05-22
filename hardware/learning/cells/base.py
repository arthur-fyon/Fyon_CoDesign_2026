from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, TypeVar

from flax import nnx
from jax import Array

Cell = TypeVar("Cell", bound="BaseCell")
Cells: dict[str, type["BaseCell"]] = dict()


def cell(cls: type[Cell]) -> type[Cell]:
    Cells[cls.__name__.lower()] = cls
    return cls


class BaseCell(nnx.Module, ABC):
    """
    An abstract class for recurrent cell.
    Mandatory keys for `init`:
    - input_size: int
        Size of inputs
    - output_size: int
        Size of outputs
    - rngs: nnx.Rngs
        PRN generator
    """

    _input_size: int
    _output_size: int
    _rngs: nnx.Rngs

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._rngs = rngs

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @staticmethod
    @abstractmethod
    def rec_params() -> Sequence[str]:
        """
        Returns a sequence of strings whose contains the name of the parameters that
        should be trained with the recurrent lr.
        """
        raise NotImplementedError()

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
    ) -> Array:
        """
        Returns the initial state of the cell.

        Arguments:
        ----------
        - batch_size: int
            The size of the batch.

        Returns:
        --------
        - Array
            The initial state with shape (batch_size, state_size)
        """

        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        inputs: Array,
        initial_state: Array,
    ) -> dict[str, Array]:
        """
        Forward pass of the cell. Returns a dict containing at least an "outputs" and
        an "states" entry.

        Arguments:
        ----------
        - inputs : Array
            The input tensor, with shape (batch_size, seq_length, input_size).
        - initial_state : Array
            The initial state of the cell with shape (batch_size, state_size).

        Returns:
        --------
        - dict[str, Array]
            A dict containing at least an "outputs" and an "states" entry.
        """

        raise NotImplementedError()
