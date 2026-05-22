from abc import ABC, abstractmethod
from typing import Any, TypeVar

from flax import nnx
from jax import Array

PyTree = Any

Model = TypeVar("Model", bound="BaseModel")
Models: dict[str, type["BaseModel"]] = dict()


def model(cls: type[Model]) -> type[Model]:
    Models[cls.__name__.lower()] = cls
    return cls


class BaseModel(nnx.Module, ABC):
    """
    Abstract class for models.
    Mandatory keys for `init`:
    - input_size: int
        Input size
    - output_size: int
        Output size
    """

    _input_size: int
    _output_size: int
    _rngs: nnx.Rngs

    @abstractmethod
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

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
    ) -> PyTree:
        """
        Return the initial states of the recurrent cells as a PyTree.

        Arguments:
        ----------
        - batch_size: int
            The batch size.

        Returns:
        --------
        - PyTree
            The initial states of the recurrent cells.
        """

        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        inputs: Array,
        initial_state: PyTree,
    ) -> dict[str, Any]:
        """
        Forward pass of the model. Returns a dict with at least the outputs.

        Arguments:
        ----------
        - inputs: Array with shape (batch_size, seq_length, input_size)
            The inputs.
        - initial_state: PyTree
            The initial state of the recurrent cells, returned by `init_state`.

        Returns:
        --------
        - dict[str, Any]
            A dictionnary containing at least the outputs.
        """

        raise NotImplementedError()
