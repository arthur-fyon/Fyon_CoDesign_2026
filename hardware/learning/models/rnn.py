from typing import Any, Mapping, Optional, Sequence

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cells import BaseCell, Cells
from models.base import BaseModel, model
from models.utils import PositionalEncoder


@model
class RNN(BaseModel):

    _cells: Sequence[str]
    _cell_configs: dict[str, dict]
    _num_recs: int
    _positional_encodings_dims: int
    _skip: bool
    _aggregate: str
    _num_fcs: int

    _pre_proj: nnx.Linear
    _positional_encoder: Optional[PositionalEncoder]
    _recs: Sequence[Mapping[str, BaseCell]]
    _fcs: Sequence[nnx.Linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        cells: Optional[Sequence[str]] = None,
        cell_configs: Optional[dict[str, dict]] = None,
        num_recs: int = 1,
        positional_encodings_dims: int = 0,
        skip: bool = False,
        aggregate: str = "sum",
        num_fcs: int = 1,
    ) -> None:
        """
        Initialize a new RNN.

        Args:
        -----
        input_size: int
            Size of inputs
        output_size: int
            Size of outputs
        rngs: nnx.Rngs
            Random number generator
        model_dim: int
            Hidden dimension of model
        cells: Sequence[str]
            List of recurrent cells to use at each layer
        cell_configs: dict[str, dict]
            Configuration of each recurrent cell
        num_recs: int
            Number of recurrent cells
        positional_encodings_dims: int
            Number of dimensions for the positional encodings
        skip: bool
            Whether to add skip connections to the recurrent cells
        aggregate: str
            How to aggregate the outputs of the different cells (when several in each
            recurrent layer) ('sum' or 'concat')
        num_fcs: int
            Number of FC layers after the recurrent layers
        """

        super().__init__(input_size, output_size, rngs)

        if num_recs <= 0:
            raise ValueError("Number of recurrent cells must be greater than 0.")
        if num_fcs <= 0:
            raise ValueError("Number of fully connected layers must be greater than 0.")
        if aggregate not in ["sum", "concat"]:
            raise ValueError(f"Unknown aggregation method, got '{aggregate}'.")

        if cells is None:
            cells = ["sbc"]

        if cell_configs is None:
            cell_configs = {
                "sbc": {
                    "parallel": True,
                    "norm": "batch",
                    "dropout": 0.1,
                }
            }

        for cell_name in cells:
            if cell_name not in Cells.keys():
                raise ValueError(f"Unknown cell, got '{cell_name}'")

        self._model_dim = model_dim
        self._cells = cells
        self._cell_configs = cell_configs
        self._num_recs = num_recs
        self._positional_encodings_dims = positional_encodings_dims
        self._skip = skip
        self._aggregate = aggregate
        self._num_fcs = num_fcs

        # FC to project to model dim
        self._pre_proj = nnx.Linear(input_size, model_dim, rngs=rngs)

        # Positional encoder
        if positional_encodings_dims == 0:
            self._positional_encoder = None

        else:
            self._positional_encoder = PositionalEncoder(
                num_dims=positional_encodings_dims,
            )

        # Recurrent cells
        rec_input_size = model_dim + positional_encodings_dims

        if aggregate == "concat":
            if model_dim % len(cells) != 0:
                raise ValueError(
                    "When in aggregate mode, model_dim should be a multiple "
                    "of the number of cells in each recurrent layer, "
                    f"but {model_dim} is not a multiple of {len(cells)}."
                )

            rec_output_size = model_dim // len(cells)

        else:
            rec_output_size = model_dim

        recs = [
            {
                cell_name: Cells[cell_name](
                    input_size=rec_input_size,
                    output_size=rec_output_size,
                    rngs=rngs,
                    **cell_configs.get(cell_name, {}),
                )
                for cell_name in cells
            }
            for _ in range(num_recs)
        ]
        self._recs = recs

        # Fully connected layers
        fcs_units = [model_dim] * num_fcs + [output_size]
        self._fcs = [
            nnx.Linear(fcs_units[i], fcs_units[i + 1], rngs=rngs)
            for i in range(num_fcs)
        ]

    def init_state(
        self,
        batch_size: int,
    ) -> list[dict[str, Array]]:
        initial_state = [
            {cell_name: cell.init_state(batch_size) for cell_name, cell in rec.items()}
            for rec in self._recs
        ]

        return initial_state

    def __call__(
        self,
        inputs: Array,
        initial_state: list[dict[str, Array]],
    ) -> dict[str, Any]:

        x = inputs

        # Projection to model dim
        x = self._pre_proj(x)

        # Positional encodings
        positional_encodings: Optional[Array] = None
        if self._positional_encoder is not None:
            positional_encodings = self._positional_encoder(x)

        # Go through recurrent cells
        recs_outputs: list[dict[str, dict[str, Array]]] = list()
        for rec, rec_initial_state in zip(self._recs, initial_state):
            # Buffer for skip connection
            skip = None
            if self._skip:
                skip = x

            # Concat positional encodings
            if positional_encodings is not None:
                x = jnp.concat((x, positional_encodings), axis=-1)

            # To accumulate the outputs of the cells
            rec_outputs: dict[str, dict[str, Array]] = {}
            rec_xs: list[Array] = []

            # Forward in cells
            for cell_name, cell in rec.items():
                cell_initial_state = rec_initial_state[cell_name]
                cell_outputs = cell(
                    inputs=x,
                    initial_state=cell_initial_state,
                )

                rec_xs.append(cell_outputs.pop("outputs"))
                rec_outputs[cell_name] = cell_outputs

            # Aggregate
            if self._aggregate == "concat":
                x = jnp.concat(rec_xs, axis=-1)
            else:
                x: Array = sum(rec_xs)  # type:ignore

            # Skip connection
            if skip is not None:
                x = x + skip

            # Append rec_outputs
            recs_outputs.append(rec_outputs)

        # Go through fully-connected layers
        for i in range(len(self._fcs)):
            x = self._fcs[i](x)

            if i < len(self._fcs) - 1:
                x = nnx.gelu(x)

        # Create outputs dir
        outputs = {
            "recs": recs_outputs,
            "outputs": x,
        }

        return outputs
