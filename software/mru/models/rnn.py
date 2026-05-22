from typing import Any, Optional, Sequence

import jax.numpy as jnp
from flax import nnx
from jax import Array

from mru.cells import Cells
from .base import BaseModel, model
from .layers import MLP, RMSNorm, ScaleProj
from .positional import PositionalEncoder, LearnablePositionalEncoder


@model
class RNN(BaseModel):
    """Recurrent Neural Network with Pre-Norm pattern and optional output gating."""
    
    # Architecture config
    _cells: Sequence[str]
    _cell_configs: dict[str, dict]
    _num_recs: int
    _model_dim: int
    _skip: bool
    _aggregate: str
    _reinject_pe_per_layer: bool
    _post_rec_norm: str
    _use_gating: bool
    
    # Layers
    _pre_proj: nnx.Linear
    _pre_pe_proj: Optional[nnx.Linear]  # Project concat[x, PE] at input
    _pre_mlp: MLP  # MLP before recurrent stack
    _post_mlp: MLP  # MLP after recurrent stack
    _recs: nnx.List  # Recurrent cells
    _rec_pre_norms: nnx.List  # Pre-recurrence normalization
    _rec_post_norms: nnx.List  # Post-recurrence normalization (optional)
    _gate_projs: nnx.List  # Output gating: y = Gate(x) ⊙ h
    _mlp_pre_norms: nnx.List  # Pre-MLP normalization
    _rec_mlps: nnx.List  # MLPs within each recurrent block
    _concat_projs: nnx.List  # Projection after concatenating multiple cells
    _pe_projs: nnx.List  # Positional encoding projection per layer (optional)
    _rec_convs: nnx.List  # Causal 1D conv before each recurrent cell (optional)
    _skip_projs: nnx.List  # Skip projections for recurrence
    _skip_projs_mlp: nnx.List  # Skip projections for MLP
    _output_proj: nnx.Linear
    _rec_dropout: Optional[nnx.Dropout]
    _mlp_dropout: Optional[nnx.Dropout]
    _bidir_flags: list[dict[str, bool]]
    _pos_encoder: Optional[nnx.Module]  # PE encoder module (kept if learnable)
    _pos_encodings: Optional[Array]  # Precomputed PE (used if not learnable)
    _learnable_pe: bool  # Whether PE is learnable
    
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
        learnable_pe: bool = True,
        reinject_pe_per_layer: bool = True,
        pe_projection_per_layer: bool = False,
        post_rec_norm: str = "partial",
        skip: bool = True,
        aggregate: str = "sum",
        norm: str = "rms",
        mlp_hidden_dim: Optional[int] = None,
        mlp_activation: str = "glu",
        mlp_dropout: float = 0.1,
        mlp_ff_expansion: int = 4,
        rec_dropout: float = 0.0,
        use_gating: bool = True,
        max_seq_len: int = 65536,
        conv_kernel_size: int = 0,
    ):
        super().__init__(input_size=input_size, output_size=output_size, rngs=rngs)
        
        # Validate post_rec_norm parameter
        if post_rec_norm not in ["none", "all", "partial"]:
            raise ValueError(f"post_rec_norm must be 'none', 'all', or 'partial', got '{post_rec_norm}'")
        
        self._cells = cells
        self._cell_configs = cell_configs or {}
        self._num_recs = num_recs
        self._model_dim = model_dim
        self._skip = skip
        self._aggregate = aggregate
        self._reinject_pe_per_layer = reinject_pe_per_layer
        self._post_rec_norm = post_rec_norm
        self._learnable_pe = learnable_pe
        self._use_gating = use_gating
        
        mlp_hidden_dim = mlp_hidden_dim or model_dim
        
        # Normalization factory
        def make_norm():
            if norm == "batch":
                return nnx.BatchNorm(model_dim, use_running_average=False, rngs=rngs)
            elif norm == "layer":
                return nnx.LayerNorm(model_dim, rngs=rngs)
            elif norm == "rms":
                return RMSNorm(model_dim, rngs=rngs)
            else:
                return None
        
        # Input projection
        self._pre_proj = nnx.Linear(input_size, model_dim, rngs=rngs)
        
        # Positional encodings
        if positional_encodings_dims > 0:
            if learnable_pe:
                self._pos_encoder = LearnablePositionalEncoder(
                    num_dims=positional_encodings_dims,
                    max_seq_len=max_seq_len,
                    rngs=rngs,
                )
                self._pos_encodings = None
            else:
                pos_encoder = PositionalEncoder(
                    num_dims=positional_encodings_dims,
                    max_seq_len=max_seq_len,
                    rngs=rngs,
                )
                dummy_input = jnp.zeros((1, max_seq_len, model_dim))
                self._pos_encodings = pos_encoder(dummy_input)[0]
                self._pos_encoder = None
            
            # Projection for concat[x, PE] at input level (NO residual)
            self._pre_pe_proj = nnx.Linear(model_dim + positional_encodings_dims, model_dim, rngs=rngs)
        else:
            self._pos_encodings = None
            self._pos_encoder = None
            self._pre_pe_proj = None
        
        self._pre_mlp = MLP(model_dim, model_dim, mlp_hidden_dim, mlp_activation, mlp_dropout, mlp_ff_expansion, rngs)
        self._post_mlp = MLP(model_dim, model_dim, mlp_hidden_dim, mlp_activation, mlp_dropout, mlp_ff_expansion, rngs)
        
        # Dropouts
        self._rec_dropout = nnx.Dropout(rate=rec_dropout, rngs=rngs) if rec_dropout > 0.0 else None
        self._mlp_dropout = nnx.Dropout(rate=mlp_dropout, rngs=rngs) if mlp_dropout > 0.0 else None
        
        # Build recurrent blocks
        recs_items = []
        rec_pre_norms_items = []
        rec_post_norms_items = []
        rec_convs_items = []
        gate_projs_items = []
        mlp_pre_norms_items = []
        rec_mlps_items = []
        concat_projs_items = []
        pe_projs_items = []
        skip_projs_items = []
        skip_projs_mlp_items = []
        bidir_flags_items = []

        for layer_idx in range(num_recs):
            rec_layer = {}
            bidir_flags = {}
            
            # Determine cell input size (accounting for PE reinjection and bidirectionality)
            base_cell_input_dim = model_dim
            # Only add PE dims to cell input if we're reinjecting PE at each layer
            if positional_encodings_dims > 0 and reinject_pe_per_layer and not pe_projection_per_layer:
                base_cell_input_dim += positional_encodings_dims
            
            # Determine normalization dimension for pre-recurrence norm
            # If we reinject PE without projection, norm needs to handle the concatenated dimension
            pre_norm_dim = base_cell_input_dim if (positional_encodings_dims > 0 and reinject_pe_per_layer and not pe_projection_per_layer) else model_dim
            
            for cell_name in self._cells:
                cell_cls = Cells.get(cell_name)
                if cell_cls is None:
                    raise ValueError(f"Unknown cell: {cell_name}")

                cell_config = self._cell_configs.get(cell_name, {})
                is_bidirectional = cell_config.get("bidirectional", False)
                bidir_flags[cell_name] = is_bidirectional
                
                # Remove bidirectional flag from cell config
                cell_config_clean = {k: v for k, v in cell_config.items() if k != "bidirectional"}
                
                # Calculate actual input size to cell.
                # When a causal conv precedes the cell it always outputs model_dim,
                # so the cell input is model_dim regardless of PE reinjection.
                effective_input_dim = model_dim if conv_kernel_size > 0 else base_cell_input_dim
                if is_bidirectional:
                    # Bidirectional: concat[forward, backward] so double the input
                    cell_input_size = 2 * effective_input_dim
                else:
                    cell_input_size = effective_input_dim
                
                rec_layer[cell_name] = cell_cls(
                    input_size=cell_input_size,
                    output_size=model_dim,
                    rngs=rngs,
                    **cell_config_clean,
                )

            recs_items.append(nnx.Dict(rec_layer))
            bidir_flags_items.append(bidir_flags)

            # PE projection per layer (optional - only used if reinjecting PE)
            # Projects concat[x, PE] back to model_dim
            if positional_encodings_dims > 0 and reinject_pe_per_layer and pe_projection_per_layer:
                pe_proj = nnx.Linear(
                    model_dim + positional_encodings_dims, 
                    model_dim, 
                    rngs=rngs
                )
                pe_projs_items.append(pe_proj)
            else:
                pe_projs_items.append(None)

            # Pre-recurrence normalization (with correct dimension for PE reinjection)
            if norm == "batch":
                rec_pre_norms_items.append(nnx.BatchNorm(pre_norm_dim, use_running_average=False, rngs=rngs))
            elif norm == "layer":
                rec_pre_norms_items.append(nnx.LayerNorm(pre_norm_dim, rngs=rngs))
            elif norm == "rms":
                rec_pre_norms_items.append(RMSNorm(pre_norm_dim, rngs=rngs))
            else:
                rec_pre_norms_items.append(None)

            # Causal 1D conv before each recurrent cell (optional)
            # Pad (kernel_size-1) on the left so each position only sees past tokens.
            if conv_kernel_size > 0:
                rec_convs_items.append(nnx.Conv(
                    in_features=pre_norm_dim,
                    out_features=model_dim,
                    kernel_size=(conv_kernel_size,),
                    padding=((conv_kernel_size - 1, 0),),
                    rngs=rngs,
                ))
            else:
                rec_convs_items.append(None)

            # Post-recurrence normalization (optional, depends on post_rec_norm setting)
            # "none": no post-rec norm
            # "all": post-rec norm on all layers
            # "partial": post-rec norm on all layers except the last one
            is_last_layer = (layer_idx == num_recs - 1)
            
            if post_rec_norm == "all":
                rec_post_norms_items.append(make_norm())
            elif post_rec_norm == "partial" and not is_last_layer:
                rec_post_norms_items.append(make_norm())
            else:  # "none" or ("partial" and is_last_layer)
                rec_post_norms_items.append(None)

            # Output gating projection: y = Gate(x_norm) ⊙ h (gate from normalized input)
            # Only create gate projection if gating is enabled
            if use_gating:
                gate_proj = nnx.Linear(pre_norm_dim, model_dim, rngs=rngs)
                # Bias toward open gates at initialization
                gate_proj.bias.value = jnp.full(model_dim, 4.0)
                # sigma (4.0) ~ 0.982 -> initially mostly open gates
                gate_projs_items.append(gate_proj)
            else:
                gate_projs_items.append(None)

            # Pre-MLP normalization
            mlp_pre_norms_items.append(make_norm())

            rec_mlps_items.append(MLP(model_dim, model_dim, mlp_hidden_dim, mlp_activation, mlp_dropout, mlp_ff_expansion, rngs))

            # Concat projection (if aggregating multiple cells by concatenation)
            if aggregate == "concat" and len(self._cells) > 1:
                concat_dim = len(self._cells) * model_dim
                concat_projs_items.append(nnx.Linear(concat_dim, model_dim, rngs=rngs))
            else:
                concat_projs_items.append(None)

            # Skip projections with variance-preserving initialization (scale = 1.0)
            if skip:
                skip_scale_init = 1.0

                skip_proj = ScaleProj(model_dim, dropout=0.0, scaling_factor=skip_scale_init, rngs=rngs)
                skip_proj_mlp = ScaleProj(model_dim, dropout=0.0, scaling_factor=skip_scale_init, rngs=rngs)
                
                skip_projs_items.append(skip_proj)
                skip_projs_mlp_items.append(skip_proj_mlp)
            else:
                skip_projs_items.append(None)
                skip_projs_mlp_items.append(None)

        self._recs = nnx.List(recs_items)
        self._rec_pre_norms = nnx.List(rec_pre_norms_items)
        self._rec_post_norms = nnx.List(rec_post_norms_items)
        self._rec_convs = nnx.List(rec_convs_items)
        self._gate_projs = nnx.List(gate_projs_items)
        self._mlp_pre_norms = nnx.List(mlp_pre_norms_items)
        self._rec_mlps = nnx.List(rec_mlps_items)
        self._concat_projs = nnx.List(concat_projs_items)
        self._pe_projs = nnx.List(pe_projs_items)
        self._skip_projs = nnx.List(skip_projs_items)
        self._skip_projs_mlp = nnx.List(skip_projs_mlp_items)
        self._bidir_flags = bidir_flags_items
        self._output_proj = nnx.Linear(model_dim, output_size, rngs=rngs)
    
    @staticmethod
    def _apply_norm(norm, x: Array, training: bool) -> Array:
        """Apply normalization layer, handling BatchNorm's use_running_average flag."""
        if norm is None:
            return x
        if isinstance(norm, nnx.BatchNorm):
            return norm(x, use_running_average=not training)
        return norm(x)

    def init_state(self, batch_size: int) -> list[dict[str, Array]]:
        """Initialize recurrent states."""
        states = []
        for rec in self._recs:
            rec_state = {}
            for key, cell in rec.items():
                rec_state[key] = cell.init_state(batch_size)
            states.append(rec_state)
        return states
    
    def __call__(
        self,
        inputs: Array,
        initial_state: list[dict[str, Array]],
        training: bool = False,
        return_all_states: bool = False,
        return_infos: bool = False,
        return_carry: bool = False,
    ) -> dict[str, Any]:
        """Forward pass with Pre-Norm pattern and output gating.
        
        Implemented architecture:
            Stage 0: 
                x₀ = Linear(input)
                if PE: x₀ = Linear_pe(concat[x₀, PE])  # No residual
                x₀ = PreMLP(x₀) + x₀
            
            Stage 1-N (per block i):
                # Recurrence Sublayer
                skip = x (if enabled)
                if reinject_pe_per_layer:
                    x̃ = concat[x, PE(t)]            [Add PE]
                    if pe_proj: x̃ = Linear_pe(x̃)   [Optional projection]
                else:
                    x̃ = x                           [No PE reinjection]
                x̂ = LayerNorm(x̃)                   [Pre-norm]
                h = Recurrence(x̂)                   [Cell computation]
                if concat: h = Linear(h)            [Optional concat projection]
                if post_rec_norm: h = LayerNorm(h)  [Optional post-rec norm]
                if use_gating:
                    y = Gate(x̂) ⊙ h                 [Gated output from normalized input]
                else:
                    y = h                           [Direct output without gating]
                y = Dropout_rec(y)
                x' = ScaleProj(skip, y) or x + y   [Skip/residual]
                
                # MLP Sublayer
                skip = x' (if enabled)
                x̂' = LayerNorm(x')                 [Pre-norm]
                m = MLP(x̂')                        [FFN]
                m = Dropout_mlp(m)
                x = ScaleProj(skip, m) or x' + m   [Skip/residual]
            
            Stage N+1: 
                xₙ₊₁ = PostMLP(xₙ) + xₙ
                output = Linear(xₙ₊₁)
        """
        
        x = inputs
        
        # STAGE 0: Input processing
        # Input projection
        x = self._pre_proj(x)  # [B, T, D]
        
        # Generate positional encodings
        pos_enc = None
        if self._pos_encoder is not None:
            # Learnable PE: call encoder dynamically
            pos_enc = self._pos_encoder(x)  # [B, T, pe_dim]
            
            # Add PE to input (NO residual connection)
            x_with_pe = jnp.concatenate([x, pos_enc], axis=-1)
            x = self._pre_pe_proj(x_with_pe)
        elif self._pos_encodings is not None:
            # Fixed PE: use precomputed encodings (more efficient)
            seq_len = x.shape[1]
            batch_size = x.shape[0]
            # Slice and tile for current sequence
            pos_enc = jnp.tile(
                self._pos_encodings[None, :seq_len, :],  # (1, seq_len, pe_dim)
                (batch_size, 1, 1)
            )
            
            # Add PE to input (NO residual connection)
            x_with_pe = jnp.concatenate([x, pos_enc], axis=-1)
            x = self._pre_pe_proj(x_with_pe)
        
        # Pre-recurrent stack MLP with residual
        x = self._pre_mlp(x, training=training) + x
        
        # STAGE 1-N: Recurrent blocks
        recs_outputs = []
        carry_rec_states = []   # for return_carry: final cell states per layer
        carry_conv_buffers = []  # for return_carry: last (ks-1) x_norm values per layer

        for layer_idx, (
            rec, rec_state, bidir_flags,
            rec_pre_norm, rec_post_norm, rec_conv, pe_proj,
            gate_proj, mlp_pre_norm, mlp, concat_proj,
            skip_proj, skip_proj_mlp
        ) in enumerate(zip(
            self._recs, initial_state, self._bidir_flags,
            self._rec_pre_norms, self._rec_post_norms, self._rec_convs, self._pe_projs,
            self._gate_projs, self._mlp_pre_norms, self._rec_mlps, self._concat_projs,
            self._skip_projs, self._skip_projs_mlp
        )):
            # === RECURRENCE SUBLAYER ===
            skip = x if self._skip else None
            
            # Add positional encoding (only if reinjecting PE per layer)
            if pos_enc is not None and self._reinject_pe_per_layer:
                x_with_pe = jnp.concatenate([x, pos_enc], axis=-1)
                # Optional: project concatenated features to model_dim
                if pe_proj is not None:
                    x_with_pe = pe_proj(x_with_pe)
            else:
                # No PE reinjection - use x directly
                x_with_pe = x
            
            x_norm = self._apply_norm(rec_pre_norm, x_with_pe, training)

            # Save pre-conv x_norm for conv buffer (used when return_carry=True)
            x_norm_pre_conv = x_norm

            # Optional causal conv before recurrent cell (Conv4 → cell pattern)
            if rec_conv is not None:
                x_norm = rec_conv(x_norm)

            # Recurrence
            rec_outputs = {}
            rec_hs = []  # Hidden states from cells

            for cell_name, cell in rec.items():
                is_bidirectional = bidir_flags.get(cell_name, False)

                if is_bidirectional:
                    # Bidirectional: concat[forward, backward]
                    x_flipped = jnp.flip(x_norm, axis=1)
                    x_bidir = jnp.concatenate([x_norm, x_flipped], axis=-1)

                    cell_output = cell(x_bidir, rec_state[cell_name], training, return_infos)
                    rec_hs.append(cell_output.pop("outputs"))
                    rec_outputs[cell_name] = cell_output
                else:
                    # Unidirectional
                    cell_output = cell(x_norm, rec_state[cell_name], training, return_infos)
                    rec_hs.append(cell_output.pop("outputs"))
                    rec_outputs[cell_name] = cell_output

            # Collect carry state for stateful generation
            if return_carry:
                layer_final_states = {
                    cell_name: cell_out["final_state"]
                    for cell_name, cell_out in rec_outputs.items()
                    if "final_state" in cell_out
                }
                carry_rec_states.append(layer_final_states)
                if rec_conv is not None:
                    ks = rec_conv.kernel.value.shape[0]
                    buf_len = ks - 1
                    seq_len = x_norm_pre_conv.shape[1]
                    if seq_len >= buf_len:
                        buf = x_norm_pre_conv[:, -buf_len:, :]
                    else:
                        # Prompt shorter than kernel; left-pad with zeros
                        pad = jnp.zeros((x_norm_pre_conv.shape[0], buf_len - seq_len, x_norm_pre_conv.shape[2]))
                        buf = jnp.concatenate([pad, x_norm_pre_conv], axis=1)
                    carry_conv_buffers.append(buf)
                else:
                    carry_conv_buffers.append(None)
            
            # Aggregate multiple cell outputs
            if self._aggregate == "concat" and len(rec_hs) > 1:
                h = jnp.concatenate(rec_hs, axis=-1)  # [B, T, N*D]
                if concat_proj is not None:
                    h = concat_proj(h)  # Project back to model_dim
            else:  # sum
                h = sum(rec_hs) if len(rec_hs) > 1 else rec_hs[0]
            
            h = self._apply_norm(rec_post_norm, h, training)
            
            # Output gating: y = Gate(x_norm) ⊙ h (gate from normalized input)
            # If gating is disabled, directly use h
            if gate_proj is not None:
                gate_logits = gate_proj(x_norm)  # Gate from normalized input!
                gate = nnx.sigmoid(gate_logits)
                y = gate * h
            else:
                y = h
            
            # Dropout on gated output
            if self._rec_dropout is not None:
                y = self._rec_dropout(y, deterministic=not training)
            
            # Skip/Residual connection
            if skip is not None:
                x = skip_proj(skip, y, training)
            else:
                x = x + y
            
            # === MLP SUBLAYER ===
            skip = x if self._skip else None
            
            x_norm = self._apply_norm(mlp_pre_norm, x, training)

            # MLP
            m = mlp(x_norm, training=training)
            
            # Dropout on MLP output
            if self._mlp_dropout is not None:
                m = self._mlp_dropout(m, deterministic=not training)
            
            # Skip/Residual connection
            if skip is not None:
                x = skip_proj_mlp(skip, m, training)
            else:
                x = x + m
            
            recs_outputs.append(rec_outputs)
        
        # STAGE N+1: Post-recurrent stack processing
        # Post-MLP with residual connection
        x = self._post_mlp(x, training=training) + x
        
        outputs = self._output_proj(x)
        
        # Aggregate z_mean across all cells and layers into a single penalty scalar
        z_means = [
            cell_out["z_mean"]
            for rec_layer in recs_outputs
            for cell_out in rec_layer.values()
            if "z_mean" in cell_out
        ]
        z_penalty = jnp.mean(jnp.stack(z_means)) if z_means else jnp.zeros(())

        result = {"recs": recs_outputs, "outputs": outputs, "z_penalty": z_penalty}
        if return_all_states:
            result["all_states"] = x
        if return_carry:
            result["final_rec_states"] = carry_rec_states
            result["final_conv_buffers"] = carry_conv_buffers

        return result
    
    def step_generate(
        self,
        token_oh: Array,
        rec_states: list[dict[str, Array]],
        conv_buffers: list,
    ):
        """
        Process a single token during autoregressive generation.

        token_oh      : (1, 1, vocab_size) one-hot for the current token
        rec_states    : list[dict[cell_name -> (1, state_dim)]] — one entry per layer,
                        as returned by __call__(return_carry=True)["final_rec_states"]
        conv_buffers  : list[(1, kernel_size-1, model_dim) | None] — one per layer

        Returns (logits, new_rec_states, new_conv_buffers) where logits is (1, vocab_size).
        """
        import jax

        x = token_oh  # (1, 1, vocab_size)

        # STAGE 0
        x = self._pre_proj(x)  # (1, 1, model_dim)
        x = self._pre_mlp(x, training=False) + x

        new_rec_states = []
        new_conv_buffers = []

        for layer_idx, (
            rec, rec_state_layer, bidir_flags,
            rec_pre_norm, rec_post_norm, rec_conv, pe_proj,
            gate_proj, mlp_pre_norm, mlp, concat_proj,
            skip_proj, skip_proj_mlp,
            conv_buffer,
        ) in enumerate(zip(
            self._recs, rec_states, self._bidir_flags,
            self._rec_pre_norms, self._rec_post_norms, self._rec_convs, self._pe_projs,
            self._gate_projs, self._mlp_pre_norms, self._rec_mlps, self._concat_projs,
            self._skip_projs, self._skip_projs_mlp,
            conv_buffers,
        )):
            skip = x if self._skip else None

            # No PE reinjection for LM configs (positional_encodings_dims=0)
            x_norm = self._apply_norm(rec_pre_norm, x, training=False)  # (1, 1, D)
            x_norm_pre_conv = x_norm

            # Causal conv: use explicit left-context from buffer instead of zero-padding
            if rec_conv is not None:
                ks = rec_conv.kernel.value.shape[0]
                if conv_buffer is None:
                    conv_buffer = jnp.zeros((1, ks - 1, self._model_dim))
                # Concatenate buffer + current token: (1, ks, D)
                x_ctx = jnp.concatenate([conv_buffer, x_norm], axis=1)
                # Apply conv with VALID padding (no extra padding since we provide full context)
                x_norm = jax.lax.conv_general_dilated(
                    x_ctx,
                    rec_conv.kernel.value,
                    window_strides=(1,),
                    padding="VALID",
                    dimension_numbers=("NHC", "HIO", "NHC"),
                )
                if rec_conv.use_bias:
                    x_norm = x_norm + rec_conv.bias.value
                new_conv_buffer = jnp.concatenate([conv_buffer[:, 1:, :], x_norm_pre_conv], axis=1)
            else:
                new_conv_buffer = None

            # Recurrence (each cell sees a sequence of length 1)
            rec_hs = []
            layer_new_states = {}
            for cell_name, cell in rec.items():
                cell_state = rec_state_layer[cell_name]  # (1, state_dim)
                cell_output = cell(x_norm, cell_state, training=False, return_infos=False)
                rec_hs.append(cell_output["outputs"])  # (1, 1, model_dim)
                layer_new_states[cell_name] = cell_output["final_state"]  # (1, state_dim)

            # Aggregate
            if self._aggregate == "concat" and len(rec_hs) > 1:
                h = jnp.concatenate(rec_hs, axis=-1)
                if concat_proj is not None:
                    h = concat_proj(h)
            else:
                h = sum(rec_hs) if len(rec_hs) > 1 else rec_hs[0]

            h = self._apply_norm(rec_post_norm, h, training=False)

            if gate_proj is not None:
                gate = nnx.sigmoid(gate_proj(x_norm))
                y = gate * h
            else:
                y = h

            if skip is not None:
                x = skip_proj(skip, y, training=False)
            else:
                x = x + y

            # MLP sublayer
            skip = x if self._skip else None
            x_norm_mlp = self._apply_norm(mlp_pre_norm, x, training=False)
            m = mlp(x_norm_mlp, training=False)
            if skip is not None:
                x = skip_proj_mlp(skip, m, training=False)
            else:
                x = x + m

            new_rec_states.append(layer_new_states)
            new_conv_buffers.append(new_conv_buffer)

        # STAGE N+1
        x = self._post_mlp(x, training=False) + x
        logits = self._output_proj(x)  # (1, 1, vocab_size)
        logits = logits[0, 0, :]       # (vocab_size,)

        return logits, new_rec_states, new_conv_buffers

    def run_sanity_check(self, inputs: Array, initial_state: list[dict[str, Array]]) -> bool:
        """Run sanity checks on all cells by simulating the actual forward pass."""
        x = inputs
        
        # STAGE 0: Input processing (mimic forward pass)
        x = self._pre_proj(x)
        
        # Generate positional encodings
        pos_enc = None
        if self._pos_encoder is not None:
            # Learnable PE: call encoder
            pos_enc = self._pos_encoder(x)
            x_with_pe = jnp.concatenate([x, pos_enc], axis=-1)
            x = self._pre_pe_proj(x_with_pe)
        elif self._pos_encodings is not None:
            # Fixed PE: use precomputed
            seq_len = x.shape[1]
            batch_size = x.shape[0]
            pos_enc = jnp.tile(
                self._pos_encodings[None, :seq_len, :],
                (batch_size, 1, 1)
            )
            x_with_pe = jnp.concatenate([x, pos_enc], axis=-1)
            x = self._pre_pe_proj(x_with_pe)
        
        x = self._pre_mlp(x, training=False) + x
        
        # STAGE 1-N: Test each recurrent layer properly
        for layer_idx, (
            rec, rec_state, bidir_flags,
            rec_pre_norm, rec_post_norm, rec_conv, pe_proj,
            gate_proj, mlp_pre_norm, mlp, concat_proj,
            skip_proj, skip_proj_mlp
        ) in enumerate(zip(
            self._recs, initial_state, self._bidir_flags,
            self._rec_pre_norms, self._rec_post_norms, self._rec_convs, self._pe_projs,
            self._gate_projs, self._mlp_pre_norms, self._rec_mlps, self._concat_projs,
            self._skip_projs, self._skip_projs_mlp
        )):
            # === RECURRENCE SUBLAYER ===
            skip = x if self._skip else None
            
            # Add positional encoding (mimic forward pass logic)
            if pos_enc is not None and self._reinject_pe_per_layer:
                x_with_pe = jnp.concatenate([x, pos_enc], axis=-1)
                if pe_proj is not None:
                    x_with_pe = pe_proj(x_with_pe)
            else:
                x_with_pe = x
            
            x_norm = self._apply_norm(rec_pre_norm, x_with_pe, training=False)

            # Optional causal conv before recurrent cell
            if rec_conv is not None:
                x_norm = rec_conv(x_norm)

            # Test each cell's sanity check
            rec_hs = []
            for cell_name, cell in rec.items():
                if hasattr(cell, "sanity_check"):
                    is_bidirectional = bidir_flags.get(cell_name, False)
                    
                    # Prepare input as it would be in forward pass
                    if is_bidirectional:
                        x_flipped = jnp.flip(x_norm, axis=1)
                        x_test = jnp.concatenate([x_norm, x_flipped], axis=-1)
                    else:
                        x_test = x_norm
                    
                    if not cell.sanity_check(x_test, rec_state[cell_name]):
                        print(f"Sanity check failed for layer {layer_idx}, cell '{cell_name}'")
                        return False
                
                # Simulate cell output for next layer
                cell_output = cell(
                    x_norm if not bidir_flags.get(cell_name, False) else jnp.concatenate([x_norm, jnp.flip(x_norm, axis=1)], axis=-1),
                    rec_state[cell_name],
                    training=False,
                    return_infos=False
                )
                rec_hs.append(cell_output["outputs"])
            
            # Aggregate outputs
            if self._aggregate == "concat" and len(rec_hs) > 1:
                h = jnp.concatenate(rec_hs, axis=-1)
                if concat_proj is not None:
                    h = concat_proj(h)
            else:
                h = sum(rec_hs) if len(rec_hs) > 1 else rec_hs[0]
            
            h = self._apply_norm(rec_post_norm, h, training=False)
            
            # Gating (gate from normalized x)
            # If gating is disabled, directly use h
            if gate_proj is not None:
                gate_logits = gate_proj(x_norm)
                gate = nnx.sigmoid(gate_logits)
                y = gate * h
            else:
                y = h
            
            # Skip/Residual
            if skip is not None:
                x = skip_proj(skip, y, training=False)
            else:
                x = x + y
            
            # === MLP SUBLAYER ===
            skip = x if self._skip else None
            
            x_norm = self._apply_norm(mlp_pre_norm, x, training=False)

            m = mlp(x_norm, training=False)
            
            # Skip/Residual
            if skip is not None:
                x = skip_proj_mlp(skip, m, training=False)
            else:
                x = x + m
        
        return True
    
    def print_cell_param_stats(self, prefix: str = "") -> None:
        """Print a compact summary of key cell parameters (for cells that support it)."""
        for layer_idx, rec in enumerate(self._recs):
            for cell_name, cell in rec.items():
                if hasattr(cell, "cell_param_stats"):
                    stats = cell.cell_param_stats()
                    print(f"{prefix}[{cell_name} L{layer_idx}] {stats}", flush=True)

    def set_surr_alpha_all(self, new_surr_alpha: float):
        """Set surrogate alpha for all cells."""
        for rec in self._recs:
            for cell_key, cell in rec.items():
                if hasattr(cell, "set_surr_alpha"):
                    cell.set_surr_alpha(new_surr_alpha)

    def set_epsilon_all(self, new_epsilon: float):
        """Set epsilon for all cells."""
        for rec in self._recs:
            for cell_key, cell in rec.items():
                if hasattr(cell, "set_epsilon"):
                    cell.set_epsilon(new_epsilon)
    
    def scale_surr_alpha_all(self, scale: float):
        """Scale surrogate alpha for all cells."""
        for rec in self._recs:
            for cell_key, cell in rec.items():
                if hasattr(cell, "scale_surr_alpha"):
                    cell.scale_surr_alpha(scale)