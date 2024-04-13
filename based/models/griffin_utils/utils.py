# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utilities for both the Jax and Pytorch modules."""

from collections.abc import Sequence
import enum
from typing import Any, NamedTuple, Mapping
import itertools

@enum.unique
class TemporalBlockType(enum.Enum):
  """Type of temporal mixing to use in a residual block."""

  ATTENTION = enum.auto()
  RECURRENT = enum.auto()


@enum.unique
class ScanType(enum.Enum):
  """Which Jax implementation to use for the scan in the RG-LRU in Jax.

  On TPUs Pallas is faster, hence when using `AUTO` the code will pick Pallas
  automatically if you are running on a TPU device and otherwise will fallback
  to the NATIVE Jax for loop.
  """

  AUTO = enum.auto()
  LINEAR_NATIVE = enum.auto()
  ASSOCIATIVE_NATIVE = enum.auto()
  LINEAR_PALLAS = enum.auto()


@enum.unique
class GriffinPreset(enum.Enum):
  """All default preset variants."""

  RECURRENT_GEMMA_2B_V1 = enum.auto()

from dataclasses import dataclass, field, asdict
@dataclass
class GriffinConfig:
  """Griffin config - https://arxiv.org/abs/2402.19427.

  Attributes:
    vocab_size: The number of tokens in the vocabulary.
    width: The dimenonality of the model, e.g. the dimensonality of the
      embeddings and the output of each layer.
    mlp_expanded_width: The width of the hidden layer in the MLP block.
    num_heads: The number of heads for the attention block and the number of
      heads/blocks for the block-diagonal layers used in the RG-LRU gates. This
      number must divide `width` and `lru_width`.
    block_types: A sequence containing the type of the residual blocks in the
      architecture, specifying each block in order if it should use a recurrent
      or an attention sub-block for the temporal-mixing.
    lru_width: The width of the RG-LRU if different from `width`.
    embeddings_scale_by_sqrt_dim: Whether to scale the output of the embeddings
      by `sqrt(width)`.
    attention_window_size: The size of the attention window used in the
      attention block.
    logits_soft_cap: This will cap the values of the final logits to not exceed
      this cap in absolute value by applying a `tanh`. If `None` no capping is
      applied.
    scan_type: If running Flax, this specifies which implementation to use for
      the scan in the RG-LRU.
  """

  vocab_size: int = 50277
  width: int = 1024
  mlp_expanded_width: int = 3072
  num_heads: int = 16
  num_layers: int = 26
  lru_width: int  = 1024 
  embeddings_scale_by_sqrt_dim: bool = True
  attention_window_size: int = 2048
  logits_soft_cap: float = 30.0
  scan_type: ScanType = ScanType.AUTO

  # unused params for compatibility
  reorder_and_upcast_attn: bool = False
  scale_attn_by_inverse_layer_idx: bool = False
  n_positions: int = 0
  n_embd: int = 0
  n_head: int = 0
  use_flash_attn: bool = False
  fused_dropout_add_ln: bool = False
  fused_mlp: bool = False
  fused_bias_fc: bool = False
  n_layer: int = 0
  residual_in_fp32: bool = False
  pad_vocab_size_multiple: int = 0

  @property
  def max_cache_length(self) -> int:
    """The maximum length of the cache used for the model."""
    return self.attention_window_size

  # @property
  # def num_layers(self) -> int:
  #   """The number of layers of the model."""
  #   return len(self.block_types)
  
  # @property
  # def block_types(self):
  #   pattern = (
  #       TemporalBlockType.RECURRENT,
  #       TemporalBlockType.RECURRENT,
  #       TemporalBlockType.ATTENTION,
  #   )
  #   return tuple(itertools.islice(itertools.cycle(pattern), self.num_layers))


  # @classmethod
  # def from_preset(
  #     cls,
  #     vocab_size: int,
  #     width: int,
  #     mlp_expanded_width: int,
  #     num_heads: int,
  #     lru_width: int,
  #     block_types: Sequence[TemporalBlockType],
  #     preset: GriffinPreset,
  #     max_sequence_length: int | None = None,
  # ) -> "GriffinConfig":
  #   """Creates a `GriffinConfig` for a given preset."""
  #   match preset:
  #     case GriffinPreset.RECURRENT_GEMMA_2B_V1:
  #       return cls(
  #           vocab_size=vocab_size,
  #           width=width,
  #           mlp_expanded_width=mlp_expanded_width,
  #           num_heads=num_heads,
  #           lru_width=lru_width,
  #           block_types=tuple(block_types),
  #           embeddings_scale_by_sqrt_dim=True,
  #           attention_window_size=min(2048, max_sequence_length or 2048),
  #           logits_soft_cap=30.0,
  #           scan_type=ScanType.AUTO,
  #       )

  # @classmethod
  # def from_flax_params_or_variables(
  #     cls,
  #     flax_params_or_variables: Mapping[str, Any],
  #     max_sequence_length: int | None = None,
  #     preset: GriffinPreset = GriffinPreset.RECURRENT_GEMMA_2B_V1,
  # ) -> "GriffinConfig":
  #   """Creates a `GriffinConfig` from Flax parameters.

  #   Args:
  #     flax_params_or_variables: The Flax parameters or variables (a dict
  #       containing a key 'params' corresponding to the actual parameters) to
  #       use to reconstruct the config.
  #     max_sequence_length: The maximum sequence length this models is intended
  #       to process. If this is lower than 2048, the `attention_window_size` will
  #       best to this value instead, in order to not create a cache that is
  #       larger than necessary.
  #     preset: Which model preset is being loaded.

  #   Returns:
  #     The reconstructed `GriffinConfig`.
  #   """
  #   if "params" in flax_params_or_variables:
  #     params = flax_params_or_variables["params"]
  #   else:
  #     params = flax_params_or_variables

  #   vocab_size, width = params["embedder"]["input_embedding"].shape
  #   mlp_exp_width = params["blocks.0"]["mlp_block"]["ffw_up"]["w"].shape[-1]

  #   # Defaults
  #   lru_width = None
  #   num_heads = None

  #   block_types = []
  #   i = 0
  #   while f"blocks.{i}" in params:
  #     block_params = params[f"blocks.{i}"]
  #     if "recurrent_block" in block_params:
  #       block_types.append(TemporalBlockType.RECURRENT)

  #       rg_lru = block_params["recurrent_block"]["rg_lru"]
  #       num_heads, head_dim, _ = rg_lru["a_gate"]["w"].shape
  #       lru_width = num_heads * head_dim

  #     elif "attention_block" in block_params:
  #       block_types.append(TemporalBlockType.ATTENTION)

  #       k_proj = block_params["attention_block"]["proj_k"]
  #       heads_dim = k_proj["kernel"].shape[1]
  #       num_heads = width // heads_dim

  #     else:
  #       raise ValueError(
  #           f"Can't recongnize the type of blocks.{i} with keys"
  #           f"{block_params.keys()}."
  #       )

  #     i += 1

  #   return cls.from_preset(
  #       vocab_size=vocab_size,
  #       width=width,
  #       mlp_expanded_width=mlp_exp_width,
  #       num_heads=num_heads,
  #       lru_width=lru_width,
  #       block_types=tuple(block_types),
  #       max_sequence_length=max_sequence_length,
  #       preset=preset,
  #   )

  # @classmethod
  # def from_torch_params(
  #     cls,
  #     params: dict[str, Any],
  #     preset: GriffinPreset = GriffinPreset.RECURRENT_GEMMA_2B_V1,
  # ) -> "GriffinConfig":
  #   """Creates a `GriffinConfig` from Pytorch parameters.

  #   Args:
  #     params: The Pytorch parameters to use to reconstruct the config.
  #     preset: Which model preset is being loaded.

  #   Returns:
  #     The reconstructed `GriffinConfig`.
  #   """

  #   vocab_size, width = params["embedder.input_embedding"].shape
  #   mlp_exp_width = params["blocks.0.mlp_block.ffw_up.w"].shape[-1]

  #   # Defaults
  #   lru_width = None
  #   num_heads = None

  #   block_types = []
  #   i = 0

  #   while f"blocks.{i}.channel_pre_norm.scale" in params:
  #     if f"blocks.{i}.recurrent_block.rg_lru.a_gate.w" in params:
  #       block_types.append(TemporalBlockType.RECURRENT)

  #       w = params[f"blocks.{i}.recurrent_block.rg_lru.a_gate.w"]
  #       num_heads, head_dim, _ = w.shape
  #       lru_width = num_heads * head_dim

  #     elif f"blocks.{i}.attention_block.proj_k.weight" in params:
  #       block_types.append(TemporalBlockType.ATTENTION)

  #       heads_dim = params[f"blocks.{i}.attention_block.proj_k.weight"].shape[1]
  #       num_heads = width // heads_dim

  #     else:
  #       raise ValueError(f"Can't recongnize the type of blocks.{i}.")

  #     i += 1

  #   return cls.from_preset(
  #       vocab_size=vocab_size,
  #       width=width,
  #       mlp_expanded_width=mlp_exp_width,
  #       num_heads=num_heads,
  #       lru_width=lru_width,
  #       block_types=tuple(block_types),
  #       preset=preset,
  #   )