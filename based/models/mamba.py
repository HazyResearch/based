# Copyright (c) 2023, Albert Gu, Tri Dao.
# This class is used to construct the Mamba architecture.

import math
from functools import partial
import json
import os
import re

from collections import namedtuple

import torch
import torch.nn as nn

from based.models.mixers.mamba.modules.mamba_simple import Mamba, Block
from based.models.mixers.mamba.utils.generation import GenerationMixin
from based.models.mixers.mamba.utils.hf import load_config_hf, load_state_dict_hf

try:
    from based.models.mixers.mamba.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from dataclasses import dataclass, field, asdict


@dataclass
class MambaConfig:
    d_model: int = 1024
    n_layer: int = 48
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    reorder_and_upcast_attn: bool = False
    scale_attn_by_inverse_layer_idx: bool = False
    n_positions: int = 2048
    n_embd: int = 1024
    n_head: int = 16
    use_flash_attn: bool = False
    fused_dropout_add_ln: bool = False
    fused_mlp: bool = False
    fused_bias_fc: bool = False
    use_fast_path: bool = True

    def to_dict(self):
        return asdict(self)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    use_fast_path=True,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        use_fast_path=use_fast_path,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        use_fast_path=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    use_fast_path=use_fast_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        if type(input_ids) == list:
            input_ids = input_ids[0]
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            use_fast_path=config.use_fast_path,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained_hf(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)

        # remove the 'model.' prefix from the keys
        state_dict = {re.sub("^model\.", "", k): v for k, v in state_dict.items()}
        # remove Unexpected key(s) in state_dict: "train_metrics.num-tokens.count", "val_metrics.num-tokens.count", "test_metrics.num-tokens.count". from the state_dict
        state_dict = {k: v for k, v in state_dict.items() if "metrics" not in k}

        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)

