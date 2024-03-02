"""
Linear attention in Based. 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe
from einops import rearrange
from typing import Optional, Tuple
import numpy as np

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaRotaryEmbedding

from based.generation import InferenceParams

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    # fast transformers linear attention cuda kernel
    from based.csrc.causal_dot_prod import causal_dot_product  
    print(f"Succesfully imported the causal dot product kernel... ")
except:
    print(f"Could not impfort the causal dot product kernel... ")
    causal_dot_product = None

def init_feature_map(feature_map: str='none', **kwargs: any):
    """
    Initialize query and key mapping for linear attention
    """
    if feature_map in [None, 'none', 'identity']:
        return FeatureMap(**kwargs)
    # Taylor series approximations to exp(x)
    elif feature_map == 'taylor_exp':
        return TaylorExp(**kwargs)
    else:
        raise NotImplementedError(f'Sorry "{feature_map}" feature map not implemented.')
        
        
class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """
    def __init__(self, 
                 input_dim: int,                 
                 temp: int = None,
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12, 
                 **kwargs: any):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx     
        self.temp = 1. if temp is None else temp
        self.eps = eps
        
    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x
    
class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(
            self, 
            input_dim: int, 
            scale_dim: Optional[int] = None, 
            **kwargs: any
        ):
        super().__init__(input_dim, **kwargs)
        self.r2  = math.sqrt(2)
        if scale_dim is None:
            scale_dim = self.input_dim
        self.rd  = math.sqrt(scale_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        # SE: raising to power 0 is a hacky way to get ones without calling torch.ones
        # which is incompatible with cuda graph caching 
        return torch.cat([x[..., :1] ** 0, 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)



class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        head_dim: int = None, # when None, head_dim = d_model / num_key_value_heads
        num_key_value_heads: int = 16,
        num_heads: int = 16,
        feature_name: str = "taylor_exp",
        eps: float = 1e-12,
        use_act: bool = False,
        use_rotary_apply: bool = False,
        fa_rotary: bool = False,
        rope_theta: int = 10000.0,
        expand_rotary: int = 1,
        attention_dropout: nn.Module = nn.Dropout(0.0),
        train_view: str = "linear",
        device: torch.device = None,
        dtype: torch.dtype = None,
        layer_idx: int = None,
        use_decay_proj: bool = False,
        scale_dim: Optional[int] = None,
        feature_expanded_dim: int = 256,
        **kwargs
    ):
        super().__init__()

        print(f"Linear attention:")
        print(f"-- {feature_dim=}")
        print(f"-- {feature_name=} -- {c0=} -- {c1=} -- {c2=}")
        print(f"-- {num_key_value_heads=}")
        print(f"-- {num_heads=}")
        print(f"-- {use_rotary_apply=}")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.train_view = train_view
        self.d_model = d_model
        self.l_max = l_max
        self.use_act = use_act

        # linear attention 
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        if head_dim is None:
            self.head_dim = self.d_model // self.num_key_value_heads
        else:
            self.head_dim = head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        feature_map_kwargs = {
            'input_dim': self.feature_dim,
            'head_dim_idx': -1,
            'temp': 1.,
            'eps': 1e-12,
            "scale_dim": scale_dim,
            "no_dim_norm": False,
            "halfspace": False,
            "scale_fullspace": False,
        }

        self.feature_map = init_feature_map(feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        self.use_decay_proj = use_decay_proj
        if self.use_decay_proj:
            # we have an n x n matrix of decay values - lets make it input-dependent
            self.decay_proj = nn.Linear(self.d_model, self.num_heads, bias=False)

        self.dropout = nn.Identity()
        self.eps = eps

        # parameters
        self.use_rotary_apply = use_rotary_apply
        if self.use_rotary_apply:
            self.expand_rotary = expand_rotary
            self.rope_theta = rope_theta
            self.dropout = attention_dropout
            self.scale = 1 / self.head_dim ** 0.5  # For scaled dot-prod attention

            self.q_shape = [self.num_heads, self.feature_dim]
            self.k_shape = [self.num_key_value_heads, self.feature_dim]
            self.v_shape = [self.num_key_value_heads, self.head_dim]
            self.rotary_emb = LlamaRotaryEmbedding(
                self.feature_dim,
                max_position_embeddings=self.l_max,
                base=self.rope_theta,
            )

    def process_qkv(self, 
                    hidden_states: torch.Tensor,
                    position_ids: Optional[torch.LongTensor] = None,
                    use_cache: bool = False,
                    apply_rotary: bool = True,):
        """
        Get Q, K, V tensors from hidden_states, e.g., by applying projections, 
        positional embeddings, KV cache
        -> Follow the original LlamaAttention API
        """
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        
        # Following HF Llama source code to get (b, h, l, d)
        q = q.view(b, l, *self.q_shape).transpose(1, 2)
        k = k.view(b, l, *self.k_shape).transpose(1, 2)
        v = v.view(b, l, *self.v_shape).transpose(1, 2)
        kv_seq_len = k.shape[-2] * self.expand_rotary
            
        # Apply rotary embeddings
        if apply_rotary:
            if position_ids is None:
                position_ids = torch.arange(
                    kv_seq_len, dtype=torch.long, device=hidden_states.device
                )
                position_ids = position_ids.unsqueeze(0).expand((b, kv_seq_len))
            cos, sin = self.rotary_emb(v, seq_len=self.l_max)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return q, k, v, kv_seq_len


    def forward(self, 
            hidden_states: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            inference_params: InferenceParams = None,
            use_cache: bool = False,
            decay: Optional[Tuple[torch.Tensor]] = None,
            *args: any, 
            **kwargs: any
        ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        decay, decay_recurrent = decay if decay is not None else (None, None)

        if self.use_rotary_apply:
            b, l, d = hidden_states.shape
            assert d == self.d_model, f'Hidden_states.shape should be size {(b, l, d)} but is shape {hidden_states.shape}'
            q, k, v, kv_seq_len = self.process_qkv(
                hidden_states, position_ids, use_cache
            )
        else:
            b, l, _ = hidden_states.size()
            q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
            q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
            k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
            v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        
        # Compute attention
        if inference_params is not None and inference_params.seqlen_offset > 0:
            # check if we are after the first step of inference, step if so
            kv_state, k_state = self._get_inference_cache(inference_params)
            return self.step(hidden_states, kv_state, k_state, q, k, v, decay=decay_recurrent)
        
        if self.train_view == "linear" and inference_params is None:
            # Note: No support for recurrent view with the kernel
            if decay is not None:
                raise NotImplementedError("Decay not implemented for linear train view")

            if causal_dot_product is not None:
                v = causal_dot_product(q.contiguous().to(dtype=torch.float32), k.contiguous().to(dtype=torch.float32),v.contiguous().to(dtype=torch.float32),)
                z = 1 / (
                    torch.einsum(
                        "bhld,bhld->bhl", 
                        q.to(dtype=torch.float32), 
                        k.to(dtype=torch.float32).cumsum(2)
                    ) + self.eps)
                y = v * z[..., None]
                y = y.to(hidden_states.dtype)
            else: 
                q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
                kv_state = (k * v).cumsum(dim=2)  # causal 
                k_state = k.cumsum(dim=2)
                y = ((q * kv_state).sum(dim=-1) / 
                        ((q * k_state).sum(dim=-1) + self.eps))
                
                if inference_params is not None:
                    kv_state, k_state = kv_state[:, :, -1:], k_state[:, :, -1:]  # only need last position
                    if self.layer_idx in inference_params.key_value_memory_dict:
                        # update the state in-place when graph caching is enabled
                        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(kv_state)
                        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(k_state)
                    else: 
                        inference_params.key_value_memory_dict[self.layer_idx] = (kv_state, k_state)

        elif self.train_view == "quadratic" or inference_params is not None:
            # this is just a regular attention implementation without the softmax
            # https://github.com/microsoft/torchscale/blob/d51f10354d57e67be82dc660505f18322e82d4af/torchscale/component/multiscale_retention.py#L76
            # https://github.com/microsoft/torchscale/blob/d51f10354d57e67be82dc660505f18322e82d4af/torchscale/architecture/retnet.py#L62
            # OOMs

            cumsum_matrix = torch.tril(torch.ones((l, l))).to(q.device, q.dtype)
            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
            if decay is not None:
                decay = decay[:, :l, :l]
                if len(decay.shape) == 3:
                    decay = decay.unsqueeze(0)
                if self.use_decay_proj:
                    dt_out = self.decay_proj(hidden_states) # (b l d) --> (b, l, h)
                    assert decay.shape[2] >= l, f"decay matrix {decay.shape} to short for sequence length {l}"
                    decay_mat = dt_out.transpose(1,2).unsqueeze(-1) * decay   # (b, h, l, 1) * (1, h, l, l)
                elif decay is not None:
                    decay_mat = decay
                A_qk = A_qk * decay_mat
            else:
                A_qk = A_qk * cumsum_matrix
                
            out = torch.einsum("bhnm,bhme->bhne", A_qk.to(hidden_states.dtype), v.to(hidden_states.dtype))

            # denom
            z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
            y = out * z[..., None]
            y = y.to(hidden_states.dtype)

            if inference_params is not None:
                if decay is not None:
                    k_decay = decay[:, :, l - 1 , :l, None] * k
                    kv_state = torch.einsum("bhnd,bhnf->bhfd", k_decay, v)[:, :, None]
                else:
                    kv_state = torch.einsum("bhnd,bhnf->bhfd", k, v)[:, :, None]
                k_state = k.sum(dim=2)[:, :, None, None]
                if self.layer_idx in inference_params.key_value_memory_dict:
                    # # update the state in-place when grapy.h caching is enabled
                    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(kv_state)
                    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(k_state)
                else: 
                    inference_params.key_value_memory_dict[self.layer_idx] = (kv_state, k_state)

        else:
            raise NotImplementedError(f"Train view {self.train_view} not implemented")
          
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.out_proj(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)

    
    def step(self, hidden_states: torch.Tensor, kv_state: torch.Tensor, k_state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor=None):
        """
        Compute linear attention with recurrent view
        -> Assume q.shape is (b, h, 1, d); k and v.shape are (b, h, l, d)
        """
        b, h, l, d = q.shape
        assert l == 1, f'q.shape is {q.shape} but should be ({b}, {h}, 1, {d})'
        # Expand dims for broadcasting to compute linear attention
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        if decay is not None: 
            kv_state.copy_(torch.einsum("h,bhldf->bhldf", decay, kv_state) + k[:, :, -1:] * v[:, :, -1:])
        else:
            kv_state += k[:, :, -1:] * v[:, :, -1:]
        k_state  += k[:, :, -1:]

        # Compute linear attention
        num = (q * kv_state).sum(dim=-1)

        if self.use_decay_proj:
            dt_out = self.decay_proj(hidden_states).squeeze(1)
            num = num * dt_out[..., None, None]


        y = num / ((q * k_state).sum(dim=-1) + self.eps)


        y = rearrange(y, 'b h l d -> b l (h d)').to(q.dtype)
        return self.dropout(self.out_proj(y))
 
    
    def expanded_size(self):
        return self.feature_dim ** 2 + self.feature_dim + 1

    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape ..."""

        kv_shape = (
            batch_size, self.num_key_value_heads, 1, self.head_dim, self.expanded_size()
        )
        k_shape = (
            batch_size, self.num_key_value_heads, 1, 1, self.expanded_size()
        )
        kv_state = torch.zeros(*kv_shape, dtype=dtype, device=self.out_proj.weight.device)
        k_state = torch.zeros(*k_shape, dtype=dtype, device=self.out_proj.weight.device)
        return (kv_state, k_state)


    def _get_inference_cache(self, inference_params: InferenceParams):
        return inference_params.key_value_memory_dict[self.layer_idx]
