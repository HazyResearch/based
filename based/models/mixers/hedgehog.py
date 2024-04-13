import torch
import torch.nn as nn
import copy
from einops import rearrange

import triton

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from based.generation import InferenceParams
from .hedgehog_utils.fwd_tri import parallel_based_fwd_kernel_hedgehog
from .hedgehog_utils.triton_state_update import hedgehog_step

# import fast transformers kernel
try:
    sys.path.append("/var/cr05_data/sim_data/code/release/based/train/")
    from csrc.causal_dot_prod import causal_dot_product
    print(f"Successfully imported the causal dot product kernel! ")
except:
    print(f"Could not import the causal dot product kernel... ")
    causal_dot_product = None


class TiedHeadMLP(nn.Module):
    """
    Use same linear weights applied to all attention heads
    """
    def __init__(self, 
                 num_heads: int,
                 head_dim: int,     # input dim
                 feature_dim: int,  # output dim
                 dtype: torch.dtype,
                 device: torch.device,
                 skip_connection: bool = True,
                 bias: bool = False,
                 zero_init: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.device = device
        self.skip_connection = skip_connection
        self.zero_init = zero_init
        self.init_weights_()
        
        if self.zero_init: 
            self.zero_init_with_skip_() if self.skip_connection else self.zero_init_()

        if self.skip_connection:
            assertion_fail = f'If self.skip_connection we need self.head_dim == self.feature_dim but self.head_dim is {self.head_dim} != self.feature_dim is {self.feature_dim}'
            assert self.head_dim == self.feature_dim, assertion_fail

    def zero_init_with_skip_(self):
        with torch.no_grad():
            nn.init.zeros_(self.layer.weight)

    def zero_init_(self):
        with torch.no_grad():
            nn.init.eye_(self.layer.weight)

    def init_weights_(self):
        self.layer = nn.Linear(self.head_dim, self.feature_dim, bias=False,
                               dtype=self.dtype, device=self.device)

    def forward(self, x: torch.Tensor):
        """Assume x.shape is b h l d"""
        return x + self.layer(x) if self.skip_connection else self.layer(x)


class UntiedHeadMLP(TiedHeadMLP):
    """
    Use different weights per head
    """
    def init_weights_(self):
        self.layer = nn.Conv1d(in_channels=self.head_dim * self.num_heads,
                               out_channels=self.feature_dim * self.num_heads,
                               kernel_size=1, groups=self.num_heads,
                               bias=False, dtype=self.dtype, device=self.device)

    def zero_init_(self):
        with torch.no_grad():
            nn.init.eye_(self.layer.weight[..., 0])

    def _forward(self, x: torch.Tensor):
        b, h, l, d = x.shape
        x = rearrange(x, 'b h l d -> b (h d) l', h=self.num_heads)
        x = self.layer(x)
        x = rearrange(x, 'b (h d) l -> b h l d', h=self.num_heads)
        return x

    def forward(self, x: torch.Tensor):
        """Assume x.shape is b h l d"""
        return x + self._forward(x) if self.skip_connection else self._forward(x)


class UntiedHeadEinsumMLP(UntiedHeadMLP):
    """
    Alternate implementation with untied heads that uses einsum
    """
    def __init__(self, 
                 normal_init: bool = False, 
                 *args: any, **kwargs: any):
        if normal_init:
            self.nn_init_ = self.normal_init_
        else:
            self.nn_init_ = nn.init.kaiming_uniform_
        super().__init__(*args, **kwargs)
    
    def init_weights_(self):
        self.layer = nn.Parameter(torch.zeros(
            (self.num_heads, self.head_dim, self.feature_dim),
            dtype=self.dtype, device=self.device,
        ))
        self.nn_init_(self.layer)

    def normal_init_(self, layer: torch.Tensor):
        with torch.no_grad():
            for i in range(layer.shape[0]):
                nn.init.normal_(layer[i])

    def zero_init_with_skip_(self):
        with torch.no_grad():
            nn.init.zeros_(self.layer)

    def zero_init_(self):
        with torch.no_grad():
            for i in range(self.layer.shape[0]):
                nn.init.eye_(self.layer[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assume x.shape is b h l d"""
        if self.skip_connection:
            return x + torch.einsum('hdf,bhld->bhlf', self.layer, x)  
        o = torch.einsum('hdf,bhld->bhlf', self.layer, x)
        return o


class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """
    def __init__(self, 
                 input_dim: int,                 
                 temp: int = None,
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12, 
                 mlp: nn.Module = None,
                 **kwargs: any):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx     
        self.temp = 1. if temp is None else temp
        self.eps = eps
        self.mlp = mlp if mlp is not None else nn.Identity()
        
    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return self.mlp(x)


class FullSpaceMap(nn.Module):
    """
    Project positive features to upper and lower "halfspaces"
    """
    def __init__(self, 
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12,
                 **kwargs: any):
        super().__init__()
        self.head_dim_idx = head_dim_idx
        self.eps = eps
        
    def forward(self, x: torch.Tensor, fmap = None):
        return torch.cat([x, -x], dim=self.head_dim_idx).clamp(min=self.eps)


class ExpDim(FeatureMap):
    """
    Feature maps based on applying exp() element- or dimension-wise
    """
    def __init__(self, 
                 fullspace: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.fs_map = FullSpaceMap(**kwargs)

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return torch.exp(self.fs_map(x * self.temp))


class SoftmaxDim(ExpDim):
    """
    Compute softmax across fullspace
    """
    def __init__(self, *args: any, **kwargs: any):
        super().__init__(*args, **kwargs)
        self.fs_map = None

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        x = x * self.temp
        return torch.cat([
            torch.softmax( x, dim=self.head_dim_idx),
            torch.softmax(-x, dim=self.head_dim_idx)
        ], dim=self.head_dim_idx).clamp(min=self.eps)


class Hedgehog(nn.Module):
    def __init__(
            self,
            input_dim: int, 
            num_heads: int = 16, 
            head_dim: int = 16,
            feature_dim: int = 128, 
            skip_connection: bool = False, 
            zero_init: bool = False, 
            bias: bool = False, 
            use_triton: bool = False,
            use_fast_transformers: bool = False,
            dtype: torch.dtype = torch.float32,
            layer_idx: int = None,
            **kwargs: any
        ):
        super().__init__()

        self.layer_idx =  layer_idx
        
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = self.input_dim // self.num_heads
        self.feature_dim = feature_dim
        
        self.skip_connection = skip_connection
        self.zero_init = zero_init
        self.bias = bias
        self.dtype = dtype
        self.eps = torch.tensor(1e-12, dtype=self.dtype, device='cuda')
        
        self.use_triton = use_triton
        self.use_fast_transformers = use_fast_transformers

        layer_kwargs = {
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dtype': self.dtype,
            'device': 'cuda',
        }
        kernel_kwargs = {
            'feature_dim': self.feature_dim,
            'skip_connection': self.skip_connection,
            'zero_init': self.zero_init,
            'bias': self.bias,
        }
        feature_map_kwargs = {
            'input_dim': self.input_dim,
            'eps': self.eps,
            'fullspace': True,
        }

        self.learned_kernel = UntiedHeadEinsumMLP(**layer_kwargs, **kernel_kwargs)
        self.feature_map_q = SoftmaxDim(mlp=self.learned_kernel, **feature_map_kwargs)
        self.feature_map_k = copy.deepcopy(self.feature_map_q)
        
        # for triton
        self.BS_k_d = min(128, triton.next_power_of_2(self.head_dim))

        # standard attention projections
        self.proj_q = nn.Linear(self.input_dim, self.head_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.input_dim, self.head_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.input_dim, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.input_dim, bias=False)
    

    def parallel_forward(self, q, k, v):
        # feature map
        q, k = self.feature_map_q(q), self.feature_map_k(k)
            
        # compute causal dot product
        if self.use_triton:  
            print("Triton Prefill")      
            # below is the pytorch equiv of the triton kernel
            # in terms of style of computation 

            # # compute linear attention
            # cumsum_matrix = torch.tril(torch.ones((q.size(2), q.size(2)), device=q.device, dtype=q.dtype))
            # A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) * cumsum_matrix
            # o = torch.einsum("bhnm,bhme->bhne", A_qk, v)

            o = torch.empty_like(v)
            z = torch.empty(q.size(0), q.size(1), q.size(2), dtype=q.dtype, device=q.device)
            
            BS_q_n = 128
            BS_kv_n = 32
            
            BS_k_d = min(128, triton.next_power_of_2(k.shape[-1]))
            BS_v_dv = min(128, triton.next_power_of_2(v.shape[-1]))
            BS_k_d, BS_v_dv = max(BS_k_d, 16), max(BS_v_dv, 16)
            
            D = q.shape[-1] # head_dim
            DV = v.shape[-1]  # feature_dim
            
            NK = triton.cdiv(D, BS_k_d)
            NV = triton.cdiv(DV, BS_v_dv)
            
            num_stages = 2
            num_warps = 4
            
            grid = (NK * NV, triton.cdiv(q.size(2), BS_q_n), q.size(0) * q.size(1))
            
            scale = 1.0
            dt = q.dtype
            parallel_based_fwd_kernel_hedgehog[grid](
                q.to(dt), k.to(dt), v.to(dt), o.to(dt), z.to(dt),
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                q.size(0), q.size(1), q.size(2),
                scale,
                BS_q_n=BS_q_n, BS_kv_n=BS_kv_n, 
                BS_k_d=BS_k_d, 
                BS_v_dv=BS_v_dv, 
                DK=D, 
                DV=DV,
                num_warps=num_warps,
                num_stages=num_stages
            )
            o = o / (z[..., None] + self.eps)
            
        else:
            print(f"PyTorch Prefill")
            # compute linear attention
            if 0: #causal_dot_product is not None and self.use_fast_transformers:
                o = causal_dot_product(
                    q.contiguous().to(dtype=torch.float32), 
                    k.contiguous().to(dtype=torch.float32),
                    v.contiguous().to(dtype=torch.float32),
                )
                z = 1 / (
                    torch.einsum(
                        "bhld,bhld->bhl", 
                        q.to(dtype=torch.float32), 
                        k.to(dtype=torch.float32).cumsum(2)
                    ) + self.eps
                )
                o = o * z[..., None]

            else:
                scale = 1.0

                q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
                o = (q * (k * v).cumsum(dim=2)).sum(dim=-1)
                
                z = (q * k.cumsum(dim=2)).sum(dim=-1) + self.eps
                o = o / z
            
        y = rearrange(o, 'b h l d -> b l (h d)')
        return self.out_proj(y.to(q.dtype))


    # (kv_state, k_state, q, k, v, denom: bool=False)
    def recurrent_forward(self, kv_state, k_state, q, k, v):
        """
        Compute linear attention with recurrent view
        -> Assume q.shape is (b, h, 1, d); k and v.shape are (b, h, l, d)
        """
        b, h, l, d = q.shape
        assert l == 1, f'q.shape is {q.shape} but should be ({b}, {h}, 1, {d})'

        if self.use_triton:
            print(f"Triton recurrent")
            # assert self.use_norm == True, "Triton only supports normalization"
            y = hedgehog_step(
                kv_state.view(-1, kv_state.shape[-2], kv_state.shape[-1]), 
                k_state.view(-1, k_state.shape[-1]),
                q=q.view(-1, q.shape[-1]), 
                k=k.view(-1, k.shape[-1]), 
                v=v.view(-1, v.shape[-1]),
            )
            y = y.view(kv_state.shape[0], kv_state.shape[1], 1, v.shape[-1])
        else:
            print(f"PyTorch recurrent")
            # Expand dims for broadcasting to compute linear attention
            q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
            kv_state += k[:, :, -1:] * v[:, :, -1:]
            k_state  += k[:, :, -1:]

            # Compute linear attention
            num = (q * kv_state).sum(dim=-1)
            y = num / ((q * k_state).sum(dim=-1) + self.eps)

        y = rearrange(y, 'b h l d -> b l (h d)').to(q.dtype)
        o = self.out_proj(y)
        return o


    def forward(
            self,
            hidden_states: torch.Tensor,
            inference_params: InferenceParams = None, 
            *args: any,
            **kwargs: any
        ): 

        # to be replaced by standard attention projections that we'll distill from
        b, l, _ = hidden_states.size()
        q = self.proj_q(hidden_states)
        k = self.proj_k(hidden_states)
        v = self.proj_v(hidden_states)
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        if inference_params is None:
            return self.parallel_forward(q, k, v)
        else:
            # check if we are doing prefill or generation
            if inference_params.seqlen_offset > 0: # recurrent
                kv_state, k_state = self._get_inference_cache(inference_params)
                q, k = self.feature_map_q(q), self.feature_map_k(k)
                return self.recurrent_forward(kv_state, k_state, q, k, v)
            else:
                y = self.parallel_forward(q, k, v)
                q, k = self.feature_map_q(q), self.feature_map_k(k)
                kv_state = torch.einsum("bhnd,bhnf->bhfd", k, v)[:, :, None]
                k_state = k.sum(dim=2)[:, :, None, None]
                if self.layer_idx in inference_params.key_value_memory_dict:
                    # # update the state in-place when graph caching is enabled
                    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(kv_state)
                    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(k_state)
                else: 
                    inference_params.key_value_memory_dict[self.layer_idx] = (kv_state, k_state)
                return y

    
    def expanded_size(self):
        return self.feature_dim * 2 # 2 is due to the FullSpace


    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape ..."""

        kv_shape = (
            batch_size, self.num_heads, 1, self.head_dim, self.expanded_size()
        )
        k_shape = (
            batch_size, self.num_heads, 1, 1, self.expanded_size()
        )
        kv_state = torch.zeros(*kv_shape, dtype=dtype, device=self.out_proj.weight.device)
        k_state = torch.zeros(*k_shape, dtype=dtype, device=self.out_proj.weight.device)
        return (kv_state, k_state)
     

    def _get_inference_cache(self, inference_params: InferenceParams):
        return inference_params.key_value_memory_dict[self.layer_idx]


