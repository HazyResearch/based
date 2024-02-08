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

class ShortConvolution(nn.Module):
    """
    Simple wrapper around nn.Conv1d that accepts dimension last. 
    """

    def __init__(
        self, 
        d_model: int,
        kernel_size: int,
        layer_idx: int=None,
        **kwargs,
    ): 
        super().__init__()
        self.d_model = d_model 
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
            bias=False
        )
    
    def forward(
        self, x: torch.Tensor, inference_params: InferenceParams=None, **kwargs
    ):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        if inference_params is not None:
            state = self._get_state(inference_params.key_value_memory_dict)
            if inference_params.seqlen_offset > 0:
                # check if we are after the first step of inference, step if so
                return self.step(x, state)
            # otherwise, we are at the first step of inference, so we update the state
            k = min(self.kernel_size, x.shape[1])
            state[..., -k: ] = x[:, -k:].transpose(1, 2)
        
        l = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)

        return y 


def init_feature_map(feature_map: str='none', **kwargs: any):
    """
    Initialize query and key mapping for linear attention
    """
    if feature_map in [None, 'none', 'identity']:
        return FeatureMap(**kwargs)
    # Taylor series approximations to exp(x)
    elif feature_map == 'taylor_exp':
        return TaylorExp(**kwargs)
    elif feature_map == 'taylor_exp_3':
        return TaylorExp3(**kwargs) 
    # Generalized quadratic polynomial
    elif feature_map == 'polynomial_2':
        return Polynomial2FeatureMap(**kwargs)
    elif feature_map in "taylor_exp_no_1_q": 
        return TaylorExpNo1Q(**kwargs)
    elif feature_map == 'exp':
        return Exp(**kwargs)
    elif feature_map == 'cosformer':
        return CosFormerFeatureMap(**kwargs)
    elif feature_map == 'performer':
        return PerformerFeatureMap(**kwargs)
    elif feature_map == 'expanded_performer':
        return ExpandedPerformerFeatureMap(**kwargs)
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


class Exp(FeatureMap):
    """
    Spiky feature maps based on applying exp() element- or dimension-wise
    """
    def __init__(self, 
                 fullspace: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor):
        return torch.exp(x)

    
class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(
            self, 
            input_dim: int, 
            scale_dim: Optional[int] = None, 
            mem_save: bool = False,
            **kwargs: any
        ):
        super().__init__(input_dim, **kwargs)
        self.mem_save = mem_save
        self.r2  = math.sqrt(2)
        if scale_dim is None:
            scale_dim = self.input_dim
        self.rd  = math.sqrt(scale_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):
        if self.mem_save:
            return self.forward_mem_save(x)
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        # SE: raising to power 0 is a hacky way to get ones without calling torch.ones
        # which is incompatible with cuda graph caching 
        return torch.cat([x[..., :1] ** 0, 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)
    
    def forward_mem_save(self, x: torch.Tensor):
        """
        Compute f(x) s.t. f(x)^T f(x') = 1 + x^Tx' + (x^Tx')^2 / 2
        -> Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        # Slow but memory-saving way to compute 2nd-order terms; how do w/o outer-product first?
        x2  = oe.contract('...m,...n->...mn', x, x) / self.rd
        x2d = torch.diagonal(x2, dim1=-2, dim2=-1) / self.r2
        x2  = x2[..., self.tril_indices[0], self.tril_indices[1]] 
        x   = torch.cat([x[..., :1] ** 0,  
                         x / self.rrd, x2d, x2], dim=-1)
        return x 


class TaylorExpNo1Q(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(
            self, 
            input_dim: int, 
            scale_dim: Optional[int] = None, 
            mem_save: bool = False,
            **kwargs: any
        ):
        super().__init__(input_dim, **kwargs)
        self.mem_save = mem_save
        self.r2  = math.sqrt(2)
        if scale_dim is None:
            scale_dim = self.input_dim
        self.rd  = math.sqrt(scale_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return x2 / self.rd
        

class TaylorExp3(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, **kwargs: any):
        super().__init__(input_dim, **kwargs)
        self.r3 = math.sqrt(8)
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(self.input_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        # Get 3rd-order terms (rearrange(x * x * x), '... m n -> ... (m n)')
        x3 = (x2.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r3
        # SE: raising to power 0 is a hacky way to get ones without calling torch.ones
        # which is incompatible with cuda graph caching 
        return torch.cat([x[..., :1] ** 0, 
                          x / self.rrd, x2 / self.rd, x3 / (self.rd * self.rd)], dim=self.head_dim_idx)
        

class Polynomial2FeatureMap(FeatureMap):
    """
    Generalization of Taylor exp. feature map with alternative polynomial coefficients
    
    To recover TaylorExp, we can set:
    scale_dim = None, scale_norm = True, c0 = 1, c1 = 1, c2 = 0.5
    Though may be better to deviate from results here: https://api.wandb.ai/links/hazy-research/tckn8ibj
    e.g., scale_dim = 1, scale_norm = False, c0 = 0.5, c1 = 1, c2 = 0.5
    """
    def __init__(self, 
                 input_dim: int, 
                 scale_dim: Optional[int] = None,
                 scale_norm: Optional[bool] = False,
                 c0: float = 1, c1: float = 1, c2: float = 0.5,
                 **kwargs: any):
        super().__init__(input_dim=input_dim, **kwargs)
        self.r2  = math.sqrt(2)
        if scale_dim is None:
            scale_dim = self.input_dim
        self.scale_norm = scale_norm
        self.rd  = math.sqrt(scale_dim)
        self.rrd = math.sqrt(self.rd)
        self.c0, self.c1, self.c2 = math.sqrt(c0), math.sqrt(c1), math.sqrt(c2)

    def forward(self, x: torch.Tensor):
        x = x / self.rrd
        if self.scale_norm:
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        # 0th and 2nd order terms
        x0 = self.c0 * torch.ones(x[..., :1].shape, dtype=x.dtype, device=x.device)
        x2 = self.c2 * (x.unsqueeze(-2) * x.unsqueeze(-1)).flatten(start_dim=-2)
        return torch.cat([x0, self.c1 * x, x2], dim=-1)



class CosFormerFeatureMap(FeatureMap):
    """
    Code from https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
    """
    MAX_SEQ_LEN = 4096
    def __init__(self, act_fun = 'relu', 
                 *args: any, **kwargs: any):
        super().__init__(*args, **kwargs)
        self.n_heads = None
        self.head_dim = None
        self.act_fun = self.get_act_fun(act_fun=act_fun)

        index = np.pi / 2 * torch.arange(1, self.MAX_SEQ_LEN + 1).reshape(1, -1, 1)
        self.register_buffer('index', index)
        
    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)
    
    def get_act_fun(self, act_fun = 'relu'):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
        
    def forward(self, x: torch.Tensor):
        b, h, l, d = x.shape
        if self.head_dim is None:
            self.head_dim = d
        if self.n_heads is None:
            self.n_heads = h
        
        x = self.act_fun(x)
        
        x = rearrange(x, 'b h l d -> (b h) l d')

        # cos transform
        m = x.shape[1]
        tgt_len = m
        
        weight_index = self.index
        x = torch.cat([x * torch.sin(weight_index[:, :tgt_len, :] / m), 
                       x * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        return rearrange(x, '(b h) l d -> b h l d', b=b)

    def expanded_size(self):
        return 2 * self.input_dim



class PerformerFeatureMap(FeatureMap):
    """
    Code from https://github.com/teddykoker/performer/blob/main/performer.py
    """
    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_feats = None
        
    def forward(self, x: torch.Tensor):
        """Assume x.shape is (b, h, l, d)"""
        d = x.shape[-1]
        
        # Build + cache random features
        if self.random_feats is None:
            m = d  # self.output_dim   # * d
            self.random_feats = self.orthogonal_gaussian(m, d).type(x.type())
            # print(self.random_feats.shape, self.random_feats)
            
        fmap = self.feature_map(self.h, [torch.exp], self.random_feats)
        return fmap(x / (d ** 0.25))
    
    def feature_map(self, h, fs, random_feats):
        """Random feature map"""
        m, d = random_feats.shape
        # return lambda x: (self.h(x) / math.sqrt(m) * torch.cat(
        #     [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
        #     dim=-1))
        return lambda x: (1 / math.sqrt(m) * torch.cat(
            [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
            dim=-1))
    
    def h(self, x):
        """Adjust to get softmax (from RBF)"""
        x = torch.exp(-torch.pow(x, 2).sum(dim=-1, keepdims=True) / 2)
        return x
    
    def iid_gaussian(self, m, d):
        """Generate IID Gaussian random features"""
        return torch.randn(size=(m, d))

    def orthogonal_gaussian(self, m, d):
        """Generate orthogonal Gaussian random features"""
        def orthogonal_square():
            # create orthogonal square matrix using Gram-Schmidt
            q, _ = np.linalg.qr(self.iid_gaussian(d, d))
            return q.T

        num_squares = int(m / d)
        blocks = [orthogonal_square() for _ in range(num_squares)]

        remainder = m - d * num_squares
        if remainder:
            blocks.append(orthogonal_square()[:remainder])

        matrix = np.vstack(blocks)
        matrix /= np.sqrt(num_squares + remainder / d)
        # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

        return torch.from_numpy(matrix) 

    def expanded_size(self):
        return self.input_dim 


class ExpandedPerformerFeatureMap(FeatureMap):
    """
    Code from https://github.com/teddykoker/performer/blob/main/performer.py
    """
    def __init__(
        self, 
        input_dim: int = 16, 
        expanded_dim: int = 256,
        *args, 
        **kwargs
    ):
        super().__init__(input_dim=input_dim, *args, **kwargs)
        self.expanded_dim = expanded_dim
        print(f"using expanded dim {expanded_dim}!!")
        
        random_feats = self.orthogonal_gaussian(
            self.expanded_dim, 
            self.input_dim
        )
        self.register_buffer("random_feats", random_feats)
        
    def forward(self, x: torch.Tensor):
        """Assume x.shape is (b, h, l, d)"""
        d = x.shape[-1]
            
        fmap = self.feature_map(self.h, [torch.exp], self.random_feats)
        return fmap(x / (d ** 0.25))
    
    def feature_map(self, h, fs, random_feats):
        """Random feature map"""
        m, d = random_feats.shape
        # MZ 1/30/24 -> add back scaling factor to get SM from RBF
        return lambda x: (self.h(x) / math.sqrt(m) * torch.cat(
            [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
            dim=-1))
        # return lambda x: (1 / math.sqrt(m) * torch.cat(
        #     [f(torch.einsum("...d,md->...m", x, random_feats)) for f in fs],
        #     dim=-1))
    
    def h(self, x):
        """Adjust to get softmax (from RBF)"""
        x = torch.exp(-torch.pow(x, 2).sum(dim=-1, keepdims=True) / 2)
        return x
    
    def iid_gaussian(self, m, d):
        """Generate IID Gaussian random features"""
        return torch.randn(size=(m, d))

    def orthogonal_gaussian(self, m, d):
        """Generate orthogonal Gaussian random features"""
        def orthogonal_square():
            # create orthogonal square matrix using Gram-Schmidt
            q, _ = np.linalg.qr(self.iid_gaussian(d, d))
            return q.T

        num_squares = int(m / d)
        blocks = [orthogonal_square() for _ in range(num_squares)]

        remainder = m - d * num_squares
        if remainder:
            blocks.append(orthogonal_square()[:remainder])

        matrix = np.vstack(blocks)
        matrix /= np.sqrt(num_squares + remainder / d)
        # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

        return torch.from_numpy(matrix) 


    def expanded_size(self):
        return self.expanded_dim

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
        mem_save: bool = False,
        c0: float = 0.5,
        c1: float = 1,
        c2: float = 0.5,
        eps: float = 1e-12,
        use_act: bool = False,
        use_rotary_apply: bool = False,
        fa_rotary: bool = False,
        init_weights: bool = True,
        rope_theta: int = 10000.0,
        expand_rotary: int = 1,
        attention_dropout: nn.Module = nn.Dropout(0.0),
        train_view: str = "linear",
        device: torch.device = None,
        dtype: torch.dtype = None,
        layer_idx: int = None,
        use_proj_g: bool = False,
        linear_attn_init: bool = False,
        use_output_gating: bool = False,
        use_decay_proj: bool = False,
        use_decay_proj_init_1: bool = False,
        attn_fp32: bool = False,
        use_exp_combo: bool = False,
        use_positive: bool = False,
        initial_conv: bool = False,
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
        self.mem_save = mem_save
        if self.use_act:
            self.act = nn.SiLU()

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

        if self.feature_name in ['taylor_exp', 'taylor_exp_3', 'exp']:
            feature_map_kwargs = {
                'input_dim': self.feature_dim,
                'head_dim_idx': -1,
                'temp': 1.,
                'eps': 1e-12,
                "mem_save": mem_save,
                "scale_dim": scale_dim,
                "no_dim_norm": False,
                "halfspace": False,
                "scale_fullspace": False,
            }
        elif self.feature_name == 'polynomial_2':
            feature_map_kwargs = {
                'input_dim': self.feature_dim,
                'head_dim_idx': -1,
                'eps': 1e-12,
                'scale_dim': scale_dim,
                'scale_norm': False,
                'c0': c0,
                'c1': c1,
                'c2': c2,
            }
        elif self.feature_name == "expanded_performer":
            feature_map_kwargs = {
                'input_dim': self.feature_dim,
                'expanded_dim': feature_expanded_dim,
            }
        else:
            feature_map_kwargs = {
                'input_dim': self.feature_dim,
            }

        self.initial_conv = initial_conv
        if self.initial_conv:
            # additions
            self.input_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.conv = ShortConvolution(d_model, kernel_size=4, layer_idx=(layer_idx, "conv1"))

        self.feature_map = init_feature_map(feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        self.use_decay_proj = use_decay_proj
        self.use_decay_proj_init_1 = use_decay_proj_init_1
        self.attn_fp32 = attn_fp32
        self.use_exp_combo = use_exp_combo
        self.use_positive = use_positive
        if self.use_decay_proj:
            # we have an n x n matrix of decay values - lets make it input-dependent
            self.decay_proj = nn.Linear(self.d_model, self.num_heads, bias=False)
            if self.use_decay_proj_init_1:
                nn.init.constant_(self.decay_proj.weight, 1)
        
        self.use_output_gating = use_output_gating
        if self.use_output_gating:
            # we have an n x n matrix of decay values - lets make it input-dependent
            self.output_gate_projection = nn.Linear(self.d_model, self.num_heads, bias=False)

        self.use_proj_g = use_proj_g
        if self.use_proj_g:
            self.proj_g = nn.Linear(self.d_model, self.d_model, bias=False) 

        self.dropout = nn.Identity()
        self.eps = eps

        if linear_attn_init:
            self.init_proj()

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


    def init_proj(self):
        print("init")
        import time
        time.sleep(1)
        nn.init.xavier_normal_(self.proj_q.weight)
        if self.proj_q.bias is not None:
            nn.init.constant_(self.proj_q.bias, 0)
        nn.init.xavier_normal_(self.proj_k.weight)
        if self.proj_k.bias is not None:
            nn.init.constant_(self.proj_k.bias, 0)
        nn.init.xavier_normal_(self.proj_v.weight)
        if self.proj_v.bias is not None:
            nn.init.constant_(self.proj_v.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)


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

            # SE (01/30): For generation we want can't pass kv_seq_len here and needs to
            # pass self.l_max to avoid an out of range index
            # cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            cos, sin = self.rotary_emb(v, seq_len=self.l_max)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # float(), seqlen larger, dims choices.

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

            if self.initial_conv:
                u = self.conv(hidden_states)
                u = nn.functional.silu(u)
                hidden_states = u * self.input_proj(hidden_states)
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
            # TODO(SE): no support for recurrent view with the kernel
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

            if self.attn_fp32:
                A_qk = torch.einsum("bhnd,bhmd->bhnm", q.to(torch.float32), k.to(torch.float32)).to(torch.float32)
                if self.use_decay_proj:
                    dt_out = self.decay_proj(hidden_states) # (b, l, h)
                    if self.use_exp_combo:
                        decay = torch.exp(dt_out.transpose(1,2).unsqueeze(-1) * decay.unsqueeze(0))    # (b, h, l, 1) * (1, h, l, l) -> (b, h, l, l)
                    else:
                        decay = dt_out.transpose(1,2).unsqueeze(-1) * decay.unsqueeze(0)    # (b, h, l, 1) * (1, h, l, l) -> (b, h, l, l)

                if decay is not None:
                    if len(decay.shape) == 3:
                        decay = decay.unsqueeze(0)
                    A_qk = A_qk.to(torch.float32) * decay.to(torch.float32)
                else:
                    A_qk = A_qk * cumsum_matrix
                out = torch.einsum("bhnm,bhme->bhne", A_qk.to(torch.float32), v.to(torch.float32)).to(torch.float32)
            else:
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

                if self.use_output_gating:
                    # SE: this output gating has the same affect as the decay_proj, but can 
                    # be used without the decay matrix. It projects the hidden states to one
                    # scalar per head. That head is then gated with the scalar. 
                    g = self.output_gate_projection(hidden_states) # (b, l, h)
                    out = torch.einsum("bhld,blh->bhld", out, g)

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
        if self.use_proj_g:
            y = y * nn.functional.silu(self.proj_g(hidden_states))
        
        if self.use_act:
            y = self.act(self.out_proj(y.to(hidden_states.dtype)))
        else:
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
        if self.mem_save:
            return self.feature_dim * (self.feature_dim + 1) // 2 + self.feature_dim + 1
        else:
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


class LinearAttentionSelective(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 16,
        num_heads: int = 16,
        feature_name: str = "taylor_exp",
        c0: float = 0.5,
        c1: float = 1,
        c2: float = 0.5,
        eps: float = 1e-12,
        causal: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        n_lookups: int = None,
        selection_noise: float = 0.1,
        selection_penalty: float = 0.1,
        training: bool = False,
        **kwargs
    ):
        super().__init__()

        # print(f"Linear attention:")
        # print(f"-- {feature_dim=}")
        # print(f"-- {feature_name=} -- {c0=} -- {c1=} -- {c2=}")
        # print(f"-- {num_key_value_heads=}")
        # print(f"-- {num_heads=}")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.l_max = l_max

        # linear attention 
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.causal=causal

        feature_map_kwargs = {
            "input_dim": self.feature_dim,
            "head_dim_idx": -1,
            "temp": 1.,
            "eps": 1e-12,
            "scale_dim": False,
            "no_dim_norm": False,
            "halfspace": False,
            "scale_fullspace": False,
        }

        self.feature_map = init_feature_map(feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.eps = eps

        # selection
        self.selecting = nn.Linear(d_model, 1, bias=False)
        self.n_lookups = n_lookups


    def forward(self, 
            hidden_states: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            use_cache: bool = False,
            *args: any, 
            **kwargs: any
        ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        b, l, d = hidden_states.size()

        selection = torch.sigmoid(self.selecting(hidden_states))
        n_lookups = math.sqrt(l) if self.n_lookups is None else self.n_lookups
        out = torch.topk(selection, k=math.ceil(n_lookups), dim=1, sorted=False)
        src = torch.gather(hidden_states, dim=1, index=out.indices.repeat(1, 1, d))
        base = torch.zeros_like(hidden_states)
        hidden_states = base.scatter_add(dim=1, index=out.indices.repeat(1, 1, d), src=src)

        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        if causal_dot_product is None:
            q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        
        # Compute attention
        if self.causal and causal_dot_product is not None:
            v = causal_dot_product(
                q.contiguous().to(dtype=torch.float32), 
                k.contiguous().to(dtype=torch.float32),
                v.contiguous().to(dtype=torch.float32),
            )
            z = 1 / (
                torch.einsum(
                    "bhld,bhld->bhl", 
                    q.to(dtype=torch.float32), 
                    k.to(dtype=torch.float32).cumsum(2)
                ) + self.eps)
            y = v * z[..., None]
            y = y.to(hidden_states.dtype)
        elif self.causal and causal_dot_product is None:
            y = ((q * (k * v).cumsum(dim=2)).sum(dim=-1) / 
                    ((q * k.cumsum(dim=2)).sum(dim=-1) + self.eps))
        else:
            y = ((q * (k * v).sum(dim=2, keepdim=True)).sum(dim=-1) /
                    ((q * k.sum(dim=2, keepdim=True)).sum(dim=-1) + self.eps))
        
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.out_proj(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)
