import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List

from flash_attn.utils.generation import InferenceParams
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    causal_conv1d_fn = None
    causal_conv1d_update = None


class ShortConvolution(nn.Module):
    """
    Simple wrapper around nn.Conv1d that accepts dimension last. 
    """

    def __init__(
        self, 
        d_model: int,
        kernel_size: int,
        layer_idx: int=None,
        use_cuda: bool=False,
        conv_bias: bool=False,
        **kwargs,
    ): 
        super().__init__()
        self.d_model = d_model 
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.use_cuda = use_cuda and causal_conv1d_fn is not None
        if self.use_cuda:
            conv_bias = True
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
            bias=conv_bias
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
        b, l, d = x.shape
        state = None
        if inference_params is not None:
            self._set_state(inference_params)
            state = self._get_state(inference_params)
            if inference_params.seqlen_offset > 0:
                # check if we are after the first step of inference, step if so
                return self.step(x, state)
            # otherwise, we are at the first step of inference, so we update the state
            k = min(self.kernel_size, x.shape[1])
            state[..., -k: ] = x[:, -k:].transpose(1, 2)

        if self.use_cuda:
            if state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                state.copy_(F.pad(x, (self.kernel_size - x.shape[-1], 0)))  # Update state (B D W)
            y = causal_conv1d_fn(
                x=x.transpose(1,2),
                weight=rearrange(self.conv.weight, "d 1 w -> d w"),
                bias=self.conv.bias,
                activation="silu",
            ).transpose(1, 2)
        else:
            y = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)
        return y 

    def step(self, x: torch.Tensor, state: torch.Tensor):
        if self.use_cuda:
            # Conv step
            if causal_conv1d_update is None:
                state.copy_(torch.roll(state, shifts=-1, dims=-1))  # Update state (B D W)
                state[:, :, -1] = x
                x = torch.sum(state * rearrange(self.conv.weight, "d 1 w -> d w"), dim=-1)  # (B D)
                if self.conv.bias is not None:
                    x = x + self.conv1d.bias
                x = self.act(x).to(dtype=x.dtype)
            else:
                x = causal_conv1d_update(
                    x.squeeze(),
                    state,
                    rearrange(self.conv.weight, "d 1 w -> d w"),
                    self.conv.bias,
                    "silu",
                ).unsqueeze(1)
            return x
        else:
            state.copy_(torch.roll(state, shifts=-1, dims=-1))  # Update state (B D W)
            state[:, :, -1] = x.squeeze(1)
            x = torch.einsum("bdk,dgk->bd", state, self.conv.weight).to(x.dtype)
            return x.unsqueeze(1)
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape (b, d, k)"""
        return torch.zeros(
            batch_size, 
            self.d_model, 
            self.kernel_size, 
            device=self.conv.weight.device, 
            dtype=self.conv.weight.dtype if dtype is None else dtype
        )
    
    def _set_state(
            self, 
            inference_params: InferenceParams, 
            layer_idx: int=None,
            state: dict = None
        ):
        """Sets the state tensors for the given layer.
        If state is already set, does nothing.
        """
        if state is None:
            state = inference_params.key_value_memory_dict
        if layer_idx is None:
            layer_idx = self.layer_idx

        if isinstance(layer_idx, (int, str)):
            if layer_idx not in state:
                state[layer_idx] = self.allocate_inference_cache(
                    batch_size=inference_params.max_batch_size,
                    max_seqlen=inference_params.max_seqlen,
                )     
        else:
            if layer_idx[0] not in state:
                state[layer_idx[0]] = {}
            self._set_state(
                inference_params, 
                layer_idx[1],
                state[layer_idx[0]]
            )

    def _get_state(self, inference_params: InferenceParams, layer_idx: int=None):
        """Returns the state tensors for the given layer.
        Adds support for nested states. 
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        if isinstance(layer_idx, (int, str)):
            return inference_params.key_value_memory_dict[layer_idx]
        else:
            return self._get_state(inference_params, layer_idx[0])[layer_idx[1]]
        

class BaseConvWithSiLU4(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        kernel_sizes: List[int]=[3],
        layer_idx: int=None,
        implicit_long_conv: bool=True,
        use_bias=True,
        expand_proj: int=2,
        use_cuda: bool=False,
        **kwargs
    ):
        super().__init__()
        # kernel_sizes = [3]
        
        self.d_model = d_model
        self.l_max = l_max
        self.layer_idx=layer_idx

        self.d_inner = expand_proj*self.d_model // 2
        self.in_proj = nn.Linear(self.d_model,  expand_proj*self.d_model, bias=use_bias)
        self.out_proj = nn.Linear(self.d_inner,  self.d_model, bias=use_bias)

        self.use_cuda = use_cuda and causal_conv1d_fn is not None

        # prepare convolution
        convs = []
        for kernel_size in kernel_sizes:
            convs.append(ShortConvolution(self.d_inner, kernel_size=kernel_size, use_cuda=self.use_cuda, layer_idx=(layer_idx, "conv1")))
        self.conv1 = convs[0]

    def forward(self, u, position_ids=None, inference_params: InferenceParams=None, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        u = self.in_proj(u)
        u1, u2 = torch.split(u, self.d_inner, dim=-1)
        u_conv1 = self.conv1(u1, inference_params=inference_params)
        if not self.use_cuda:
            # SA: the silu is fused in the cuda version.
            u_conv1 = nn.functional.silu(u_conv1)
        v = u_conv1 * u2
        y = self.out_proj(v)
        return y 

        
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape (b, d, k)"""
        return {
            "conv1": self.conv1.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs),
        }
    
