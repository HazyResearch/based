"""
Linear attention in Based. 
"""
import math

import torch
import torch.nn as nn
from einops import rearrange

from based.generation import InferenceParams

import sys
sys.path.append("../../../")

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        head_dim: int = None,
        num_heads: int = 16,
        layer_idx: int = None,
        seg_len: int = 64, 
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        **kwargs
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.d_model   = d_model
        self.l_max     = l_max

        # set dimension 
        self.num_heads = num_heads
        self.head_dim  = self.d_model // self.num_heads if head_dim is None else head_dim      
        
        self.seg_len = seg_len
        self.noise_eps = 1e-5

        # initialize projections and feature map
        self.proj_q   = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj_k   = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj_v   = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # initialize Cylon-style mixing parameters
        self.terrace_mix = nn.Parameter(torch.ones(self.seg_len) * 0.5)
        self.index_decays = nn.Parameter(
            -0.1 * torch.ones(self.num_heads, self.seg_len) + 
            torch.randn(self.num_heads, self.seg_len) * 0.01
        )
        self.mega_decays = nn.Parameter(
            -0.1 * torch.ones(self.num_heads) + 
            torch.randn(self.num_heads) * 0.01
        )
        
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.rotary_emb_dim = rotary_emb_dim
        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved
            )
            
    def _apply_rotary(self, qkv, seqlen_offset=0, max_seqlen=None):
        if self.rotary_emb_dim == 0:
            return qkv
        
        if max_seqlen is not None:
            self.rotary_emb._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        
        return self.rotary_emb(qkv, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)
        
    def forward(self, 
        hidden_states: torch.Tensor,
        inference_params: InferenceParams = None,
        *args: any, 
        **kwargs: any
    ):
        """
        x (torch.Tensor): tensor of shape (b, l, d)
        y (torch.Tensor): tensor of shape (b, l, d)
        """
        b, l, _ = hidden_states.size()
        
        seqlen_offset = 0
        rotary_max_seqlen = None
        if inference_params is not None:
            seqlen_offset = inference_params.lengths_per_sample if inference_params.lengths_per_sample is not None else inference_params.seqlen_offset
            rotary_max_seqlen = inference_params.max_seqlen
        
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        
        qkv = torch.cat([q, k, v], dim=-1)
        qkv = self._apply_rotary(qkv, seqlen_offset, rotary_max_seqlen)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(b, l, self.num_heads, self.head_dim)
        k = k.view(b, l, self.num_heads, self.head_dim)
        v = v.view(b, l, self.num_heads, self.head_dim)
        
        seq_len_pre_pad = q.shape[1]
        pad = -1
        if seq_len_pre_pad % self.seg_len != 0:
            # add padding to the end of the sequence dimension (dim=1)
            pad = self.seg_len - (seq_len_pre_pad % self.seg_len)
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
            
        batch_size, seq_len, n_heads, d_qk = tuple(q.shape)
        _, _, _, d_vo = tuple(v.shape)
        
        num_segments = seq_len//self.seg_len
        
        scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / (d_qk**.5)
        scores_terrace = scores.clone()
        
        causal_mask = torch.triu(torch.ones(scores.size(-1), scores.size(-1), device=q.device), diagonal=1).bool()
        scores      = scores.masked_fill(causal_mask, float('-inf'))
        
        # calculate terrace mask
        l = self.seg_len
        m = math.ceil(max(q.shape[1], k.shape[1]) / self.seg_len)
        mask  = torch.block_diag(*[torch.ones((l, l), )] * m)
        mask += torch.roll(mask, -l, -1) # this adds the terracing
        if mask.shape[0] > q.shape[1]:
            mask = mask[-q.shape[1]:]
        if mask.shape[1] > k.shape[1]:
            mask = mask[:, -k.shape[1]:]
            
        mask = mask[None, None, ...]
        mask = torch.tril(mask).to(q.device)
        
        scores_terrace = scores.clone()
        scores_terrace = scores_terrace.masked_fill(~mask.bool(), float('-inf'))
        
        # calculate terrace attention
        attention_terrace = torch.nn.functional.softmax(scores_terrace, dim=-1)
        output_terrace    = torch.einsum('bhnm,bmhd->bnhd', attention_terrace, v).permute(0, 2, 1, 3).reshape(batch_size, n_heads, num_segments, self.seg_len, d_vo)
        # end terrace
        
        # apply softmax
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        # attention_weights = torch.nn.functional.dropout(attention_weights, self.dropout if self.training else 0.0)
        
        attention_weights_chunked = attention_weights.reshape(batch_size, n_heads, num_segments, self.seg_len, seq_len)
        
        per_chunk_decay = torch.exp(self.index_decays).unsqueeze(0).unsqueeze(2).unsqueeze(-1)
        
        attention_weights_chunked   = attention_weights_chunked * per_chunk_decay
        per_chunk_segsummed_weights = attention_weights_chunked.sum(dim=-2) 
        
        mega_decay_const = torch.clamp(torch.exp(self.mega_decays * self.seg_len).reshape(1, n_heads, 1), min=0.0, max=2.0)
        weightings       = [torch.zeros_like(per_chunk_segsummed_weights[:,:,0,:])]
        
        for i in range(1, num_segments):
            to_append = weightings[-1] * mega_decay_const + per_chunk_segsummed_weights[:,:,i-1,:]
            weightings.append(to_append)
            
        weightings = torch.stack(weightings, dim=-2)
        weightings = weightings / (weightings.sum(dim=-1, keepdim=True) + self.noise_eps)
        
        weighted_key_chunks      = torch.einsum('bhrs,bshd->brshd',  weightings, k)
        weighted_kv_state_chunks = torch.einsum('brshd,bshe->brhde', weighted_key_chunks, v)
        
        query_chunks     = q.reshape(batch_size, num_segments, self.seg_len, n_heads, d_qk)
        attention_output = torch.einsum('brlhd,brhde->bhrle', query_chunks, weighted_kv_state_chunks)
        
        terrace_weight = torch.sigmoid(self.terrace_mix)
        terrace_weight = terrace_weight.view(1, 1, 1, self.seg_len, 1)
        
        y = terrace_weight * output_terrace + (1 - terrace_weight) * attention_output
        y = y.reshape(batch_size, n_heads, seq_len, d_vo)
        
        # undo padding
        if pad != -1:
            y = y[:, :, :seq_len_pre_pad, :]
        
        y = y.permute(0, 2, 1, 3).contiguous()
        y = y.reshape(batch_size, -1, self.d_model)
        
        y = self.c_proj(y)
        
        return y

