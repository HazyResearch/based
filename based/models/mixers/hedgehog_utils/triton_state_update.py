import torch
import triton
import triton.language as tl

def hedgehog_step(kv_state, k_state, q, k, v, denom: bool=False):
    """
    Borrows from Mamba implementation
    Argument:
        kv_state: (batch, d_model, dstate)
        k_state: (batch, d_state)
        q: (batch, d_model)
        k: (batch, d_model)
        v: (batch, d_model)

    Return:
        out: (batch, d_model)
    """
    batch, dim, d_state = kv_state.shape
    assert v.shape == (batch, dim)
    assert q.shape == (batch, d_state)
    assert k.shape == q.shape
    out = torch.empty_like(v)
   
    # This is currently not blocking along the d_state dimension which 
    # is okay for small d_state (e.g. 16 in Mamba), but an issue for the state sizes
    # we're dealing with 
    BLOCK_SIZE_M, num_warps = (16,16)
    BLOCK_SIZE_DSTATE = d_state
    grid = lambda META: (triton.cdiv(dim, BLOCK_SIZE_M), batch)

    with torch.cuda.device(v.device.index):
        _hedgehog_step[grid](
            kv_state, k_state, v, k, q, out,
            dim, d_state,
            kv_state.stride(0), kv_state.stride(1), kv_state.stride(2),
            k_state.stride(0), k_state.stride(1),
            v.stride(0), v.stride(1),
            k.stride(0), k.stride(1),
            q.stride(0), q.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_DSTATE,
            num_warps=num_warps,
        )
    return out


@triton.jit
def _hedgehog_step(
    # Pointers to matrices
    kv_state_ptr, k_state_ptr, V_ptr, K_ptr, Q_ptr, out_ptr, 
    # Matrix dimensions
    dim, dstate,
    # Strides
    stride_kv_state_batch, stride_kv_state_dim, stride_kv_state_dstate,
    stride_k_state_batch, stride_k_state_dstate,
    stride_v_batch, stride_v_dim,
    stride_k_batch, stride_k_dstate,
    stride_q_batch, stride_q_dstate,
    stride_out_batch, stride_out_dim,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    kv_state_ptr += pid_b * stride_kv_state_batch
    k_state_ptr += pid_b * stride_k_state_batch
    V_ptr += pid_b * stride_v_batch
    K_ptr += pid_b * stride_k_batch
    Q_ptr += pid_b * stride_q_batch
    out_ptr += pid_b * stride_out_batch

    # set pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    kv_state_ptrs = kv_state_ptr + (offs_m[:, None] * stride_kv_state_dim + offs_n[None, :] * stride_kv_state_dstate)
    k_state_ptrs = k_state_ptr + offs_n * stride_k_state_dstate
    v_ptrs = V_ptr + offs_m * stride_v_dim
    k_ptrs = K_ptr + offs_n * stride_k_dstate
    q_ptrs = Q_ptr + offs_n * stride_q_dstate
    out_ptrs = out_ptr + offs_m * stride_out_dim

    # loads
    kv_state = tl.load(kv_state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    k_state = tl.load(k_state_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    V = tl.load(v_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    K = tl.load(k_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    Q = tl.load(q_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)

    # compute state updates
    kv_state = kv_state + K[None, :] * V[:, None]
    tl.store(kv_state_ptrs, kv_state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    k_state = k_state + K  
    tl.store(k_state_ptrs, k_state, mask=(offs_n < dstate))
    
    # compute output
    num = tl.sum(kv_state * Q[None, :], axis=1)

    # if use norm
    den = tl.sum(k_state  * Q[None, :], axis=1) + 1e-12
    out = num / den
    # else
    # out = num

    tl.store(out_ptrs, out, mask=offs_m < dim)
