import torch
import time
import based_inference as mod

from collections import defaultdict
from statistics import median
import matplotlib.pyplot as plt


def pytorch_step(kv_state, kv_state_t, k_state, q, k, v, denom: bool=True, eps: float=1e-6):
    """
    Argument:
        kv_state: (batch, d_model, dstate)
        k_state: (batch, d_state)
        q: (batch, d_model)
        k: (batch, d_model)
        v: (batch, d_model)

    Return:
        out: (batch, d_model)
    """

    # prepare inputs
    kv_state_ref = kv_state.detach().clone() # So that the kernel can write to different memory
    k_state_ref  = k_state.detach().clone()
    
    # compute
    torch.cuda.synchronize()
    t0 = time.time()

    k_state_ref += k
    kv_state_ref += torch.einsum("bf,bd->bdf", k, v)
    num = torch.einsum("bf,bdf->bd", q, kv_state_ref)
    # den = torch.einsum("bf,bf->b", q, k_state) + eps
    y = num #/ den.unsqueeze(-1)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1 - t0
    return y, kv_state_ref, k_state_ref, tot


def based_step_test(kv_state, kv_state_t, k_state, q, k, v):
    out_tc       = torch.zeros_like(v)

    torch.cuda.synchronize()
    t0 = time.time()

    mod.based_step(q, k, v, kv_state_t, k_state, out_tc)

    torch.cuda.synchronize()
    t1 = time.time()
    tot = t1 - t0
    return out_tc, tot


def based_step_benchmark(dt, device='cuda'):
    num_iters = 10

    methods = {
        'Pure PyTorch': pytorch_step, 
        'Based Kernel': based_step_test,
    }
    method2timing = defaultdict(dict)
    for b in [8, 16, 32, 64, 128, 256]:
        h = 16  
        dv = 64
        d_state = 320   # rounding 1 + 16 + 16^2 to the nearest multiple of 64 for hardware.
        print(f"{b=}, {dv=}, {h=}, {d_state=}")

        # prepare inputs 
        torch.random.manual_seed(0)
        kv_state = torch.randn(b*h, dv, d_state, dtype=dt, device=device)/d_state
        k_state = torch.randn(b*h, d_state, dtype=dt, device=device)/d_state
        v = torch.randn(b*h, dv, device=device, dtype=dt)/dv
        k = torch.randn(b*h, d_state, device=device, dtype=dt)/d_state
        q = torch.randn(b*h, d_state, device=device, dtype=dt)/d_state
        kv_state_t = kv_state.transpose(1,2).contiguous() # We prefer the other order for iteration.

        for name, fn, in methods.items():
            lst = [
                fn(kv_state=kv_state, kv_state_t=kv_state_t, k_state=k_state, v=v, k=k, q=q)
                for _ in range(num_iters)
            ]
            lst_time = [x[-1] for x in lst]
            _time = median(lst_time)
            if b > 8: method2timing[name][b] = _time * 1000

    # plot: time vs. batch
    fig, ax = plt.subplots()
    for name, timing in method2timing.items():
        ax.plot(timing.keys(), timing.values(), label=name, marker='o')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time (ms)')
    ax.legend()
    # save pdf
    plt.savefig('h100_based_step_benchmark.pdf', format='pdf', bbox_inches='tight')


def based_step_correct(dt, device='cuda'):
    # specify dimensions
    batch_size = 128
    heads = 16  
    d_model = 64
    d_state = 320

    # prepare inputs 
    torch.random.manual_seed(0)
    kv_state = torch.randn(batch_size*heads, d_model, d_state, dtype=dt, device=device)/d_state
    k_state = torch.randn(batch_size*heads, d_state, dtype=dt, device=device)/d_state
    v = torch.randn(batch_size*heads, d_model, device=device, dtype=dt)/d_model
    k = torch.randn(batch_size*heads, d_state, device=device, dtype=dt)/d_state
    q = torch.randn(batch_size*heads, d_state, device=device, dtype=dt)/d_state
    kv_state_t = kv_state.transpose(1,2).contiguous() # We prefer the other order for iteration.

    # run implementations
    out_ref, kv_state_ref, k_state_ref, _   = pytorch_step(kv_state=kv_state, kv_state_t=kv_state_t, k_state=k_state, v=v, k=k, q=q)
    out_tc       = torch.zeros_like(v)
    mod.based_step(q, k, v, kv_state_t, k_state, out_tc)

    # correctness checks
    _kv_state = kv_state_t.transpose(1,2) 
    print(f"out max diff: {(out_tc - out_ref).abs().max().item()}")
    print(f"k_state  max diff: {(k_state - k_state_ref).abs().max().item()}")
    print(f"kv_state max diff: {(_kv_state - kv_state_ref).abs().max().item()}")


print("Benchmarking based_step...")
based_step_benchmark(torch.bfloat16)
print(f"\nCorrectness check for based_step...")
based_step_correct(torch.bfloat16)

