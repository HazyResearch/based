import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)
from src.pyutils.test_build_utils import __eq
idx = 0
debug = False
tile  = 16
import linear_attend_causal_reg as mod


def make_causal(X):
    (b,h,n,m) = X.shape
    mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
    X[mask] = 0.
    return X
def make_causal_local(X):
    (n,m) = X.shape
    mask= ~(torch.arange(n).view(n,1) >= torch.arange(n).view(1,n)).expand(n,n)
    X[mask] = 0.
    return X

b = 1
h = 1
n = 16*8*2*4
d = 16
dv = 64
### Debug
warps     = 8
warp_size = 16
last_A0    = torch.zeros(d)
last_A1    = torch.zeros(d,d)
 

def print_tiles_nb(str, t):
    print(f"TILES NB {t.shape}")
    for j in range(t.size(0)//d):
        print(f"{str} TILE NB tile={j}")
        print(f"{t[j*tile:(j+1)*tile,:]}")


def a12_test_debug(dt,verbose=True):
    s = (b,h,n,d)
    # Q,K = [torch.ones(b,h,n,d, dtype=dt, device='cuda')*(1+torch.arange(n,device='cuda')).reshape(1,1,n,1) for _ in range(2)]
    # V   = torch.ones(b,h,n,dv, dtype=dt, device='cuda')*(1+torch.arange(dv,device='cuda').reshape(1,1,1,dv))
    # Q,K,V
    #Q,K = [torch.randn(b,h,n,d, dtype=dt, device='cuda')/d for _ in range(2)]
    Q   = torch.ones(b,h,n,d, dtype=dt, device='cuda')*(1+torch.arange(d,device='cuda').reshape(1,1,1,d))/d
    K   = torch.ones(b,h,n,d, dtype=dt, device='cuda')*(1+torch.arange(n,device='cuda') % 16).reshape(1,1,n,1)/16
    V   = torch.ones(b,h,n,dv, dtype=dt, device='cuda')*(1+torch.arange(dv,device='cuda').reshape(1,1,1,dv))/dv
    A1_out  = torch.zeros_like(V)
    A1_o    = torch.zeros_like(V)
    A1y_o   = torch.zeros_like(V)

    Qf, Kf, Vf = [_.float() for _ in [Q,K,V]]
    O  = torch.einsum("bhnd,bhmd->bhnm", Qf, Kf)**2
    if verbose: print(f"O[0,0,:16,]\n{O[0,0,:16,:16]}")

    O2 = make_causal(O)
    T2 = torch.einsum("bhnm,bhmd->bhnd", O2, Vf).bfloat16()
    if verbose:
        A1_inc = torch.zeros(d,dv,dtype=dt, device='cuda')
        for r in range(n//16):
            s = slice(r*16,(r+1)*16)
            Ktc = K[0,0,s].transpose(0,1)
            Vc  = V[0,0,s]
            Qc  = Q[0,0,s]
            # Qc@A1 + make_causal(Qc@Ktc)@Vc

            a1y = Qc@A1_inc + make_causal_local(Qc@Ktc)@Vc
            A1y_o[0,0,s] = a1y
            A1_out[0,0,s] = A1_inc
            print(f"A1 tile={r}\n{A1_inc}")
            print(f"KTC={Ktc}")
            print(f"Vc={Vc}")
            A1_inc += Ktc@Vc
            print(f"a1y tile={r}\n{a1y}")
    o = torch.zeros_like(V)
    A1_y = torch.zeros_like(V)
    A0o  = torch.zeros_like(V)
    A0_out = V.cumsum(dim=2)
    mod.a012_compute_debug(Q,K,V, o, A0o, A1_o, A1_y)
    torch.set_printoptions(threshold=1<<16)
    full_term  = A0_out + A1y_o + T2/2
    guess_term = A0o    + A1_y  + T2/2 

    __eq("A0 == A0_out", A0_out, A0o , debug=False)
    __eq("A1 == A1_out", A1_out, A1_o, debug=False)
    __eq("A1y == A1y"  , A1y_o , A1_y, debug=False)
    __eq("First Terms"  , A0_out + A1y_o, A0o + A1_y, debug=False)
    __eq("Full Term", full_term, o, debug=False)
    __eq("Full Term : Guess", full_term, guess_term, debug=False)
    T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Qf, Kf))
    T1 = torch.einsum("bhnm,bhme->bhne", T1a, Vf)  
    __eq("T1 == A1y", T1, A1y_o, debug=False)
    if verbose:
        print(f"shapes = {A1y_o.squeeze().shape} {A1_y.squeeze().shape} {A1_o.shape, A1_out.shape}")
        print_tiles_nb("full_term", full_term.squeeze())
        print_tiles_nb("o", o.squeeze())
        print_tiles_nb("guess_term", guess_term.squeeze())


def a12_test(dt,verbose=False, use_ones=False, profile=False):
    if profile:
        (b,h,n,d,dv) = 16,16,16384,16,64
        print(f"profiling at Q,K ({b,h,n,d}) and V ({b,h,n,dv}) ")
    else:
        (b,h,n,d,dv) = 1,1,128,16,64
        print("Testing at Q,K ({b,h,n,d}) and V ({b,h,n,dv}) ")

    if use_ones:
        Q   = torch.ones(b,h,n,d, dtype=dt, device='cuda')*(1+torch.arange(d,device='cuda').reshape(1,1,1,d))/d
        K   = torch.ones(b,h,n,d, dtype=dt, device='cuda')*(1+torch.arange(n,device='cuda') % 16).reshape(1,1,n,1)/16
        V   = torch.ones(b,h,n,dv, dtype=dt, device='cuda')*(1+torch.arange(dv,device='cuda').reshape(1,1,1,dv))/dv
    else:
        Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
        K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
        V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

    o   = torch.zeros_like(V)
    mod.a012_compute(Q,K,V, o)

    if not profile:
        Qf, Kf, Vf = [_.float() for _ in [Q,K,V]]
        O   = torch.einsum("bhnd,bhmd->bhnm", Qf, Kf)**2
        O2  = make_causal(O)
        T2  = torch.einsum("bhnm,bhmd->bhnd", O2, Vf)
        T1a = make_causal(torch.einsum("bhnd,bhmd->bhnm", Qf, Kf))
        T1  = torch.einsum("bhnm,bhme->bhne", T1a, Vf)  
        T0  = Vf.cumsum(dim=2)

        full_term  = T0 + T1 + T2/2
        __eq("Full Term", full_term.bfloat16(), o, debug=False)


profile = False
a12_test(torch.bfloat16, verbose=False, profile=profile)
#if not profile:a a12_test_debug(torch.bfloat16)

