"""We used this code in constructing Figure 1 (Left) in the Based paper."""


import time
import torch

num_iters = 100
warmup_iters = 100
dt = torch.bfloat16

batch_size = 512

for tile_size in [1, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096]:

    A = torch.randn(batch_size, tile_size, tile_size).to(dt).to("cuda")
    B = torch.randn(batch_size, tile_size, tile_size).to(dt).to("cuda")

    for i in range(warmup_iters):
        C = torch.bmm(A, B)

    timings = []
    for i in range(num_iters):

        torch.cuda.synchronize()
        start = time.time()

        C = torch.bmm(A, B)

        torch.cuda.synchronize()
        end = time.time()
        timings.append(end - start)
    
    print(f"{sum(timings)/len(timings):.6f}")