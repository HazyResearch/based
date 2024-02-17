#include <iostream>
#include <math.h>
#include <assert.h>
#include <mma.h>
using namespace nvcuda;

#include "src/global_warp_tile/warp_tile_abstract.cuh"
#include "src/reg_tile/register_tile.cuh" 
#include "src/reg_tile/register_frag.cuh"
#include "src/pyutils/torch_helpers.cuh"

// **** ASYNC INCLUDE *****
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <ATen/cuda/CUDAContext.h>  // Include necessary for getting the current stream


const int nThreads = 256;

template<typename H>
__device__
void thread_block_store(typename H::T *_dst, gwt::tile<H> &_src) {
    float4* dst = (float4*) _dst;
    float4* src = (float4*) _src.ptr(); 
    using T = typename H::T;

    auto bytes_per_row    = H::cols * sizeof(T); // non-padded
    auto f4_stride        = (H::_row_stride*sizeof(T))/sizeof(float4);
    DEBUG_ONLY(assert(H::_row_stride*sizeof(T) % sizeof(float4)  == 0);)

    auto reads_per_row    = bytes_per_row / sizeof(float4);
    DEBUG_ONLY(assert(bytes_per_row % sizeof(float4) == 0);)

    auto rows_per_block    = nThreads / reads_per_row; // rows per thread block.
    auto row_skipping_only = (nThreads % reads_per_row) == 0; // if we read complete rows.

    auto f4_elements      = (H::nElements * sizeof(T)) / sizeof(float4);
    
    if( row_skipping_only ) {
        auto col      = threadIdx.x % reads_per_row; // this will be fixed
        auto row_base = threadIdx.x / reads_per_row; 
        auto _stride  = f4_stride*rows_per_block; // we we will just skip!
         __syncthreads();
        auto idx = row_base*f4_stride + col;
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads, idx += _stride) {
            dst[i] = src[idx]; 
        }
    } else {
         __syncthreads();
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads) {
            auto col = i % reads_per_row;
            auto row = i / reads_per_row;
            dst[i] = src[row*f4_stride + col];
        }
    }
}


template<typename H>
__device__
void thread_block_load_async(gwt::tile<H> &_dst, const typename H::T *_src, 
                            cuda::barrier<cuda::thread_scope::thread_scope_block> &barrier, const int nThreads=256) {
    float4* dst = (float4*) _dst.ptr();
    float4* src = (float4*) _src; 
    using T = typename H::T;

    auto bytes_per_row    = H::cols * sizeof(T); // non-padded
    auto f4_stride        = (H::_row_stride*sizeof(T))/sizeof(float4);
    DEBUG_ONLY(assert(H::_row_stride*sizeof(T) % sizeof(float4)  == 0);)

    auto reads_per_row    = bytes_per_row / sizeof(float4);
    DEBUG_ONLY(assert(bytes_per_row % sizeof(float4) == 0);)

    auto rows_per_block    = nThreads / reads_per_row; // rows per thread block.
    auto row_skipping_only = (nThreads % reads_per_row) == 0; // if we read complete rows.

    auto f4_elements      = (H::nElements * sizeof(T)) / sizeof(float4);
    
    if( row_skipping_only ) {
        auto col      = threadIdx.x % reads_per_row; 
        auto row_base = threadIdx.x / reads_per_row; 
        auto _stride  = f4_stride*rows_per_block; // we we will just skip!
         __syncthreads();
        auto idx = row_base*f4_stride + col;
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads, idx += _stride) {
            cuda::memcpy_async(dst + idx, src + i, sizeof(float4), barrier);
        }
    } else {
         __syncthreads();
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads) {
            auto col = i % reads_per_row;
            auto row = i / reads_per_row;
            cuda::memcpy_async(dst + row*f4_stride + col, src + i, sizeof(float4), barrier);
        }
    }
}

template<typename H>
__device__
void thread_block_load(gwt::tile<H> &_dst, const typename H::T *_src, const int nThreads=256) {
    float4* dst = (float4*) _dst.ptr();
    float4* src = (float4*) _src; 
    using T = typename H::T;

    auto bytes_per_row    = H::cols * sizeof(T); // non-padded
    auto f4_stride        = (H::_row_stride*sizeof(T))/sizeof(float4);
    DEBUG_ONLY(assert(H::_row_stride*sizeof(T) % sizeof(float4)  == 0);)

    auto reads_per_row    = bytes_per_row / sizeof(float4);
    DEBUG_ONLY(assert(bytes_per_row % sizeof(float4) == 0);)

    auto rows_per_block    = nThreads / reads_per_row; // rows per thread block.
    auto row_skipping_only = (nThreads % reads_per_row) == 0; // if we read complete rows.

    auto f4_elements      = (H::nElements * sizeof(T)) / sizeof(float4);
    
    if( row_skipping_only ) {
        auto col      = threadIdx.x % reads_per_row; 
        auto row_base = threadIdx.x / reads_per_row; 
        auto _stride  = f4_stride*rows_per_block; // we we will just skip!
         __syncthreads();
        auto idx = row_base*f4_stride + col;
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads, idx += _stride) {
            dst[idx] = src[i];
        }
    } else {
         __syncthreads();
        for(auto i = threadIdx.x; i < f4_elements; i+=nThreads) {
            auto col = i % reads_per_row;
            auto row = i / reads_per_row;
            dst[row*H::_row_stride + col] = src[i];
        }
    }
}


using reg1x1_t = reg_tile::reg_tile_desc<1,1>;
using reg1x2_t = reg_tile::reg_tile_desc<1,2>;
using reg2x1_t = reg_tile::reg_tile_desc<2,1>;
__device__ const static reg_tile::rt<reg1x1_t> rt1x1;
__device__ const static reg_tile::rt<reg1x2_t> rt1x2;
__device__ const static reg_tile::rt<reg2x1_t> rt2x1;


template<typename op>
__device__ 
void shm_broadcast(float &f, float *shm, const int workers = 4) {
    auto warpid = gwt::warp_id();
    auto lane   = gwt::laneid();
    shm[warpid] = f;
    __syncthreads();
    if(warpid == 0) {
        if(lane == 0) {
            for(auto j = 1; j < workers; j++) {f = op::op(f,shm[j]);}
            for(auto j = 0; j < workers; j++) {shm[j] = f;}
        }
        __syncwarp();
    }
    __syncthreads();
    f = shm[warpid];
}

// Input is: a vector of q, the matrix K and the matrix V
// our grid is (batch, head)

// Here is the pytorch to simulate the algorithm
// 
// qs = [q for j in range(4)] # broadcast q to each warp
// ks = [k[:,j*d//4:(j+1)*d//4] for j in range(4)] # shard k
// ws = [torch.einsum("d, de->e", qs[j],ks[j]) for j in range(4)]
// 
// local_max = [ws[j].max() for j in range(4)] # compute local, then global max
// the_max = torch.tensor(local_max).max()
// 
// ews = [torch.exp(ws[j] - the_max) for j in range(4)]
// es  = [ews[j].sum() for j in range(4)]
// the_sum = torch.tensor(es).sum()
// # broadcast ews
// w   = torch.concat(ews)
// w  /= the_sum
// vs  = [v[:,j*d//4:(j+1)*d//4] for j in range(4)]
// os  = [torch.einsum("d, de->e", w,vs[j]) for j in range(4)]
// o_e = torch.concat(os)

template<typename H, typename T>
__global__
void sliding_window_ker(int n, int j, bool just_q, const T* __q, const T* __k, const T* __v, T* __o) {
    // Read q, k, v into shared memory.
     auto warpid = gwt::warp_id();
    const int d = 64;
    const int window_size = 64;
    const int workers = 4;
    const int threads = workers * gwt::WARP_SIZE;
    auto head_offset  = blockIdx.x * n * d;
    
    const H* _q = device_cast(__q) + (just_q ? blockIdx.x*d : head_offset);
    const H* _k = device_cast(__k) + head_offset;
    const H* _v = device_cast(__v) + head_offset;
          H* _o = device_cast(__o) + blockIdx.x*d; // just a single vector

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    int *_pshm = &__shm[0];
    // padding to remove bank conflicts
    _tile4x4<H> &k  =  gwt::shm_advance<_tile4x4<H>>(_pshm);
    _tile4x4<H> &v  =  gwt::shm_advance<_tile4x4<H>>(_pshm);
    
    __shared__ _rowvec4<H> q,o;
    __shared__ _rowvec4<float> w;
    __shared__ float _max[workers], _sum[workers];
    // *****
    // Load the relevant data structures
    // TODO: Split these loads across warps:
    //  * Each warp will hold a column fragment of K and V.
    //    that is, K[:,warpid] and V[:,warpid] (indexing in tiles)
    //  * We will broadcast the current working vector (q, w)

    const auto start_idx = just_q ? 0 : (j-window_size)*d;
    thread_block_load(k, _k + start_idx, threads);
    thread_block_load(v, _v + start_idx, threads);
    if(warpid == 0) { 
        auto vec_idx = just_q ? 0 : j * d;
        gwt::load(q, _q + vec_idx); 
    }

    // * A row vec means that it has *4* chunks.
    // * In each one, the vector is stored colocated with its row. 
    reg_tile::reg_tile_desc<1,4>::row_vec qv; // full local copy 
    reg_tile::reg_tile_desc<1,4>::col_vec ws; // full local copy 
    reg_tile::reg_tile_desc<1,4>::accum k_slice;
    
    reg_tile::reg_tile_desc<4,1>::col_vec wv; // full local copy 
    reg_tile::reg_tile_desc<4,1>::row_vec os; // shards.
    reg_tile::reg_tile_desc<4,1>::accum v_slice; // Each of the 4 workers stores a column.

    reg_tile::rt<reg_tile::reg_tile_desc<1,4>> rt1x4;
    reg_tile::rt<reg_tile::reg_tile_desc<4,1>> rt4x1;

    // These are column slices of the matrix.
    // | K_1 | K_2 | K_3 |
    __syncthreads();
    rt1x4.vec_to_rvec(qv, q.data); // every warp gets a full copy of q    
    rt1x4.tile_to_accum(k_slice, k.template subtile<1,4>(warpid, 0));

    // ********
    // The algorithm.
    // qs = [q for j in range(4)] # broadcast q to each warp
    // ks = [k[:,j*d//4:(j+1)*d//4] for j in range(4)] # shard k
    // ws = [torch.einsum("d, de->e", qs[j],ks[j]) for j in range(4)]
    rt1x4.zero(ws);
    rt1x4.gemv(ws, qv, k_slice);

    // local_max = [ws[j].max() for j in range(4)] # compute local, then global max
    // the_max = torch.tensor(local_max).max()
    float local_max= -INFINITY;
    rt1x4.max(local_max, ws);
    shm_broadcast<typename decltype(rt4x1)::max_op>(local_max, _max);
    
    // ews = [torch.exp(ws[j] - the_max) for j in range(4)]
    rt1x4.sub_rvec_const(ws, local_max);
    rt1x4.exp(ws);
    // es  = [ews[j].sum() for j in range(4)]
    float local_sum = 0.f;
    rt1x4.sum(local_sum, ws);
    shm_broadcast<typename decltype(rt4x1)::sum_op>(local_sum, _sum);
    
    // w  /= the_sum
    rt1x4.div_rvec_const(ws, local_sum);

    // broadcast w back to shared memory
    rt1x4.rvec_to_vec(&w.data[warpid*gwt::TILE_DIM], ws);
    __syncthreads(); // let the writes complete
    rt4x1.vec_to_rvec(wv, w.data); // read the *whole* v here.
    
    // we want a column stripe of V
    rt4x1.tile_to_accum(v_slice, v.template subtile<4,1>(0, warpid));
    rt4x1.zero(os);
    rt4x1.gemv(os, wv, v_slice);
    
    // now we have a fragment of v and we write, this write is to *global* memory.
    rt1x1.rvec_to_vec(_o + warpid*gwt::TILE_DIM, os);
}

template<typename H, typename T>
__global__
void sliding_window_ker_hack(int n, int j, bool just_q, const T* __q, const T* __k, const T* __v, T* __o) {
    // Read q, k, v into shared memory.
     auto warpid = gwt::warp_id();
    const int d = 64;
    const int window_size = 64;
    const int workers = 4;
    const int threads = workers * gwt::WARP_SIZE;
    auto head_offset  = blockIdx.x * n * d;
    
    const H* _q = device_cast(__q) + (just_q ? blockIdx.x*d : head_offset);
    const H* _k = device_cast(__k) + head_offset;
    const H* _v = device_cast(__v) + head_offset;
          H* _o = device_cast(__o) + blockIdx.x*d; // just a single vector

    __shared__ _tile4x4<H> k,v;
    __shared__ _rowvec4<H> q,o;
    __shared__ _rowvec4<float> w;
    __shared__ float _max[workers], _sum[workers];
    // *****
    // Load the relevant data structures
    // TODO: Split these loads across warps:
    //  * Each warp will hold a column fragment of K and V.
    //    that is, K[:,warpid] and V[:,warpid] (indexing in tiles)
    //  * We will broadcast the current working vector (q, w)

    const auto start_idx = just_q ? 0 : (j-window_size)*d;
    thread_block_load(k, _k + start_idx, threads);
    thread_block_load(v, _v + start_idx, threads);
    if(warpid == 0) { 
        auto vec_idx = just_q ? 0 : j * d;
        gwt::load(q, _q + vec_idx); 
    }

    // A row vec means that it has *4* chunks.
    // In each one, the vector is stored colocated with its row. 
    reg_tile::reg_tile_desc<1,4>::row_vec qv; // full local copy 
    reg_tile::reg_tile_desc<1,4>::col_vec ws; // full local copy 
    reg_tile::reg_tile_desc<1,4>::accum k_slice;
    
    reg_tile::reg_tile_desc<4,1>::col_vec wv; // full local copy 
    reg_tile::reg_tile_desc<4,1>::row_vec os; // shards.
    reg_tile::reg_tile_desc<4,1>::accum v_slice; // Each of the 4 workers stores a column.

    reg_tile::rt<reg_tile::reg_tile_desc<1,4>> rt1x4;
    reg_tile::rt<reg_tile::reg_tile_desc<4,1>> rt4x1;

    // These are column slices of the matrix.
    // | K_1 | K_2 | K_3 |
    __syncthreads();
    rt1x4.vec_to_rvec(qv, q.data); // every warp gets a full copy of q    
    rt1x4.tile_to_accum(k_slice, k.template subtile<1,4>(warpid, 0));

    // ********
    // The algorithm.
    // qs = [q for j in range(4)] # broadcast q to each warp
    // ks = [k[:,j*d//4:(j+1)*d//4] for j in range(4)] # shard k
    // ws = [torch.einsum("d, de->e", qs[j],ks[j]) for j in range(4)]
    rt1x4.zero(ws);
    rt1x4.gemv(ws, qv, k_slice);

    // local_max = [ws[j].max() for j in range(4)] # compute local, then global max
    // the_max = torch.tensor(local_max).max()
    float local_max= -INFINITY;
    rt1x4.max(local_max, ws);
    shm_broadcast<typename decltype(rt4x1)::max_op>(local_max, _max);
    
    // ews = [torch.exp(ws[j] - the_max) for j in range(4)]
    rt1x4.sub_rvec_const(ws, local_max);
    rt1x4.exp(ws);
    // es  = [ews[j].sum() for j in range(4)]
    float local_sum = 0.f;
    rt1x4.sum(local_sum, ws);
    shm_broadcast<typename decltype(rt4x1)::sum_op>(local_sum, _sum);
    
    // w  /= the_sum
    rt1x4.div_rvec_const(ws, local_sum);

    // broadcast w back to shared memory
    rt1x4.rvec_to_vec(&w.data[warpid*gwt::TILE_DIM], ws);
    __syncthreads(); // let the writes complete
    rt4x1.vec_to_rvec(wv, w.data); // read the *whole* v here.
    
    // we want a column stripe of V
    rt4x1.tile_to_accum(v_slice, v.template subtile<4,1>(0, warpid));
    rt4x1.zero(os);
    rt4x1.gemv(os, wv, v_slice);
    
    // now we have a fragment of v and we write, this write is to *global* memory.
    rt1x1.rvec_to_vec(_o + warpid*gwt::TILE_DIM, os);
}

void 
sliding_window(int j,   
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    torch::Tensor o) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    
    bool capturing_graph = true;
    uint batch = q.size(0);
    uint head  = q.size(1);
    uint d     = q.size(3);
    TORCH_CHECK(d == 64, "Only dimension 64 implemented...");

    bool k_same = true, v_same = true;
    for(auto i = 0; i < 2; i++) { 
        k_same &= q.size(i) == k.size(i);
        v_same &= q.size(i) == v.size(i);
    }
    k_same &= d == k.size(3);
    v_same &= d == v.size(3);
    uint n     = k.size(2);
    v_same &= v.size(2) == n;

    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "X and K_out should be same size");
    TORCH_CHECK(v_same, "X and V_out should be same size");
    
    const int workers = 4;

    using H = __nv_bfloat16;
    using T = c10::BFloat16;

    unsigned long mem_size  = 2*sizeof(_tile4x4<H>); // q, k and v are double buffered.    
    auto threads = workers * gwt::WARP_SIZE;

    if(capturing_graph) {
        // When using cuda graphs we want to use statically allocated memory 
        auto stream_wrapper = at::cuda::getCurrentCUDAStream(q.device().index());
        cudaStream_t stream = stream_wrapper.stream();
        DEBUG_ONLY(printf("Capturing graph"));
        sliding_window_ker_hack<H,T><<<batch*head,threads,0,stream>>>(n, j, q.size(2) == 1,
                        q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), 
                        o.data_ptr<T>());
        DEBUG_ONLY(printf("after the kernel call.\n"));
    } else {
        CHECK_CUDA_ERROR(cudaFuncSetAttribute(
                sliding_window_ker<H, T>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
        sliding_window_ker<H,T><<<batch*head,threads,mem_size>>>(n, j, q.size(2) == 1,
                        q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), 
                        o.data_ptr<T>());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    DEBUG_ONLY(printf("ending\n")); 
}

