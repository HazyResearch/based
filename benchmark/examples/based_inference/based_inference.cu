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
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <ATen/cuda/CUDAContext.h>

const int d_model =  64; 
const int d_state = 320;

template<typename H> __device__ float4* _f4p(H *x)  { return (float4* ) x;}
template<typename H> __device__ const float4* _f4pc(H *x)  { return (const float4* ) x;}
template<typename H> __device__ int _bank(H*x) { return ((uint64_t)x)/4 % 32; }


template <typename H, typename T>
__global__
void based_simple_ker(const T* __q, const T* __k, const T* __v, // single row of q, k, v of size q model.
                    T* __kv_state, T* __k_state,
                    T* __out) { 

    auto block_start = blockIdx.x;
    auto warpid = gwt::warp_id();
    auto lane   = gwt::laneid();
    const int workers  = 8; 
    const int nThreads = workers*gwt::WARP_SIZE;

    // Data size information
    const int kv_state_size = d_model * d_state;
    const int k_state_size  = d_state;

    const H *q_g = device_cast(__q)+block_start*d_state;
    const H *k_g = device_cast(__k)+block_start*d_state;
    const H *v_g = device_cast(__v)+block_start*d_model;
          H *kv_state_g = device_cast(__kv_state)+block_start*kv_state_size;
          H *k_state_g  = device_cast(__k_state)+block_start*k_state_size;
          H *num_g      = device_cast(__out)+block_start*d_model;

    // Setup the extended shared memory. We want to be 128 byte aligned.
    const int row_bytes   = d_model * sizeof(H);
    const int align_pad   = (row_bytes & 127) == 0 ? 0 : 128 - row_bytes & 127;
    const int buffer_rows = workers * 8; 
    const int buffer_size = (d_model + align_pad)*buffer_rows;
    assert(align_pad == 0); 

    // Use the simple shared memory 
    __shared__ alignas(alignof(float4)) H kv_state[2][buffer_size];
    __shared__ alignas(alignof(float4)) H q[d_state];
    __shared__ alignas(alignof(float4)) H k[d_state];
    __shared__ alignas(alignof(float4)) H k_state[d_state];
    __shared__ alignas(alignof(float4)) H v[d_model];
    __shared__ alignas(alignof(float4)) H num[d_model];
    __shared__ alignas(alignof(float4)) H num_help[workers][d_model];

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (threadIdx.x == 0) {init(&barrier, block.size());}
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // need to make sure none calls before setup.
    
    // How many float4s -- these are 16 bytes.
    assert(d_state*sizeof(H) % sizeof(float4) == 0);
    assert(d_model*sizeof(H) % sizeof(float4) == 0);

    const auto d_state_shape = cuda::aligned_size_t<alignof(float4)>(d_state*sizeof(H)); 
    const auto d_model_shape = cuda::aligned_size_t<alignof(float4)>(d_model*sizeof(H)); 

    cuda::memcpy_async(block, q, q_g, d_state_shape, barrier_cheat);
    cuda::memcpy_async(block, k, k_g, d_state_shape, barrier_cheat);
    cuda::memcpy_async(block, k_state, k_state_g, d_state_shape, barrier_cheat);
    cuda::memcpy_async(block, v, v_g, d_model_shape, barrier_cheat);

    int tic = 0;
    int toc = 1;
    
    // Read the initial buffer slice of kv_state
    const auto buffer_row_shape = cuda::aligned_size_t<alignof(float4)>(buffer_rows*d_model*sizeof(H)); 
    cuda::memcpy_async(block, kv_state[tic], kv_state_g, buffer_row_shape, barrier);

    // Sum k to kstate and do kv. k_state += k
    barrier_cheat.arrive_and_wait(); // make sure q,k,v have arrived.
    for(auto i = threadIdx.x; i < d_state; i+=nThreads) {k_state[i] += k[i];}
    // store v across threads for the next phase.
    // assumes d_model fits in register and is a multiple of 32
    register H v_vals[d_model / gwt::WARP_SIZE];
    register H num_vals[d_model / gwt::WARP_SIZE];
    DEBUG_ONLY(assert(d_model % gwt::WARP_SIZE == 0);)

    auto j0 = 0;
    for(auto j = lane; j < d_model; j+=gwt::WARP_SIZE, ++j0) {
        v_vals[j0]   = v[j];
        num_vals[j0] = gwt::__typeconvert<float,H>(0.f);
    }
    // conduct the write back!
    __syncthreads(); // make sure all of k_state is written.
    for(auto i = threadIdx.x; i < d_state*sizeof(H)/sizeof(float4); i+=nThreads) {_f4p(k_state_g)[i] = _f4p(k_state)[i];}
    
    auto outer_batches = d_state / buffer_rows;
    auto extra_batch   = d_state % buffer_rows;
    
    const int bytes_per_batch = buffer_rows*d_model*sizeof(H);
    const int extra_bytes     = extra_batch*d_model*sizeof(H);
    const auto batch_shape    = cuda::aligned_size_t<alignof(float4)>(bytes_per_batch); 
    
    // Each row is of length d_model
    auto total_batches = (extra_batch > 0) ? outer_batches + 1 : outer_batches; 
    auto _buffer_rows  = buffer_rows; // will change if extra batches!
    // We iterate through each batch.
    // * We load the next batch asynchronously (into toc)
    // * We work on the current batch.
    // There is some cleanup about the extra batch.
    for(auto ob = 0; ob < total_batches; ob++, tic ^=1, toc ^=1) {
        auto cur_batch_idx  = ob       * buffer_rows * d_model;
        auto next_batch_idx = (ob + 1) * buffer_rows * d_model;

        barrier.arrive_and_wait(); // wait on the work buffer to be free
        if(ob + 1 < outer_batches) {// if there is more work fetch the next one.
            cuda::memcpy_async(block, kv_state[toc], 
                        kv_state_g + next_batch_idx, 
                        batch_shape , barrier);
        } else {// last batch!
            if(extra_batch > 0) {
                // NOTE: dmodel must be aligned for this to be true 
                DEBUG_ONLY(assert(d_model * sizeof(H) * extra_batch % sizeof(float4) == 0);)
                const auto extra_batch_shape = cuda::aligned_size_t<alignof(float4)>(extra_bytes); 
                cuda::memcpy_async(block, kv_state[toc], 
                        kv_state_g + next_batch_idx, 
                        extra_batch_shape, barrier); 
            }
        }
        // end load.

        _buffer_rows = (ob == outer_batches) ? extra_batch : buffer_rows;
        
        // Here we compute our chunk of: kv_state += torch.einsum("f,d->df", k, v)
        // we also compute: num = torch.einsum("f,df->d", q, kv_state)
        //
        // Each warp grabs a single value of k, k[i] and q[i]
        //
        // * The warp loops across the the kv_state on row i.
        // * Above, we stored all of v in register (v_vals) namely
        //   v[j] is stored in thread "j % 32" in a register.
        // * After reading and writing p_kvs that row is updatd.
        // * we compute num in a partitioned way, it represents:
        //   q[i]*kv_store[i,j]

        __syncwarp();
        for(auto i = warpid; i < _buffer_rows; i += workers) {
            H k_val = k[ob*buffer_rows + i]; // broadcast this value to each thread in the warp.
            H q_val = q[ob*buffer_rows + i];
            auto p_kvs = kv_state[tic] + i*d_model; // pointer to the row
            auto j0 = 0;
            for(auto j = lane; j < d_model; j += gwt::WARP_SIZE, j0++) { 
                p_kvs[j]     += k_val*v_vals[j0];
                num_vals[j0] += q_val*p_kvs[j];    
            }
        }
        __syncthreads(); // make sure the work is complete, then write back the buffered rows
        auto _stores = (ob == outer_batches) ? (extra_bytes/sizeof(float4)) : (bytes_per_batch/sizeof(float4));
        for(auto j = threadIdx.x; j < _stores; j+=nThreads) {
            _f4p(kv_state_g + cur_batch_idx)[j] = _f4p(kv_state[tic])[j];
        }
    }
    
    // At the end of the loop, the threads hold fragments (shared on j and i) 
    // for q*KV i num_vals[j0]. 
    // We need to aggregate across the warps.
    __syncwarp();
    j0 = 0;
    for(auto j = lane; j < d_model; j+= gwt::WARP_SIZE, ++j0) {
        num_help[warpid][j] = num_vals[j0];
    }
    __syncthreads(); // num_help is done
    for(auto j = threadIdx.x; j < d_model; j+=nThreads) {
        H nj = num_help[0][j];
        #pragma unroll
        for(auto w = 1; w < workers; w++) { nj += num_help[w][j]; }
        num[j] = nj;
    }

    __syncthreads();
    for(auto j = threadIdx.x; j < (d_model * sizeof(H))/sizeof(float4); j+=nThreads) {_f4p(num_g)[j] = _f4p(num)[j];}
}

void 
based_step(torch::Tensor q, torch::Tensor k, torch::Tensor v, 
            torch::Tensor kv_state, torch::Tensor k_state, torch::Tensor out) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(kv_state);
    CHECK_INPUT(k_state);
    CHECK_INPUT(out);

    auto batch = q.size(0);
    TORCH_CHECK(batch == k.size(0) && batch == v.size(0) && batch == kv_state.size(0) && k_state.size(0) == batch && out.size(0) == batch, "Differing batch sizes?");
    TORCH_CHECK(q.size(1) == d_state, "Q is d_state?");
    TORCH_CHECK(k.size(1) == d_state, "K is d_state?");
    TORCH_CHECK(v.size(1) == d_model, "V is d_model?");
    TORCH_CHECK(kv_state.size(1) == d_state && kv_state.size(2) == d_model);
    TORCH_CHECK(k_state.size(1) == d_state, "k_state is d_state");
    TORCH_CHECK(out.size(1) == d_model, "out is d_model (the size of v)");

    const int workers = 8;
    auto _run = [&]<typename H, typename T>() {
        auto threads = workers * gwt::WARP_SIZE;
        // unsigned long mem_size  = workers * 8 * 2 * d_model * sizeof(H);   
        // printf("[based_inference] Requesting %lu bytes of memory for %d (%d) workers for %d batches\n", mem_size, workers, threads, batch);

        auto stream_wrapper = at::cuda::getCurrentCUDAStream(q.device().index());
        cudaStream_t stream = stream_wrapper.stream();
        based_simple_ker<H,T><<<batch,threads,0,stream>>>(
                q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
                kv_state.data_ptr<T>(), k_state.data_ptr<T>(),
                out.data_ptr<T>());
    };
    DISPATCH(q, (_run.operator()<H,T>()););
}
