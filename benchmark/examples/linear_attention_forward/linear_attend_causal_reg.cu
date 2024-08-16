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


// Compute A0.
// We are computing V.cumsum(dim=0) in this example (Across the sequence)
// We first compute the local cumulative sum.
// Each has their local copy of V, we have to add in two elements
// 1. the preceding a0 from the last iteration (Stored in total_a0)
// 2. We need to compute a cumulative sum across these tiles.
// To handle 1, we add in total_a0 to a0
template<typename T> 
__device__
void tb_cumsum(gwt::tile<T> (&dst)[gwt::nWarps], 
                gwt::vec<gwt::vec_desc<T::width, typename T::T>> &total, // tile should have same width as the tile and same type.
                const gwt::tile<T> (&src)[gwt::nWarps],  const int nThreads = 256) {
    auto col = threadIdx.x % (gwt::TILE_DIM*T::width);
    
    // Threads are assigned to cols, and then go sequentially through the
    // all rows in the warps
    __syncthreads();
    for(auto col = threadIdx.x; col < gwt::TILE_DIM*T::width; col+= nThreads) {
        // this is resonsible for this column value.
        typename T::T v = total.data[col];
        for(auto w = 0; w < gwt::nWarps; w++) {
                  typename T::T *_dst = dst[w].raw_ptr();
            const typename T::T *_src = src[w].ptr();
            auto idx = col; 
        
            for(auto r = 0; r < T::rows; r++, idx += T::_row_stride) {
                v += _src[idx];
                _dst[idx] = v;
            }
        }
        total.data[col] = v;
    } 
}

// We write the local copy, and we want to compute a cumulative sum:
// 1. we need to add in the A0 that we computed in the last loop
// 2. we need the A1 fragments computed from the preceding war.

// To handle both, we do a cumulative sum. 1 is handled by warp  adding
// to its copy and letting the cumulative sum take care of it.
// At this stage, a1 has the "preceding" a1 for each warp
// and total_a1 is the next stage of what we need to build.
template<typename T> 
__device__
void tb_cumsum_delay_tiles_inplace(gwt::tile<T> (&x)[gwt::nWarps], gwt::tile<T> &total, const int nThreads = 256) {
    auto col = threadIdx.x % (gwt::TILE_DIM*T::width);
    auto row = threadIdx.x / (gwt::TILE_DIM*T::width); 
    __syncthreads(); 
    auto idx = row*T::_row_stride+col;
    assert(nThreads % (gwt::TILE_DIM * T::width) == 0);
    auto rows_per_block = nThreads / (gwt::TILE_DIM*T::width);
    auto row_skip       = rows_per_block * T::_row_stride;

    for(auto h = 0; h < T::height; h++) {
        for(auto rows = 0; rows < rows_per_block; rows ++, idx += row_skip) {
            typename T::T t = x[0].tile_start(h,0)[idx];
            // The "delay" happens here. We store the history in the first warp, total.
            // Then the cumulative sum is delayed by 1 warp (which corresponds to the history)
            x[0].raw_tile_start(h,0)[idx] = total.tile_start(h,0)[idx];
            t += total.tile_start(h,0)[idx];

            for(int wrp = 1; wrp < gwt::nWarps; wrp++) {
                    typename T::T t1 = x[wrp].tile_start(h,0)[idx];
                    x[wrp].raw_tile_start(h,0)[idx] = t;        
                    t += t1;
            }
            total.raw_tile_start(h,0)[idx] = t; // store the full count in Y
        } 
    }
   
}

template<typename T> 
__device__
void reduce_tile_tiles(gwt::tile<T> &dst, const gwt::tile<T> (&src)[gwt::nWarps], const int nThreads = 256) {
    auto col = threadIdx.x % (gwt::TILE_DIM*T::width);
    auto row = threadIdx.x / (gwt::TILE_DIM*T::width); 
    __syncthreads(); 
    auto idx = row*T::_row_stride+col;
    assert(nThreads % (gwt::TILE_DIM * T::width) == 0);
    auto rows_per_block = nThreads / (gwt::TILE_DIM*T::width);
    auto row_skip       = rows_per_block * T::_row_stride;

    for(auto h = 0; h < T::height; h++) {
        for(auto rows = 0; rows < rows_per_block; rows ++, idx += row_skip) {
            typename T::T t = src[0].tile_start(h,0)[idx]; 
            for(int wrp = 1; wrp < gwt::nWarps; wrp++) {t += src[wrp].tile_start(h,0)[idx];}
            dst.raw_tile_start(h,0)[idx] += t;
        } 
    }
   
}

template <typename H, typename T, bool _debug_build>
__global__
void a012_compute_ker(int n, int d, int dv, const T* __q, const T* __k, 
                                 const T* __v, T* __y, T* __a0, T* __a1, T* __a1y) { 

    auto warpid = gwt::warp_id();
    auto lane   = gwt::laneid();
    const int workers = gwt::nWarps;

    const H *_q   = reinterpret_cast<const H*>(__q)+blockIdx.x*(n*d);
    const H *_k   = reinterpret_cast<const H*>(__k)+blockIdx.x*(n*d);
    const H *_v   = reinterpret_cast<const H*>(__v)+blockIdx.x*(n*dv);
          H *_y   = reinterpret_cast<H*>(__y)+blockIdx.x*(n*dv);
    
    // Debugging Data structures
    H *_a0  = _debug_build ? reinterpret_cast<H*>(__a0)+blockIdx.x*(n*dv) : NULL;
    H *_a1  = _debug_build ? reinterpret_cast<H*>(__a1)+blockIdx.x*(n*dv) : NULL;
    H *_a1y = _debug_build ? reinterpret_cast<H*>(__a1y)+blockIdx.x*(n*dv) : NULL;

    using   tile_desc = gwt::tile_desc<1,1,8,H>;
    typedef gwt::tile<tile_desc> tile_;
    using  tile_desc_1x4 = gwt::tile_desc<1,4,8,H>;
    typedef gwt::tile<tile_desc_1x4> tile_1x4;
    
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    int *_pshm = &__shm[0];
    tile_(&q) [2][workers]     = gwt::shm_advance<tile_,2,workers>(_pshm);  
    tile_(&k) [2][workers]     = gwt::shm_advance<tile_,2,workers>(_pshm);
    tile_1x4(&v) [2][workers]  = gwt::shm_advance<tile_1x4,2,workers>(_pshm); 
    tile_1x4(&y) [workers]     = gwt::shm_advance<tile_1x4,workers>(_pshm);
    tile_1x4(&ty) [workers]    = gwt::shm_advance<tile_1x4,workers>(_pshm);
    tile_1x4(&a0) [workers]    = gwt::shm_advance<tile_1x4,workers>(_pshm);
    tile_1x4(&a1) [workers]    = gwt::shm_advance<tile_1x4,workers>(_pshm);

    __shared__ gwt::vec<typename gwt::vec_desc<4,H>> total_a0;
    __shared__ tile_1x4 total_a1;
    // a2 is stored in register throughout

    // register fragments
    const static auto &rt    = reg1x1_fl;
    const static auto &rt1x4 = reg1x4_fl;
    using _rtd_qk = reg_tile::reg_tile_desc<1,1>;
    using _rtd_v  = reg_tile::reg_tile_desc<1,4>;

    
    // Registers per thread for fragments.
    register typename _rtd_qk::row_frag qj0, qj1, kj0, kj1;
    register typename _rtd_qk::row_frag qfrag, qkfrag;
    register typename _rtd_qk::col_frag kfrag; 
    register typename _rtd_qk::accum temp_accum; 
    
    register typename _rtd_v::col_frag A2j0, A2j1, vfrag;
    register typename _rtd_v::accum A2j0_accum, A2j1_accum, o_accum, qA2_accum;
    // Register for a1
    register typename _rtd_qk::row_frag qk_a1_f;
    register typename _rtd_qk::accum qk_a1; 
    register typename _rtd_v::accum a1_accum, a1_out;
    register typename _rtd_v::row_frag a1_frag;
    register typename _rtd_v::col_frag a1_col_frag; 
     
    // Pipeline handlers and barriers
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> qkv_barrier;
    if (threadIdx.x == 0) {init(&qkv_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    block.sync(); // Make sure no gets to the barrier before its initialized.

    // elements
    const int qk_tile_elements = tile_desc::nElements;
    const int  v_tile_elements = tile_desc_1x4::nElements; 
    auto n_tiles  = n/gwt::TILE_DIM;
    auto n_blocks = n_tiles/workers;
    assert(n_tiles % workers == 0);

    gwt::load_async(q[tic][warpid], _q + warpid*qk_tile_elements, d,  qkv_barrier);
    gwt::load_async(k[tic][warpid], _k + warpid*qk_tile_elements, d,  qkv_barrier);
    gwt::load_async(v[tic][warpid], _v + warpid*v_tile_elements , dv, qkv_barrier);
    // Load in a bunch of QKV as we go. 
                
    // Set the tiles and accumulators to 0.
    rt1x4.zero(A2j0);
    rt1x4.zero(A2j1);
    rt1x4.zero(A2j0_accum);
    rt1x4.zero(A2j1_accum);

    // a1 zeros
    rt1x4.zero(a1_accum);
    if(warpid == 0) { // zero the small elements
        gwt::zero(total_a1);
        gwt::zero(total_a0);
    }
    __syncthreads();

    for(auto cur_block = 0; cur_block < n_blocks; cur_block++, tic ^= 1, toc ^= 1) {
        // Handle IO and loading
        qkv_barrier.arrive_and_wait(); 

        // Kick off the next block load.
        if(cur_block < n_blocks - 1) {
            auto next_idx = (cur_block + 1)*workers + warpid; 
            gwt::load_async(q[toc][warpid], _q + next_idx * qk_tile_elements, d, qkv_barrier);
            gwt::load_async(k[toc][warpid], _k + next_idx * qk_tile_elements, d, qkv_barrier);
            gwt::load_async(v[toc][warpid], _v + next_idx * v_tile_elements, dv, qkv_barrier);
        } 
        
        // We first handle the causal portion, the diagonal elements.
        // 1. We multiply (QK.T) on the diagonal tiles.
        // 2. Entry-wise square this
        // 3. Multiply by V.
        // Do the multiplication (qk)^2@V
        // and store the result in y[warpid]
        rt.tile_to_frag<tile_desc>(qfrag, q[tic][warpid]);
        // note we want an outer product here of Q and K, so we load K transposed from ocl.
        rt.tile_to_frag_col<tile_desc>(kfrag, k[tic][warpid]);
        rt.transpose_inplace(kfrag);
        
        rt.zero(temp_accum);
        rt.smm<1,1,1>(qfrag, kfrag, temp_accum);
        rt.make_causal(temp_accum);
        rt.copy(qk_a1, temp_accum);
        rt.mul(temp_accum, temp_accum); // square it, since this is the A2 term.
        
        rt1x4.tile_to_frag_col<tile_desc_1x4>(vfrag, v[tic][warpid]);
        rt1x4.zero(o_accum);

        // Compute the a0 portion. 
        // We are computing V.cumsum(dim=0) in this example (Across the sequence)
        // We first compute the local cumulative sum.
        // Each has their local copy of V, we have to add in two elements
        // 1. the preceding a0 from the last iteration (Stored in total_a0)
        // 2. We need to compute a cumulative sum across these tiles.
        // To handle 1, we add in total_a0 to a0
        tb_cumsum(a0, total_a0, v[tic]);
        __syncthreads();
        // ******************************
        // compute the a1 output portion
        // Qc@A1 + make_causal(Qc@Ktc)@Vc
        rt1x4.zero(a1_out);
        rt.accum_to_frag(qk_a1_f, qk_a1);
        rt1x4.smm<1,4,1>(qk_a1_f, vfrag, o_accum);
        
        // This is updating our local slice.
        // This is the update for A1 "after" our slice, i.e., containing everything it has seen.
        // A1 += Kt[:,whole_slice]@V[whole_slice]
        rt1x4.zero(a1_accum);
        _rtd_qk::row_frag rkfrag;
        rt.col_frag_to_row_frag(rkfrag, kfrag);
        rt1x4.smm<1,4,1>(rkfrag, vfrag, a1_accum);

        // We write the local copy, and we want to compute a cumulative sum:
        // 1. we need to add in the A0 that we computed in the last loop
        // 2. we need the A1 fragments computed from the preceding war.
        //
        // To handle both, we do a cumulative sum. 1 is handled by warp  adding
        // to its copy and letting the cumulative sum take care of it.
        rt1x4.accum_to_tile_bf16(a1[warpid], a1_accum);
        // At this stage, a1 has the "preceding" a1 for each warp
        // and total_a1 is the next stage of what we need to build.
        tb_cumsum_delay_tiles_inplace(a1, total_a1); 
        __syncthreads(); // need the writes to a1 to finish.
        // Now, each warp loads a1[warpid] into a1_col_frag for the multiplication 
        // this captures all the history add it to v_temp_accum
        rt1x4.tile_to_frag_col(a1_col_frag, a1[warpid]);
        rt1x4.smm<1,4,1>(qfrag, a1_col_frag, o_accum);
        // Now v_temp accum contains the whole part of a1y
        
        // from above o_accum holds + causal(QK)@V + Q@A1.
        // this computes += causal(QK)**2@V/2 
        rt.mul_accum_const(temp_accum, 0.5);
        rt.accum_to_frag(qkfrag, temp_accum);
        rt1x4.smm<1,4,1>(qkfrag, vfrag, o_accum);

        // Copy in the the a0 portion
        // the a1 and a2 portions are in o_accum
        gwt::copy(y[warpid], a0[warpid]);
        rt1x4.sum_tile_bf16_accum(y[warpid], o_accum);
        if(_debug_build && blockIdx.x == 0) { gwt::simple_print_tile("Y", y[warpid]); }

        // ********************************************************
        // ** This is the A2 non-diag caseand handles the update **
        // ********************************************************
        // This is the in-shared-mem portion We keep A2 in register spread
        // across the warps. Each warp has a 2 fragments of q and k and 1
        // fragment of v in memory.
        // * The indexing is below, but these are the outer products. 
        __syncthreads();
        // At this point, y[0].. y[read_block-1] contains the "diagonal" blocks
        // of all the outputs.
        // * We keep A2[2*warp], A2[2*warp+1] in register.
        // * Each computes their local portion of Q[j,:]*Q*A2
        // * Stores it back in ty[warpid]
        // This is hard-coded to A2 having dimension 16.
        for(auto blk = 0; blk < workers; blk++) { 
            
            // This computes
            // Q[j]@A2[j] for j=0,dots,15.
            // The "history"
            rt.tile_to_frag(qj0, q[tic][warpid]);
            rt.copy(qj1, qj0); // faster than reloading?

            // We store Q_j <- Q[:,j]*Q
            rt.mul_col_slice(qj0[0][0], 2*warpid);
            rt.mul_col_slice(qj1[0][0], 2*warpid+1);

            // Compute qj, a2j portion
            rt1x4.zero(qA2_accum);
            rt1x4.smm<1,4,1>(qj0, A2j0, qA2_accum);
            rt1x4.smm<1,4,1>(qj1, A2j1, qA2_accum);
            rt1x4.mul_accum_const(qA2_accum, 0.5f);
            rt1x4.copy_tile_bf16_accum(ty[warpid], qA2_accum);
            
            reduce_tile_tiles(y[blk], ty);
            __syncthreads();
            if(_debug_build && warpid == 1 && blockIdx.x == 0) { gwt::simple_print_tile("y[blk] AFTER", y[blk]);}

            // ****************************
            // ***** Update the State *****
            // ****************************
            // Update state for next round only needed if there is more work.
            // Now we update A2
            // now load the copies of K. These are 256/2 = 128 registers/ 32 = 4 registers per.
            rt.tile_to_frag(kj0, k[tic][blk]);
            rt.transpose_inplace(kj0);
            rt.copy(kj1, kj0); 
            rt.mul_row_slice(kj0[0][0], 2*warpid);
            rt.mul_row_slice(kj1[0][0], 2*warpid+1);

            // Compute the A2[j] update and put it back in the register
            rt1x4.tile_to_frag_col(vfrag, v[tic][blk]);
            rt1x4.smm<1,4,1>(kj0, vfrag, A2j0_accum);
            rt1x4.accum_to_frag_col(A2j0, A2j0_accum);
            
            rt1x4.smm<1,4,1>(kj1, vfrag, A2j1_accum);
            rt1x4.accum_to_frag_col(A2j1, A2j1_accum);
            // // Note we have to reduce across threads here to get the results
            // **** End the update of A2 ***
        }
        // handle the write back
        __syncthreads();
        gwt::store(_y + (cur_block * workers + warpid)*v_tile_elements, y[warpid], dv);
    }
}

void
a012_compute_debug(torch::Tensor q, torch::Tensor k, 
     torch::Tensor v, torch::Tensor o, torch::Tensor a0, torch::Tensor a1, torch::Tensor a1y) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    CHECK_INPUT(a0);
    CHECK_INPUT(a1);
    CHECK_INPUT(a1y);

    auto batch = q.size(0);
    auto head  = q.size(1);
    auto n     = q.size(2);
    auto d     = q.size(3);
    auto dv    = v.size(3);
    bool k_same = true, o_same = true;
    for(auto i = 0; i < 4; i++) { 
        k_same &= q.size(i) == k.size(i);
        o_same &= v.size(i) == o.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "Q and K should be same size");
    TORCH_CHECK(o_same, "V and O should be same size");

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "O is a Bfloat");

    TORCH_CHECK(a0.scalar_type() == c10::ScalarType::BFloat16, "a0 is a Bfloat");
    TORCH_CHECK(a1.scalar_type() == c10::ScalarType::BFloat16, "a1 is a Bfloat");
    TORCH_CHECK(a1y.scalar_type() == c10::ScalarType::BFloat16, "a1 is a Bfloat");

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    constexpr bool _debug_build = true;
    const int workers = 8;


    using   tile_desc = gwt::tile_desc<1,1,8,H>;
    typedef gwt::tile<tile_desc> tile_;
    using  tile_desc_1x4 = gwt::tile_desc<1,4,8,H>;
    typedef gwt::tile<tile_desc_1x4> tile_1x4;

    // q,k,v, and o are all double buffered
    unsigned long mem_size  =  2*2*workers*sizeof(tile_); // q, k and v are double buffered.
                  mem_size +=    2*workers*sizeof(tile_1x4);
                  mem_size += (workers+workers)*sizeof(tile_1x4);
                  mem_size += 2*workers*sizeof(tile_1x4); // a0 and a1y

    TORCH_CHECK(n % (workers*gwt::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times stored fragments");
    
    auto threads = workers * gwt::WARP_SIZE;
    // printf("[a012_compute] Requesting %lu bytes of memory for %d (%d) workers\n", mem_size, workers, threads);
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
             a012_compute_ker<H, T, _debug_build>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    a012_compute_ker<H,T,_debug_build><<<batch*head,threads,mem_size>>>(n, d, dv, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
         o.data_ptr<T>(), a0.data_ptr<T>(), a1.data_ptr<T>(), a1y.data_ptr<T>());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


void
a012_compute(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);

    
    auto batch = q.size(0);
    auto head  = q.size(1);
    auto n     = q.size(2);
    auto d     = q.size(3);
    auto dv    = v.size(3);
    bool k_same = true, o_same = true;
    for(auto i = 0; i < 4; i++) { 
        k_same &= q.size(i) == k.size(i);
        o_same &= v.size(i) == o.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "Q and K should be same size");
    TORCH_CHECK(o_same, "V and O should be same size");

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "O is a Bfloat");

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    constexpr bool _debug_build = false;
    const int workers = 8;


    using   tile_desc = gwt::tile_desc<1,1,8,H>;
    typedef gwt::tile<tile_desc> tile_;
    using  tile_desc_1x4 = gwt::tile_desc<1,4,8,H>;
    typedef gwt::tile<tile_desc_1x4> tile_1x4;

    // q,k,v, and o are all double buffered
    unsigned long mem_size  =  2*2*workers*sizeof(tile_); // q, k and v are double buffered.
                  mem_size +=    2*workers*sizeof(tile_1x4);
                  mem_size += (workers+workers)*sizeof(tile_1x4);
                  mem_size += 2*workers*sizeof(tile_1x4); // a0 and a1y

    TORCH_CHECK(n % (workers*gwt::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times stored fragments");
    
    auto threads = workers * gwt::WARP_SIZE;
    // printf("[a012_compute] Requesting %lu bytes of memory for %d (%d) workers\n", mem_size, workers, threads);
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
             a012_compute_ker<H, T, _debug_build>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    a012_compute_ker<H,T,false><<<batch*head,threads,mem_size>>>(n, d, dv, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
          o.data_ptr<T>(), NULL, NULL, NULL);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
