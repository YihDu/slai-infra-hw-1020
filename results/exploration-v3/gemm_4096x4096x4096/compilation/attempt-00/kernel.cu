#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void matmul_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) matmul_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_s = ((void*)((char*)buf_dyn_shmem + 16384));
  float acc[32];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)a_s)[(((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 524288) + (i_1 * 131072)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)b_s)[(((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((i_2 * 131072) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  for (int p = 0; p < 63; ++p) {
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      half_t A_local[32];
      half_t B_local[8];
      for (int ki = 0; ki < 4; ++ki) {
        #pragma unroll
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          tl::ptx_ldmatrix_x4((&(((half_t*)a_s)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_3 * 8)])));
        }
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)b_s)[(((((ki * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[0])));
        for (int i_4 = 0; i_4 < 4; ++i_4) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (i_4 * 8)), reinterpret_cast<const unsigned*>(A_local + (i_4 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((i_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_4 * 8)), reinterpret_cast<const unsigned*>(B_local + 4));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)a_s)[(((((i_5 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) * 524288) + (i_5 * 131072)) + ((((int)threadIdx.x) >> 3) * 4096)) + (p * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)b_s)[(((((i_6 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((p * 262144) + (i_6 * 131072)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_1[32];
    half_t B_local_1[8];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((half_t*)a_s)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[(i_7 * 8)])));
      }
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)b_s)[(((((ki_1 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[0])));
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (i_8 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((i_8 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 16; ++i_9) {
    *(float2*)(((float*)c_s) + (((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_9 >> 2) * 1024)) + ((i_9 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_9 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_9 * 2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 8; ++i_10) {
    float c_s_local_cast_1[4];
    half_t C_local_cast[4];
    *(float4*)(c_s_local_cast_1 + 0) = *(float4*)(((float*)c_s) + ((i_10 * 1024) + (((int)threadIdx.x) * 4)));
    uint2 __1;
    float4 v_ = *(float4*)(c_s_local_cast_1 + 0);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + (((((((int)blockIdx.y) * 524288) + (i_10 * 65536)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4))) = *(uint2*)(C_local_cast + 0);
  }
}

