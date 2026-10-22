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
extern "C" __global__ void __launch_bounds__(128, 1) matmul_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* a_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* c_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* b_s = ((void*)((char*)buf_dyn_shmem + 8192));
  float acc[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)a_s)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) >> 4) * 4194304) + ((((int)blockIdx.x) & 15) * 262144)) + (i_1 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    int64_t condval;
    if ((((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)4) == (int64_t)1)) {
      condval = (((int64_t)31 - (((int64_t)((int)blockIdx.x)) >> (int64_t)4)) - ((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2));
    } else {
      condval = (((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2) + (((int64_t)((int)blockIdx.x)) >> (int64_t)4));
    }
    int64_t condval_1;
    if ((((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)4) == (int64_t)1)) {
      condval_1 = (((int64_t)31 - (((int64_t)((int)blockIdx.x)) >> (int64_t)4)) - ((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2));
    } else {
      condval_1 = (((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2) + (((int64_t)((int)blockIdx.x)) >> (int64_t)4));
    }
        int condval_2;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_2 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_2 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    int condval_3;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_3 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_3 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    int condval_4;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_4 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_4 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    int condval_5;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_5 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_5 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
tl::cp_async_gs_conditional<16>((&(((half_t*)b_s)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((int64_t)i_2) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)4096)) + (condval_1 * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), ((((condval_2 < 32) && (0 <= condval_3)) && (condval_4 < 32)) && (0 <= condval_5)));
  }
  tl::cp_async_commit();
  for (int p = 0; p < 127; ++p) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      tl::cp_async_gs<16>((&(((half_t*)a_s)[(((((((p + 1) & 1) * 2048) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((((int)blockIdx.y) >> 4) * 4194304) + ((((int)blockIdx.x) & 15) * 262144)) + (i_3 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      int64_t condval_6;
      if ((((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)4) == (int64_t)1)) {
        condval_6 = (((int64_t)31 - (((int64_t)((int)blockIdx.x)) >> (int64_t)4)) - ((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2));
      } else {
        condval_6 = (((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2) + (((int64_t)((int)blockIdx.x)) >> (int64_t)4));
      }
      int64_t condval_7;
      if ((((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)4) == (int64_t)1)) {
        condval_7 = (((int64_t)31 - (((int64_t)((int)blockIdx.x)) >> (int64_t)4)) - ((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2));
      } else {
        condval_7 = (((((int64_t)((int)blockIdx.y)) & (int64_t)15) * (int64_t)2) + (((int64_t)((int)blockIdx.x)) >> (int64_t)4));
      }
            int condval_8;
      if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
        condval_8 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
      } else {
        condval_8 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
      }
      int condval_9;
      if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
        condval_9 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
      } else {
        condval_9 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
      }
      int condval_10;
      if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
        condval_10 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
      } else {
        condval_10 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
      }
      int condval_11;
      if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
        condval_11 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
      } else {
        condval_11 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
      }
tl::cp_async_gs_conditional<16>((&(((half_t*)b_s)[(((((((((p + 1) & 1) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((int64_t)p) * (int64_t)131072) + (((int64_t)i_4) * (int64_t)32768)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)4096)) + (condval_7 * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)131072)])), ((((condval_8 < 32) && (0 <= condval_9)) && (condval_10 < 32)) && (0 <= condval_11)));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local[16];
      half_t B_local[32];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)a_s)[(((((((p & 1) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_5 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_5 * 8)])));
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)b_s)[((((((((p & 1) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_6 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_6 * 8)])));
        }
        for (int i_7 = 0; i_7 < 2; ++i_7) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((i_7 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_7 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (((i_7 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_1[16];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)a_s)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_8 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), (&(A_local_1[(i_8 * 8)])));
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)b_s)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local_1[(i_9 * 8)])));
      }
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((i_10 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (((i_10 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_11 = 0; i_11 < 32; ++i_11) {
    *(float2*)(((float*)c_s) + (((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_11 >> 4) * 2048)) + ((i_11 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_11 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_11 * 2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 16; ++i_12) {
    float c_s_local_cast_1[4];
    half_t C_local_cast[4];
    int condval_12;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_12 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_12 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    if (condval_12 < 32) {
      *(float4*)(c_s_local_cast_1 + 0) = *(float4*)(((float*)c_s) + ((i_12 * 512) + (((int)threadIdx.x) * 4)));
      uint2 __1;
      float4 v_ = *(float4*)(c_s_local_cast_1 + 0);
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(C_local_cast + 0) = __1;
      int condval_13;
      if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
        condval_13 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
      } else {
        condval_13 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
      }
      if (0 <= condval_13) {
        int condval_14;
        if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
          condval_14 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
        } else {
          condval_14 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
        }
        int condval_15;
        if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
          condval_15 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
        } else {
          condval_15 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
        }
        *(uint2*)(C + (((((((((int)blockIdx.y) >> 4) * 4194304) + ((((int)blockIdx.x) & 15) * 262144)) + (i_12 * 16384)) + ((((int)threadIdx.x) >> 5) * 4096)) + (condval_15 * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(uint2*)(C_local_cast + 0);
      }
    }
  }
}

