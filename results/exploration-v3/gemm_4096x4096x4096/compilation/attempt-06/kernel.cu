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
  void* b_s = ((void*)((char*)buf_dyn_shmem + 2048));
  float acc[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs<16>((&(((half_t*)a_s)[((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) >> 5) * 4194304) + ((((int)blockIdx.y) & 1) * 2097152)) + (((int)blockIdx.x) * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8))])));
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    int64_t condval;
    if ((((((int64_t)((int)blockIdx.y)) & (int64_t)63) >> (int64_t)5) == (int64_t)1)) {
      condval = ((int64_t)15 - ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1));
    } else {
      condval = ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1);
    }
    int64_t condval_1;
    if ((((((int64_t)((int)blockIdx.y)) & (int64_t)63) >> (int64_t)5) == (int64_t)1)) {
      condval_1 = ((int64_t)15 - ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1));
    } else {
      condval_1 = ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1);
    }
        int condval_2;
    if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
      condval_2 = (15 - ((((int)blockIdx.y) & 31) >> 1));
    } else {
      condval_2 = ((((int)blockIdx.y) & 31) >> 1);
    }
    int condval_3;
    if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
      condval_3 = (15 - ((((int)blockIdx.y) & 31) >> 1));
    } else {
      condval_3 = ((((int)blockIdx.y) & 31) >> 1);
    }
    int condval_4;
    if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
      condval_4 = (15 - ((((int)blockIdx.y) & 31) >> 1));
    } else {
      condval_4 = ((((int)blockIdx.y) & 31) >> 1);
    }
    int condval_5;
    if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
      condval_5 = (15 - ((((int)blockIdx.y) & 31) >> 1));
    } else {
      condval_5 = ((((int)blockIdx.y) & 31) >> 1);
    }
tl::cp_async_gs_conditional<16>((&(((half_t*)b_s)[((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_1 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((int64_t)i_1) * (int64_t)16384) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)4096)) + (condval_1 * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8))])), ((((condval_2 < 16) && (0 <= condval_3)) && (condval_4 < 16)) && (0 <= condval_5)));
  }
  tl::cp_async_commit();
  for (int p = 0; p < 127; ++p) {
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[64];
      for (int ki = 0; ki < 2; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)a_s)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; ++i_2) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)b_s)[((((((((((int)threadIdx.x) >> 6) * 4096) + ((i_2 >> 2) * 2048)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_2 * 8)])));
        }
        for (int j = 0; j < 8; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    tl::cp_async_gs<16>((&(((half_t*)a_s)[((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((((int)blockIdx.y) >> 5) * 4194304) + ((((int)blockIdx.y) & 1) * 2097152)) + (((int)blockIdx.x) * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      int64_t condval_6;
      if ((((((int64_t)((int)blockIdx.y)) & (int64_t)63) >> (int64_t)5) == (int64_t)1)) {
        condval_6 = ((int64_t)15 - ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1));
      } else {
        condval_6 = ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1);
      }
      int64_t condval_7;
      if ((((((int64_t)((int)blockIdx.y)) & (int64_t)63) >> (int64_t)5) == (int64_t)1)) {
        condval_7 = ((int64_t)15 - ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1));
      } else {
        condval_7 = ((((int64_t)((int)blockIdx.y)) & (int64_t)31) >> (int64_t)1);
      }
            int condval_8;
      if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
        condval_8 = (15 - ((((int)blockIdx.y) & 31) >> 1));
      } else {
        condval_8 = ((((int)blockIdx.y) & 31) >> 1);
      }
      int condval_9;
      if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
        condval_9 = (15 - ((((int)blockIdx.y) & 31) >> 1));
      } else {
        condval_9 = ((((int)blockIdx.y) & 31) >> 1);
      }
      int condval_10;
      if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
        condval_10 = (15 - ((((int)blockIdx.y) & 31) >> 1));
      } else {
        condval_10 = ((((int)blockIdx.y) & 31) >> 1);
      }
      int condval_11;
      if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
        condval_11 = (15 - ((((int)blockIdx.y) & 31) >> 1));
      } else {
        condval_11 = ((((int)blockIdx.y) & 31) >> 1);
      }
tl::cp_async_gs_conditional<16>((&(((half_t*)b_s)[((((((((((int)threadIdx.x) & 31) >> 3) * 2048) + (i_3 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((int64_t)p) * (int64_t)131072) + (((int64_t)i_3) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)4096)) + (condval_7 * (int64_t)256)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)8)) + (int64_t)131072)])), ((((condval_8 < 16) && (0 <= condval_9)) && (condval_10 < 16)) && (0 <= condval_11)));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_1[64];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((half_t*)a_s)[((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)b_s)[((((((((((int)threadIdx.x) >> 6) * 4096) + ((i_4 >> 2) * 2048)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_4 * 8)])));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    *(float2*)(((float*)c_s) + ((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + ((i_5 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + ((((int)threadIdx.x) >> 6) * 128)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc + (i_5 * 2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    float c_s_local_cast_1[4];
    half_t C_local_cast[4];
    int condval_12;
    if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
      condval_12 = (15 - ((((int)blockIdx.y) & 31) >> 1));
    } else {
      condval_12 = ((((int)blockIdx.y) & 31) >> 1);
    }
    if (condval_12 < 16) {
      *(float4*)(c_s_local_cast_1 + 0) = *(float4*)(((float*)c_s) + ((i_6 * 512) + (((int)threadIdx.x) * 4)));
      uint2 __1;
      float4 v_ = *(float4*)(c_s_local_cast_1 + 0);
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(C_local_cast + 0) = __1;
      int condval_13;
      if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
        condval_13 = (15 - ((((int)blockIdx.y) & 31) >> 1));
      } else {
        condval_13 = ((((int)blockIdx.y) & 31) >> 1);
      }
      if (0 <= condval_13) {
        int condval_14;
        if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
          condval_14 = (15 - ((((int)blockIdx.y) & 31) >> 1));
        } else {
          condval_14 = ((((int)blockIdx.y) & 31) >> 1);
        }
        int condval_15;
        if ((((((int)blockIdx.y) & 63) >> 5) == 1)) {
          condval_15 = (15 - ((((int)blockIdx.y) & 31) >> 1));
        } else {
          condval_15 = ((((int)blockIdx.y) & 31) >> 1);
        }
        *(uint2*)(C + ((((((((((int)blockIdx.y) >> 5) * 4194304) + ((((int)blockIdx.y) & 1) * 2097152)) + (((int)blockIdx.x) * 131072)) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 6) * 4096)) + (condval_15 * 256)) + ((((int)threadIdx.x) & 63) * 4))) = *(uint2*)(C_local_cast + 0);
      }
    }
  }
}

