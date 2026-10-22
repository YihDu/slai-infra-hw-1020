// Function: matmul_kernel
#include <metal_stdlib>
#include <metal_simdgroup>
#define TILELANG_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

[[kernel, max_total_threads_per_threadgroup(64)]] void matmul_kernel(  const device uint4* __restrict A [[ buffer(0) ]],
  const device uint4* __restrict B [[ buffer(1) ]],
  device half4* __restrict C [[ buffer(2) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 __gridDim [[threadgroups_per_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint __simd_group_id [[simdgroup_index_in_threadgroup]]
) {
  threadgroup uchar buf_dyn_shmem[30720];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 6144));
  simdgroup_float8x8 acc[16];
  acc[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[2] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[3] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[4] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[5] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[6] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[7] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[8] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[9] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[10] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[11] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[12] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[13] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[14] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[15] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  for (int i = 0; i < 2; ++i) {
    *(threadgroup uint4*)(a_s + ((i * 512) + (((int)threadIdx.x) * 8))) = A[((((((((int)blockIdx.x) & 1) * 65536) + (i * 32768)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)threadIdx.x) & 7) * 8)) / 8)];
  }
  #pragma clang loop unroll(disable)
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(threadgroup uint4*)(b_s + ((i_1 * 512) + (((int)threadIdx.x) * 8))) = B[((((((i_1 * 32768) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 2048)) + ((((int)blockIdx.x) >> 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)) / 8)];
  }
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(threadgroup uint4*)(a_s + (((i_2 * 512) + (((int)threadIdx.x) * 8)) + 1024)) = A[(((((((((int)blockIdx.x) & 1) * 65536) + (i_2 * 32768)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)threadIdx.x) & 7) * 8)) + 64) / 8)];
  }
  #pragma clang loop unroll(disable)
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    *(threadgroup uint4*)(b_s + (((i_3 * 512) + (((int)threadIdx.x) * 8)) + 4096)) = B[(((((((i_3 * 32768) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 2048)) + ((((int)blockIdx.x) >> 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144) / 8)];
  }
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 62; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      *(threadgroup uint4*)(a_s + (((((p + 2) % 3) * 1024) + (i_4 * 512)) + (((int)threadIdx.x) * 8))) = A[((((((((((int)blockIdx.x) & 1) * 65536) + (i_4 * 32768)) + ((((int)threadIdx.x) >> 3) * 4096)) + (p * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128) / 8)];
    }
    #pragma clang loop unroll(disable)
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      *(threadgroup uint4*)(b_s + (((((p + 2) % 3) * 4096) + (i_5 * 512)) + (((int)threadIdx.x) * 8))) = B[((((((((p * 262144) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 2048)) + ((((int)blockIdx.x) >> 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 524288) / 8)];
    }
    simdgroup_half8x8 A_local[1];
    simdgroup_half8x8 B_local[8];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(disable)
    for (int ki = 0; ki < 8; ++ki) {
      simdgroup_load(A_local[0], (&(a_s[((((p % 3) * 1024) + (((int)__simd_group_id) * 512)) + (ki * 8))])), 64, 0, (bool)0);
      #pragma clang loop unroll(disable)
      for (int j = 0; j < 8; ++j) {
        simdgroup_load(B_local[j], (&(b_s[((((p % 3) * 4096) + (ki * 512)) + (j * 8))])), 64, 0, (bool)0);
      }
      #pragma clang loop unroll(disable)
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        simdgroup_multiply_accumulate(acc[j_1], A_local[0], B_local[j_1], acc[j_1]);
      }
    }
  }
  simdgroup_half8x8 A_local_1[1];
  simdgroup_half8x8 B_local_1[8];
  #pragma clang loop unroll(disable)
  for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
    simdgroup_load(A_local_1[0], (&(a_s[(((((int)__simd_group_id) * 512) + (ki_1 * 8)) + 2048)])), 64, 0, (bool)0);
    #pragma clang loop unroll(disable)
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[(((ki_1 * 512) + (j_2 * 8)) + 8192)])), 64, 0, (bool)0);
    }
    #pragma clang loop unroll(disable)
    for (int j_3 = 0; j_3 < 8; ++j_3) {
      simdgroup_multiply_accumulate(acc[j_3], A_local_1[0], B_local_1[j_3], acc[j_3]);
    }
  }
  simdgroup_half8x8 A_local_2[1];
  simdgroup_half8x8 B_local_2[8];
  #pragma clang loop unroll(disable)
  for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
    simdgroup_load(A_local_2[0], (&(a_s[((((int)__simd_group_id) * 512) + (ki_2 * 8))])), 64, 0, (bool)0);
    #pragma clang loop unroll(disable)
    for (int j_4 = 0; j_4 < 8; ++j_4) {
      simdgroup_load(B_local_2[j_4], (&(b_s[((ki_2 * 512) + (j_4 * 8))])), 64, 0, (bool)0);
    }
    #pragma clang loop unroll(disable)
    for (int j_5 = 0; j_5 < 8; ++j_5) {
      simdgroup_multiply_accumulate(acc[j_5], A_local_2[0], B_local_2[j_5], acc[j_5]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((int)__simd_group_id) * 512)])), 64, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((int)__simd_group_id) * 512) + 8)])), 64, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((int)__simd_group_id) * 512) + 16)])), 64, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((int)__simd_group_id) * 512) + 24)])), 64, 0, (bool)0);
  simdgroup_store(acc[4], (&(c_s[((((int)__simd_group_id) * 512) + 32)])), 64, 0, (bool)0);
  simdgroup_store(acc[5], (&(c_s[((((int)__simd_group_id) * 512) + 40)])), 64, 0, (bool)0);
  simdgroup_store(acc[6], (&(c_s[((((int)__simd_group_id) * 512) + 48)])), 64, 0, (bool)0);
  simdgroup_store(acc[7], (&(c_s[((((int)__simd_group_id) * 512) + 56)])), 64, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i_6 * 256) + (((int)threadIdx.x) * 4)));
    C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
    C[((((((((((int)blockIdx.x) & 1) * 65536) + (i_6 * 16384)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 2048)) + ((((int)blockIdx.x) >> 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) / 4)] = C_local_cast[0];
  }
}


