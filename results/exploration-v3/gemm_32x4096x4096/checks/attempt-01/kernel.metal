// Function: matmul_kernel
#include <metal_stdlib>
#include <metal_simdgroup>
#define TILELANG_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

[[kernel, max_total_threads_per_threadgroup(256)]] void matmul_kernel(  const device half4* __restrict A [[ buffer(0) ]],
  const device uint4* __restrict B [[ buffer(1) ]],
  device half4* __restrict C [[ buffer(2) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 __gridDim [[threadgroups_per_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint __simd_group_id [[simdgroup_index_in_threadgroup]]
) {
  threadgroup uchar buf_dyn_shmem[20480];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 4096));
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
  *(threadgroup half4*)(a_s + (((int)threadIdx.x) * 4)) = A[(((((((int)blockIdx.x) & 1) * 65536) + ((((int)threadIdx.x) >> 4) * 4096)) + ((((int)threadIdx.x) & 15) * 4)) / 4)];
  for (int i = 0; i < 2; ++i) {
    *(threadgroup uint4*)(b_s + ((i * 2048) + (((int)threadIdx.x) * 8))) = B[((((((i * 131072) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 2048)) + ((((int)blockIdx.x) >> 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)) / 8)];
  }
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 63; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup half4*)(a_s + ((((p + 1) & 1) * 1024) + (((int)threadIdx.x) * 4))) = A[(((((((((int)blockIdx.x) & 1) * 65536) + ((((int)threadIdx.x) >> 4) * 4096)) + (p * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 64) / 4)];
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(threadgroup uint4*)(b_s + (((((p + 1) & 1) * 4096) + (i_1 * 2048)) + (((int)threadIdx.x) * 8))) = B[((((((((p * 262144) + (i_1 * 131072)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.y) * 2048)) + ((((int)blockIdx.x) >> 1) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144) / 8)];
    }
    simdgroup_half8x8 A_local[1];
    simdgroup_half8x8 B_local[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(disable)
    for (int ki = 0; ki < 8; ++ki) {
      simdgroup_load(A_local[0], (&(a_s[((((p & 1) * 1024) + ((((int)__simd_group_id) & 1) * 512)) + (ki * 8))])), 64, 0, (bool)0);
      for (int j = 0; j < 2; ++j) {
        simdgroup_load(B_local[j], (&(b_s[(((((p & 1) * 4096) + (ki * 512)) + ((((int)__simd_group_id) >> 1) * 16)) + (j * 8))])), 64, 0, (bool)0);
      }
      for (int j_1 = 0; j_1 < 2; ++j_1) {
        simdgroup_multiply_accumulate(acc[j_1], A_local[0], B_local[j_1], acc[j_1]);
      }
    }
  }
  simdgroup_half8x8 A_local_1[1];
  simdgroup_half8x8 B_local_1[2];
  #pragma clang loop unroll(disable)
  for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
    simdgroup_load(A_local_1[0], (&(a_s[((((((int)__simd_group_id) & 1) * 512) + (ki_1 * 8)) + 1024)])), 64, 0, (bool)0);
    for (int j_2 = 0; j_2 < 2; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[((((ki_1 * 512) + ((((int)__simd_group_id) >> 1) * 16)) + (j_2 * 8)) + 4096)])), 64, 0, (bool)0);
    }
    for (int j_3 = 0; j_3 < 2; ++j_3) {
      simdgroup_multiply_accumulate(acc[j_3], A_local_1[0], B_local_1[j_3], acc[j_3]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((((int)__simd_group_id) & 1) * 512) + ((((int)__simd_group_id) >> 1) * 16))])), 64, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((((int)__simd_group_id) & 1) * 512) + ((((int)__simd_group_id) >> 1) * 16)) + 8)])), 64, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  thread float4 c_s_local_cast_1[1];
  thread half4 C_local_cast[1];
  c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + (((int)threadIdx.x) * 4));
  C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
  C[(((((((((int)blockIdx.x) & 1) * 65536) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.y) * 2048)) + ((((int)blockIdx.x) >> 1) * 64)) + ((((int)threadIdx.x) & 15) * 4)) / 4)] = C_local_cast[0];
}


