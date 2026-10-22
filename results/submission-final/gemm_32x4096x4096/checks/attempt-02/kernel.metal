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
  threadgroup uchar buf_dyn_shmem[18432];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 6144));
  simdgroup_float8x8 acc[32];
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
  acc[16] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[17] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[18] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[19] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[20] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[21] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[22] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[23] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[24] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[25] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[26] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[27] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[28] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[29] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[30] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[31] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  *(threadgroup half4*)(a_s + (((int)threadIdx.x) * 4)) = A[((((((int)threadIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) & 7) * 4)) / 4)];
  *(threadgroup uint4*)(b_s + (((int)threadIdx.x) * 8)) = B[(((((((int)threadIdx.x) >> 3) * 4096) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) / 8)];
  *(threadgroup half4*)(a_s + ((((int)threadIdx.x) * 4) + 1024)) = A[(((((((int)threadIdx.x) >> 3) * 4096) + ((((int)threadIdx.x) & 7) * 4)) + 32) / 4)];
  *(threadgroup uint4*)(b_s + ((((int)threadIdx.x) * 8) + 2048)) = B[((((((((int)threadIdx.x) >> 3) * 4096) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 131072) / 8)];
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 126; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup half4*)(a_s + ((((p + 2) % 3) * 1024) + (((int)threadIdx.x) * 4))) = A[((((((((int)threadIdx.x) >> 3) * 4096) + (p * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 64) / 4)];
    *(threadgroup uint4*)(b_s + ((((p + 2) % 3) * 2048) + (((int)threadIdx.x) * 8))) = B[((((((p * 131072) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144) / 8)];
    simdgroup_half8x8 A_local[1];
    simdgroup_half8x8 B_local[4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ki = 0; ki < 4; ++ki) {
      simdgroup_load(A_local[0], (&(a_s[((((p % 3) * 1024) + ((((int)__simd_group_id) & 3) * 256)) + (ki * 8))])), 32, 0, (bool)0);
      for (int j = 0; j < 4; ++j) {
        simdgroup_load(B_local[j], (&(b_s[(((((p % 3) * 2048) + (ki * 512)) + ((((int)__simd_group_id) >> 2) * 32)) + (j * 8))])), 64, 0, (bool)0);
      }
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        simdgroup_multiply_accumulate(acc[j_1], A_local[0], B_local[j_1], acc[j_1]);
      }
    }
  }
  simdgroup_half8x8 A_local_1[1];
  simdgroup_half8x8 B_local_1[4];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    simdgroup_load(A_local_1[0], (&(a_s[(((((int)__simd_group_id) & 3) * 256) + (ki_1 * 8))])), 32, 0, (bool)0);
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[(((ki_1 * 512) + ((((int)__simd_group_id) >> 2) * 32)) + (j_2 * 8))])), 64, 0, (bool)0);
    }
    for (int j_3 = 0; j_3 < 4; ++j_3) {
      simdgroup_multiply_accumulate(acc[j_3], A_local_1[0], B_local_1[j_3], acc[j_3]);
    }
  }
  simdgroup_half8x8 A_local_2[1];
  simdgroup_half8x8 B_local_2[4];
  for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
    simdgroup_load(A_local_2[0], (&(a_s[((((((int)__simd_group_id) & 3) * 256) + (ki_2 * 8)) + 1024)])), 32, 0, (bool)0);
    for (int j_4 = 0; j_4 < 4; ++j_4) {
      simdgroup_load(B_local_2[j_4], (&(b_s[((((ki_2 * 512) + ((((int)__simd_group_id) >> 2) * 32)) + (j_4 * 8)) + 2048)])), 64, 0, (bool)0);
    }
    for (int j_5 = 0; j_5 < 4; ++j_5) {
      simdgroup_multiply_accumulate(acc[j_5], A_local_2[0], B_local_2[j_5], acc[j_5]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((((int)__simd_group_id) & 3) * 512) + ((((int)__simd_group_id) >> 2) * 32))])), 64, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((((int)__simd_group_id) & 3) * 512) + ((((int)__simd_group_id) >> 2) * 32)) + 8)])), 64, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((((int)__simd_group_id) & 3) * 512) + ((((int)__simd_group_id) >> 2) * 32)) + 16)])), 64, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((((int)__simd_group_id) & 3) * 512) + ((((int)__simd_group_id) >> 2) * 32)) + 24)])), 64, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 2; ++i) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i * 1024) + (((int)threadIdx.x) * 4)));
    C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
    C[(((((i * 65536) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) / 4)] = C_local_cast[0];
  }
}


