// Function: matmul_kernel
#include <metal_stdlib>
#include <metal_simdgroup>
#define TILELANG_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

[[kernel, max_total_threads_per_threadgroup(128)]] void matmul_kernel(  const device uint4* __restrict A [[ buffer(0) ]],
  const device uint4* __restrict B [[ buffer(1) ]],
  device half4* __restrict C [[ buffer(2) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 __gridDim [[threadgroups_per_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint __simd_group_id [[simdgroup_index_in_threadgroup]]
) {
  threadgroup uchar buf_dyn_shmem[24576];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 12288));
  simdgroup_float8x8 acc[64];
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
  acc[32] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[33] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[34] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[35] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[36] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[37] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[38] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[39] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[40] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[41] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[42] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[43] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[44] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[45] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[46] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[47] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[48] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[49] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[50] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[51] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[52] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[53] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[54] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[55] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[56] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[57] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[58] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[59] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[60] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[61] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[62] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[63] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  for (int i = 0; i < 2; ++i) {
    *(threadgroup uint4*)(a_s + ((i * 1024) + (((int)threadIdx.x) * 8))) = A[(((((((int)blockIdx.y) * 262144) + (i * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) / 8)];
  }
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(threadgroup uint4*)(b_s + ((i_1 * 1024) + (((int)threadIdx.x) * 8))) = B[(((((i_1 * 65536) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) / 8)];
  }
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(threadgroup uint4*)(a_s + (((i_2 * 1024) + (((int)threadIdx.x) * 8)) + 2048)) = A[((((((((int)blockIdx.y) * 262144) + (i_2 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32) / 8)];
  }
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(threadgroup uint4*)(b_s + (((i_3 * 1024) + (((int)threadIdx.x) * 8)) + 2048)) = B[((((((i_3 * 65536) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 131072) / 8)];
  }
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 126; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      *(threadgroup uint4*)(a_s + (((((p + 2) % 3) * 2048) + (i_4 * 1024)) + (((int)threadIdx.x) * 8))) = A[(((((((((int)blockIdx.y) * 262144) + (i_4 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64) / 8)];
    }
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      *(threadgroup uint4*)(b_s + (((((p + 2) % 3) * 2048) + (i_5 * 1024)) + (((int)threadIdx.x) * 8))) = B[(((((((p * 131072) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 262144) / 8)];
    }
    simdgroup_half8x8 A_local[4];
    simdgroup_half8x8 B_local[4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        simdgroup_load(A_local[i_6], (&(a_s[(((((p % 3) * 2048) + ((((int)__simd_group_id) & 1) * 1024)) + (i_6 * 256)) + (ki * 8))])), 32, 0, (bool)0);
      }
      for (int j = 0; j < 4; ++j) {
        simdgroup_load(B_local[j], (&(b_s[(((((p % 3) * 2048) + (ki * 512)) + ((((int)__simd_group_id) >> 1) * 32)) + (j * 8))])), 64, 0, (bool)0);
      }
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          simdgroup_multiply_accumulate(acc[((i_7 * 4) + j_1)], A_local[i_7], B_local[j_1], acc[((i_7 * 4) + j_1)]);
        }
      }
    }
  }
  simdgroup_half8x8 A_local_1[4];
  simdgroup_half8x8 B_local_1[4];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      simdgroup_load(A_local_1[i_8], (&(a_s[((((((int)__simd_group_id) & 1) * 1024) + (i_8 * 256)) + (ki_1 * 8))])), 32, 0, (bool)0);
    }
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[(((ki_1 * 512) + ((((int)__simd_group_id) >> 1) * 32)) + (j_2 * 8))])), 64, 0, (bool)0);
    }
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      for (int j_3 = 0; j_3 < 4; ++j_3) {
        simdgroup_multiply_accumulate(acc[((i_9 * 4) + j_3)], A_local_1[i_9], B_local_1[j_3], acc[((i_9 * 4) + j_3)]);
      }
    }
  }
  simdgroup_half8x8 A_local_2[4];
  simdgroup_half8x8 B_local_2[4];
  for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      simdgroup_load(A_local_2[i_10], (&(a_s[(((((((int)__simd_group_id) & 1) * 1024) + (i_10 * 256)) + (ki_2 * 8)) + 2048)])), 32, 0, (bool)0);
    }
    for (int j_4 = 0; j_4 < 4; ++j_4) {
      simdgroup_load(B_local_2[j_4], (&(b_s[((((ki_2 * 512) + ((((int)__simd_group_id) >> 1) * 32)) + (j_4 * 8)) + 2048)])), 64, 0, (bool)0);
    }
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      for (int j_5 = 0; j_5 < 4; ++j_5) {
        simdgroup_multiply_accumulate(acc[((i_11 * 4) + j_5)], A_local_2[i_11], B_local_2[j_5], acc[((i_11 * 4) + j_5)]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32))])), 64, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 8)])), 64, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 16)])), 64, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 24)])), 64, 0, (bool)0);
  simdgroup_store(acc[4], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 512)])), 64, 0, (bool)0);
  simdgroup_store(acc[5], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 520)])), 64, 0, (bool)0);
  simdgroup_store(acc[6], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 528)])), 64, 0, (bool)0);
  simdgroup_store(acc[7], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 536)])), 64, 0, (bool)0);
  simdgroup_store(acc[8], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1024)])), 64, 0, (bool)0);
  simdgroup_store(acc[9], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1032)])), 64, 0, (bool)0);
  simdgroup_store(acc[10], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1040)])), 64, 0, (bool)0);
  simdgroup_store(acc[11], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1048)])), 64, 0, (bool)0);
  simdgroup_store(acc[12], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1536)])), 64, 0, (bool)0);
  simdgroup_store(acc[13], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1544)])), 64, 0, (bool)0);
  simdgroup_store(acc[14], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1552)])), 64, 0, (bool)0);
  simdgroup_store(acc[15], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 32)) + 1560)])), 64, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(disable)
  for (int i_12 = 0; i_12 < 8; ++i_12) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i_12 * 512) + (((int)threadIdx.x) * 4)));
    C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
    C[((((((((int)blockIdx.y) * 262144) + (i_12 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) / 4)] = C_local_cast[0];
  }
}


