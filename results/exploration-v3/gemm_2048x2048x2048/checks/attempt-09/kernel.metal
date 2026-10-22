// Function: matmul_kernel
#include <metal_stdlib>
#include <metal_simdgroup>
#define TILELANG_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

[[kernel, max_total_threads_per_threadgroup(256)]] void matmul_kernel(  const device uint4* __restrict A [[ buffer(0) ]],
  const device uint4* __restrict B [[ buffer(1) ]],
  device half4* __restrict C [[ buffer(2) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 __gridDim [[threadgroups_per_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint __simd_group_id [[simdgroup_index_in_threadgroup]]
) {
  threadgroup uchar buf_dyn_shmem[16384];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 8192));
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
  *(threadgroup uint4*)(a_s + (((int)threadIdx.x) * 8)) = A[((((((((int)blockIdx.y) >> 3) * 1048576) + ((((int)blockIdx.x) & 7) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) / 8)];
  half broadcast_var = 0.000000e+00h;
  int condval_1;
  if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
    condval_1 = ((31 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 4));
  } else {
    condval_1 = (((((int)blockIdx.y) & 7) * 4) + (((int)blockIdx.x) >> 3));
  }
  int condval_2;
  if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
    condval_2 = ((31 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 4));
  } else {
    condval_2 = (((((int)blockIdx.y) & 7) * 4) + (((int)blockIdx.x) >> 3));
  }
  uint4 condval;
  if (((condval_1 < 32) && (0 <= condval_2))) {
    long condval_3;
    if ((((((long)((int)blockIdx.y)) & (long)15) >> (long)3) == (long)1)) {
      condval_3 = (((long)31 - (((long)((int)blockIdx.x)) >> (long)3)) - ((((long)((int)blockIdx.y)) & (long)7) * (long)4));
    } else {
      condval_3 = (((((long)((int)blockIdx.y)) & (long)7) * (long)4) + (((long)((int)blockIdx.x)) >> (long)3));
    }
    condval = B[(((((((long)((int)threadIdx.x)) >> (long)3) * (long)2048) + (condval_3 * (long)64)) + ((((long)((int)threadIdx.x)) & (long)7) * (long)8)) / (long)8)];
  } else {
    condval = uint4(as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)));
  }
  *(threadgroup uint4*)(b_s + (((int)threadIdx.x) * 8)) = condval;
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 63; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup uint4*)(a_s + ((((p + 1) & 1) * 2048) + (((int)threadIdx.x) * 8))) = A[((((((((((int)blockIdx.y) >> 3) * 1048576) + ((((int)blockIdx.x) & 7) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32) / 8)];
    half broadcast_var_1 = 0.000000e+00h;
    int condval_5;
    if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
      condval_5 = ((31 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 4));
    } else {
      condval_5 = (((((int)blockIdx.y) & 7) * 4) + (((int)blockIdx.x) >> 3));
    }
    int condval_6;
    if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
      condval_6 = ((31 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 4));
    } else {
      condval_6 = (((((int)blockIdx.y) & 7) * 4) + (((int)blockIdx.x) >> 3));
    }
    uint4 condval_4;
    if (((condval_5 < 32) && (0 <= condval_6))) {
      long condval_7;
      if ((((((long)((int)blockIdx.y)) & (long)15) >> (long)3) == (long)1)) {
        condval_7 = (((long)31 - (((long)((int)blockIdx.x)) >> (long)3)) - ((((long)((int)blockIdx.y)) & (long)7) * (long)4));
      } else {
        condval_7 = (((((long)((int)blockIdx.y)) & (long)7) * (long)4) + (((long)((int)blockIdx.x)) >> (long)3));
      }
      condval_4 = B[((((((((long)p) * (long)65536) + ((((long)((int)threadIdx.x)) >> (long)3) * (long)2048)) + (condval_7 * (long)64)) + ((((long)((int)threadIdx.x)) & (long)7) * (long)8)) + (long)65536) / (long)8)];
    } else {
      condval_4 = uint4(as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)));
    }
    *(threadgroup uint4*)(b_s + ((((p + 1) & 1) * 2048) + (((int)threadIdx.x) * 8))) = condval_4;
    simdgroup_half8x8 A_local[2];
    simdgroup_half8x8 B_local[4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ki = 0; ki < 4; ++ki) {
      for (int i = 0; i < 2; ++i) {
        simdgroup_load(A_local[i], (&(a_s[(((((p & 1) * 2048) + ((((int)__simd_group_id) & 3) * 512)) + (i * 256)) + (ki * 8))])), 32, 0, (bool)0);
      }
      for (int j = 0; j < 4; ++j) {
        simdgroup_load(B_local[j], (&(b_s[(((((p & 1) * 2048) + (ki * 512)) + ((((int)__simd_group_id) >> 2) * 32)) + (j * 8))])), 64, 0, (bool)0);
      }
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          simdgroup_multiply_accumulate(acc[((i_1 * 4) + j_1)], A_local[i_1], B_local[j_1], acc[((i_1 * 4) + j_1)]);
        }
      }
    }
  }
  simdgroup_half8x8 A_local_1[2];
  simdgroup_half8x8 B_local_1[4];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      simdgroup_load(A_local_1[i_2], (&(a_s[(((((((int)__simd_group_id) & 3) * 512) + (i_2 * 256)) + (ki_1 * 8)) + 2048)])), 32, 0, (bool)0);
    }
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[((((ki_1 * 512) + ((((int)__simd_group_id) >> 2) * 32)) + (j_2 * 8)) + 2048)])), 64, 0, (bool)0);
    }
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      for (int j_3 = 0; j_3 < 4; ++j_3) {
        simdgroup_multiply_accumulate(acc[((i_3 * 4) + j_3)], A_local_1[i_3], B_local_1[j_3], acc[((i_3 * 4) + j_3)]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32))])), 64, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32)) + 8)])), 64, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32)) + 16)])), 64, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32)) + 24)])), 64, 0, (bool)0);
  simdgroup_store(acc[4], (&(c_s[((((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32)) + 512)])), 64, 0, (bool)0);
  simdgroup_store(acc[5], (&(c_s[((((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32)) + 520)])), 64, 0, (bool)0);
  simdgroup_store(acc[6], (&(c_s[((((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32)) + 528)])), 64, 0, (bool)0);
  simdgroup_store(acc[7], (&(c_s[((((((int)__simd_group_id) & 3) * 1024) + ((((int)__simd_group_id) >> 2) * 32)) + 536)])), 64, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    int condval_8;
    if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
      condval_8 = ((31 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 4));
    } else {
      condval_8 = (((((int)blockIdx.y) & 7) * 4) + (((int)blockIdx.x) >> 3));
    }
    if (condval_8 < 32) {
      c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i_4 * 1024) + (((int)threadIdx.x) * 4)));
      C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
      int condval_9;
      if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
        condval_9 = ((31 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 4));
      } else {
        condval_9 = (((((int)blockIdx.y) & 7) * 4) + (((int)blockIdx.x) >> 3));
      }
      if (0 <= condval_9) {
        int condval_10;
        if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
          condval_10 = ((31 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 4));
        } else {
          condval_10 = (((((int)blockIdx.y) & 7) * 4) + (((int)blockIdx.x) >> 3));
        }
        C[((((((((((int)blockIdx.y) >> 3) * 1048576) + ((((int)blockIdx.x) & 7) * 131072)) + (i_4 * 32768)) + ((((int)threadIdx.x) >> 4) * 2048)) + (condval_10 * 64)) + ((((int)threadIdx.x) & 15) * 4)) / 4)] = C_local_cast[0];
      }
    }
  }
}


