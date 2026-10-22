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
  threadgroup uchar buf_dyn_shmem[30720];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 6144));
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
  *(threadgroup uint4*)(a_s + (((int)threadIdx.x) * 8)) = A[((((((((int)blockIdx.y) >> 4) * 2097152) + ((((int)blockIdx.x) & 15) * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) / 8)];
  for (int i = 0; i < 4; ++i) {
    half broadcast_var = 0.000000e+00h;
    int condval_1;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_1 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_1 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    int condval_2;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_2 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_2 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    uint4 condval;
    if (((condval_1 < 32) && (0 <= condval_2))) {
      long condval_3;
      if ((((((long)((int)blockIdx.y)) & (long)31) >> (long)4) == (long)1)) {
        condval_3 = (((long)31 - (((long)((int)blockIdx.x)) >> (long)4)) - ((((long)((int)blockIdx.y)) & (long)15) * (long)2));
      } else {
        condval_3 = (((((long)((int)blockIdx.y)) & (long)15) * (long)2) + (((long)((int)blockIdx.x)) >> (long)4));
      }
      condval = B[(((((((long)i) * (long)32768) + ((((long)((int)threadIdx.x)) >> (long)4) * (long)4096)) + (condval_3 * (long)128)) + ((((long)((int)threadIdx.x)) & (long)15) * (long)8)) / (long)8)];
    } else {
      condval = uint4(as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)));
    }
    *(threadgroup uint4*)(b_s + ((i * 1024) + (((int)threadIdx.x) * 8))) = condval;
  }
  *(threadgroup uint4*)(a_s + ((((int)threadIdx.x) * 8) + 1024)) = A[(((((((((int)blockIdx.y) >> 4) * 2097152) + ((((int)blockIdx.x) & 15) * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) + 32) / 8)];
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    half broadcast_var_1 = 0.000000e+00h;
    int condval_5;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_5 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_5 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    int condval_6;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_6 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_6 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    uint4 condval_4;
    if (((condval_5 < 32) && (0 <= condval_6))) {
      long condval_7;
      if ((((((long)((int)blockIdx.y)) & (long)31) >> (long)4) == (long)1)) {
        condval_7 = (((long)31 - (((long)((int)blockIdx.x)) >> (long)4)) - ((((long)((int)blockIdx.y)) & (long)15) * (long)2));
      } else {
        condval_7 = (((((long)((int)blockIdx.y)) & (long)15) * (long)2) + (((long)((int)blockIdx.x)) >> (long)4));
      }
      condval_4 = B[((((((((long)i_1) * (long)32768) + ((((long)((int)threadIdx.x)) >> (long)4) * (long)4096)) + (condval_7 * (long)128)) + ((((long)((int)threadIdx.x)) & (long)15) * (long)8)) + (long)131072) / (long)8)];
    } else {
      condval_4 = uint4(as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)));
    }
    *(threadgroup uint4*)(b_s + (((i_1 * 1024) + (((int)threadIdx.x) * 8)) + 4096)) = condval_4;
  }
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 126; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup uint4*)(a_s + ((((p + 2) % 3) * 1024) + (((int)threadIdx.x) * 8))) = A[((((((((((int)blockIdx.y) >> 4) * 2097152) + ((((int)blockIdx.x) & 15) * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64) / 8)];
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      half broadcast_var_2 = 0.000000e+00h;
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
      uint4 condval_8;
      if (((condval_9 < 32) && (0 <= condval_10))) {
        long condval_11;
        if ((((((long)((int)blockIdx.y)) & (long)31) >> (long)4) == (long)1)) {
          condval_11 = (((long)31 - (((long)((int)blockIdx.x)) >> (long)4)) - ((((long)((int)blockIdx.y)) & (long)15) * (long)2));
        } else {
          condval_11 = (((((long)((int)blockIdx.y)) & (long)15) * (long)2) + (((long)((int)blockIdx.x)) >> (long)4));
        }
        condval_8 = B[(((((((((long)p) * (long)131072) + (((long)i_2) * (long)32768)) + ((((long)((int)threadIdx.x)) >> (long)4) * (long)4096)) + (condval_11 * (long)128)) + ((((long)((int)threadIdx.x)) & (long)15) * (long)8)) + (long)262144) / (long)8)];
      } else {
        condval_8 = uint4(as_type<uint>(half2(broadcast_var_2, broadcast_var_2)), as_type<uint>(half2(broadcast_var_2, broadcast_var_2)), as_type<uint>(half2(broadcast_var_2, broadcast_var_2)), as_type<uint>(half2(broadcast_var_2, broadcast_var_2)));
      }
      *(threadgroup uint4*)(b_s + (((((p + 2) % 3) * 4096) + (i_2 * 1024)) + (((int)threadIdx.x) * 8))) = condval_8;
    }
    simdgroup_half8x8 A_local[2];
    simdgroup_half8x8 B_local[8];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        simdgroup_load(A_local[i_3], (&(a_s[(((((p % 3) * 1024) + ((((int)__simd_group_id) & 1) * 512)) + (i_3 * 256)) + (ki * 8))])), 32, 0, (bool)0);
      }
      #pragma clang loop unroll(disable)
      for (int j = 0; j < 8; ++j) {
        simdgroup_load(B_local[j], (&(b_s[(((((p % 3) * 4096) + (ki * 1024)) + ((((int)__simd_group_id) >> 1) * 64)) + (j * 8))])), 128, 0, (bool)0);
      }
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        #pragma clang loop unroll(disable)
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          simdgroup_multiply_accumulate(acc[((i_4 * 8) + j_1)], A_local[i_4], B_local[j_1], acc[((i_4 * 8) + j_1)]);
        }
      }
    }
  }
  simdgroup_half8x8 A_local_1[2];
  simdgroup_half8x8 B_local_1[8];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      simdgroup_load(A_local_1[i_5], (&(a_s[((((((int)__simd_group_id) & 1) * 512) + (i_5 * 256)) + (ki_1 * 8))])), 32, 0, (bool)0);
    }
    #pragma clang loop unroll(disable)
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[(((ki_1 * 1024) + ((((int)__simd_group_id) >> 1) * 64)) + (j_2 * 8))])), 128, 0, (bool)0);
    }
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      #pragma clang loop unroll(disable)
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        simdgroup_multiply_accumulate(acc[((i_6 * 8) + j_3)], A_local_1[i_6], B_local_1[j_3], acc[((i_6 * 8) + j_3)]);
      }
    }
  }
  simdgroup_half8x8 A_local_2[2];
  simdgroup_half8x8 B_local_2[8];
  for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      simdgroup_load(A_local_2[i_7], (&(a_s[(((((((int)__simd_group_id) & 1) * 512) + (i_7 * 256)) + (ki_2 * 8)) + 1024)])), 32, 0, (bool)0);
    }
    #pragma clang loop unroll(disable)
    for (int j_4 = 0; j_4 < 8; ++j_4) {
      simdgroup_load(B_local_2[j_4], (&(b_s[((((ki_2 * 1024) + ((((int)__simd_group_id) >> 1) * 64)) + (j_4 * 8)) + 4096)])), 128, 0, (bool)0);
    }
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      #pragma clang loop unroll(disable)
      for (int j_5 = 0; j_5 < 8; ++j_5) {
        simdgroup_multiply_accumulate(acc[((i_8 * 8) + j_5)], A_local_2[i_8], B_local_2[j_5], acc[((i_8 * 8) + j_5)]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64))])), 128, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 8)])), 128, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 16)])), 128, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 24)])), 128, 0, (bool)0);
  simdgroup_store(acc[4], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 32)])), 128, 0, (bool)0);
  simdgroup_store(acc[5], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 40)])), 128, 0, (bool)0);
  simdgroup_store(acc[6], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 48)])), 128, 0, (bool)0);
  simdgroup_store(acc[7], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 56)])), 128, 0, (bool)0);
  simdgroup_store(acc[8], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1024)])), 128, 0, (bool)0);
  simdgroup_store(acc[9], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1032)])), 128, 0, (bool)0);
  simdgroup_store(acc[10], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1040)])), 128, 0, (bool)0);
  simdgroup_store(acc[11], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1048)])), 128, 0, (bool)0);
  simdgroup_store(acc[12], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1056)])), 128, 0, (bool)0);
  simdgroup_store(acc[13], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1064)])), 128, 0, (bool)0);
  simdgroup_store(acc[14], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1072)])), 128, 0, (bool)0);
  simdgroup_store(acc[15], (&(c_s[((((((int)__simd_group_id) & 1) * 2048) + ((((int)__simd_group_id) >> 1) * 64)) + 1080)])), 128, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(disable)
  for (int i_9 = 0; i_9 < 8; ++i_9) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    int condval_12;
    if ((((((int)blockIdx.y) & 31) >> 4) == 1)) {
      condval_12 = ((31 - (((int)blockIdx.x) >> 4)) - ((((int)blockIdx.y) & 15) * 2));
    } else {
      condval_12 = (((((int)blockIdx.y) & 15) * 2) + (((int)blockIdx.x) >> 4));
    }
    if (condval_12 < 32) {
      c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i_9 * 512) + (((int)threadIdx.x) * 4)));
      C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
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
        C[((((((((((int)blockIdx.y) >> 4) * 2097152) + ((((int)blockIdx.x) & 15) * 131072)) + (i_9 * 16384)) + (((int)__simd_group_id) * 4096)) + (condval_14 * 128)) + ((((int)threadIdx.x) & 31) * 4)) / 4)] = C_local_cast[0];
      }
    }
  }
}


