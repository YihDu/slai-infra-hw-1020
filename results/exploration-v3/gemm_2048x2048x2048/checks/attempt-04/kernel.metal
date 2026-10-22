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
  threadgroup uchar buf_dyn_shmem[32768];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 8192));
  simdgroup_float8x8 acc[128];
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
  acc[64] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[65] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[66] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[67] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[68] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[69] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[70] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[71] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[72] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[73] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[74] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[75] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[76] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[77] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[78] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[79] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[80] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[81] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[82] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[83] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[84] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[85] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[86] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[87] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[88] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[89] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[90] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[91] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[92] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[93] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[94] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[95] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[96] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[97] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[98] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[99] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[100] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[101] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[102] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[103] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[104] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[105] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[106] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[107] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[108] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[109] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[110] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[111] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[112] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[113] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[114] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[115] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[116] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[117] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[118] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[119] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[120] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[121] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[122] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[123] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[124] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[125] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[126] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  acc[127] = make_filled_simdgroup_matrix<float, 8, 8>(0.000000e+00f);
  *(threadgroup uint4*)(a_s + (((int)threadIdx.x) * 8)) = A[((((((((int)blockIdx.y) >> 4) * 2097152) + (((int)blockIdx.x) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) / 8)];
  for (int i = 0; i < 2; ++i) {
    half broadcast_var = 0.000000e+00h;
    int condval_1;
    if (((((int)blockIdx.y) >> 4) == 1)) {
      condval_1 = (15 - (((int)blockIdx.y) & 15));
    } else {
      condval_1 = (((int)blockIdx.y) & 15);
    }
    int condval_2;
    if (((((int)blockIdx.y) >> 4) == 1)) {
      condval_2 = (15 - (((int)blockIdx.y) & 15));
    } else {
      condval_2 = (((int)blockIdx.y) & 15);
    }
    uint4 condval;
    if (((condval_1 < 16) && (0 <= condval_2))) {
      long condval_3;
      if (((((long)((int)blockIdx.y)) >> (long)4) == (long)1)) {
        condval_3 = ((long)15 - (((long)((int)blockIdx.y)) & (long)15));
      } else {
        condval_3 = (((long)((int)blockIdx.y)) & (long)15);
      }
      condval = B[(((((((long)i) * (long)32768) + ((((long)((int)threadIdx.x)) >> (long)4) * (long)2048)) + (condval_3 * (long)128)) + ((((long)((int)threadIdx.x)) & (long)15) * (long)8)) / (long)8)];
    } else {
      condval = uint4(as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)));
    }
    *(threadgroup uint4*)(b_s + ((i * 2048) + (((int)threadIdx.x) * 8))) = condval;
  }
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 63; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *(threadgroup uint4*)(a_s + ((((p + 1) & 1) * 2048) + (((int)threadIdx.x) * 8))) = A[((((((((((int)blockIdx.y) >> 4) * 2097152) + (((int)blockIdx.x) * 131072)) + ((((int)threadIdx.x) >> 2) * 2048)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32) / 8)];
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      half broadcast_var_1 = 0.000000e+00h;
      int condval_5;
      if (((((int)blockIdx.y) >> 4) == 1)) {
        condval_5 = (15 - (((int)blockIdx.y) & 15));
      } else {
        condval_5 = (((int)blockIdx.y) & 15);
      }
      int condval_6;
      if (((((int)blockIdx.y) >> 4) == 1)) {
        condval_6 = (15 - (((int)blockIdx.y) & 15));
      } else {
        condval_6 = (((int)blockIdx.y) & 15);
      }
      uint4 condval_4;
      if (((condval_5 < 16) && (0 <= condval_6))) {
        long condval_7;
        if (((((long)((int)blockIdx.y)) >> (long)4) == (long)1)) {
          condval_7 = ((long)15 - (((long)((int)blockIdx.y)) & (long)15));
        } else {
          condval_7 = (((long)((int)blockIdx.y)) & (long)15);
        }
        condval_4 = B[(((((((((long)p) * (long)65536) + (((long)i_1) * (long)32768)) + ((((long)((int)threadIdx.x)) >> (long)4) * (long)2048)) + (condval_7 * (long)128)) + ((((long)((int)threadIdx.x)) & (long)15) * (long)8)) + (long)65536) / (long)8)];
      } else {
        condval_4 = uint4(as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)));
      }
      *(threadgroup uint4*)(b_s + (((((p + 1) & 1) * 4096) + (i_1 * 2048)) + (((int)threadIdx.x) * 8))) = condval_4;
    }
    simdgroup_half8x8 A_local[2];
    simdgroup_half8x8 B_local[8];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        simdgroup_load(A_local[i_2], (&(a_s[(((((p & 1) * 2048) + ((((int)__simd_group_id) & 3) * 512)) + (i_2 * 256)) + (ki * 8))])), 32, 0, (bool)0);
      }
      #pragma clang loop unroll(disable)
      for (int j = 0; j < 8; ++j) {
        simdgroup_load(B_local[j], (&(b_s[(((((p & 1) * 4096) + (ki * 1024)) + ((((int)__simd_group_id) >> 2) * 64)) + (j * 8))])), 128, 0, (bool)0);
      }
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        #pragma clang loop unroll(disable)
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          simdgroup_multiply_accumulate(acc[((i_3 * 8) + j_1)], A_local[i_3], B_local[j_1], acc[((i_3 * 8) + j_1)]);
        }
      }
    }
  }
  simdgroup_half8x8 A_local_1[2];
  simdgroup_half8x8 B_local_1[8];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      simdgroup_load(A_local_1[i_4], (&(a_s[(((((((int)__simd_group_id) & 3) * 512) + (i_4 * 256)) + (ki_1 * 8)) + 2048)])), 32, 0, (bool)0);
    }
    #pragma clang loop unroll(disable)
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[((((ki_1 * 1024) + ((((int)__simd_group_id) >> 2) * 64)) + (j_2 * 8)) + 4096)])), 128, 0, (bool)0);
    }
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      #pragma clang loop unroll(disable)
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        simdgroup_multiply_accumulate(acc[((i_5 * 8) + j_3)], A_local_1[i_5], B_local_1[j_3], acc[((i_5 * 8) + j_3)]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64))])), 128, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 8)])), 128, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 16)])), 128, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 24)])), 128, 0, (bool)0);
  simdgroup_store(acc[4], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 32)])), 128, 0, (bool)0);
  simdgroup_store(acc[5], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 40)])), 128, 0, (bool)0);
  simdgroup_store(acc[6], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 48)])), 128, 0, (bool)0);
  simdgroup_store(acc[7], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 56)])), 128, 0, (bool)0);
  simdgroup_store(acc[8], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1024)])), 128, 0, (bool)0);
  simdgroup_store(acc[9], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1032)])), 128, 0, (bool)0);
  simdgroup_store(acc[10], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1040)])), 128, 0, (bool)0);
  simdgroup_store(acc[11], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1048)])), 128, 0, (bool)0);
  simdgroup_store(acc[12], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1056)])), 128, 0, (bool)0);
  simdgroup_store(acc[13], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1064)])), 128, 0, (bool)0);
  simdgroup_store(acc[14], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1072)])), 128, 0, (bool)0);
  simdgroup_store(acc[15], (&(c_s[((((((int)__simd_group_id) & 3) * 2048) + ((((int)__simd_group_id) >> 2) * 64)) + 1080)])), 128, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(disable)
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    int condval_8;
    if (((((int)blockIdx.y) >> 4) == 1)) {
      condval_8 = (15 - (((int)blockIdx.y) & 15));
    } else {
      condval_8 = (((int)blockIdx.y) & 15);
    }
    if (condval_8 < 16) {
      c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i_6 * 1024) + (((int)threadIdx.x) * 4)));
      C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
      int condval_9;
      if (((((int)blockIdx.y) >> 4) == 1)) {
        condval_9 = (15 - (((int)blockIdx.y) & 15));
      } else {
        condval_9 = (((int)blockIdx.y) & 15);
      }
      if (0 <= condval_9) {
        int condval_10;
        if (((((int)blockIdx.y) >> 4) == 1)) {
          condval_10 = (15 - (((int)blockIdx.y) & 15));
        } else {
          condval_10 = (((int)blockIdx.y) & 15);
        }
        C[((((((((((int)blockIdx.y) >> 4) * 2097152) + (((int)blockIdx.x) * 131072)) + (i_6 * 16384)) + (((int)__simd_group_id) * 2048)) + (condval_10 * 128)) + ((((int)threadIdx.x) & 31) * 4)) / 4)] = C_local_cast[0];
      }
    }
  }
}


