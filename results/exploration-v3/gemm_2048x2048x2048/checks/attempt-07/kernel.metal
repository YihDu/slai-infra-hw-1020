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
  for (int i = 0; i < 4; ++i) {
    *(threadgroup uint4*)(a_s + ((i * 512) + (((int)threadIdx.x) * 8))) = A[(((((((((int)blockIdx.y) >> 3) * 1048576) + ((((int)blockIdx.x) & 7) * 131072)) + (i * 32768)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) / 8)];
  }
  #pragma clang loop unroll(disable)
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    half broadcast_var = 0.000000e+00h;
    int condval_1;
    if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
      condval_1 = ((15 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 2));
    } else {
      condval_1 = (((((int)blockIdx.y) & 7) * 2) + (((int)blockIdx.x) >> 3));
    }
    int condval_2;
    if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
      condval_2 = ((15 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 2));
    } else {
      condval_2 = (((((int)blockIdx.y) & 7) * 2) + (((int)blockIdx.x) >> 3));
    }
    uint4 condval;
    if (((condval_1 < 16) && (0 <= condval_2))) {
      long condval_3;
      if ((((((long)((int)blockIdx.y)) & (long)15) >> (long)3) == (long)1)) {
        condval_3 = (((long)15 - (((long)((int)blockIdx.x)) >> (long)3)) - ((((long)((int)blockIdx.y)) & (long)7) * (long)2));
      } else {
        condval_3 = (((((long)((int)blockIdx.y)) & (long)7) * (long)2) + (((long)((int)blockIdx.x)) >> (long)3));
      }
      condval = B[(((((((long)i_1) * (long)8192) + ((((long)((int)threadIdx.x)) >> (long)4) * (long)2048)) + (condval_3 * (long)128)) + ((((long)((int)threadIdx.x)) & (long)15) * (long)8)) / (long)8)];
    } else {
      condval = uint4(as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)), as_type<uint>(half2(broadcast_var, broadcast_var)));
    }
    *(threadgroup uint4*)(b_s + ((i_1 * 512) + (((int)threadIdx.x) * 8))) = condval;
  }
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 63; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(threadgroup uint4*)(a_s + (((((p + 1) & 1) * 2048) + (i_2 * 512)) + (((int)threadIdx.x) * 8))) = A[(((((((((((int)blockIdx.y) >> 3) * 1048576) + ((((int)blockIdx.x) & 7) * 131072)) + (i_2 * 32768)) + ((((int)threadIdx.x) >> 2) * 2048)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32) / 8)];
    }
    #pragma clang loop unroll(disable)
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      half broadcast_var_1 = 0.000000e+00h;
      int condval_5;
      if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
        condval_5 = ((15 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 2));
      } else {
        condval_5 = (((((int)blockIdx.y) & 7) * 2) + (((int)blockIdx.x) >> 3));
      }
      int condval_6;
      if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
        condval_6 = ((15 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 2));
      } else {
        condval_6 = (((((int)blockIdx.y) & 7) * 2) + (((int)blockIdx.x) >> 3));
      }
      uint4 condval_4;
      if (((condval_5 < 16) && (0 <= condval_6))) {
        long condval_7;
        if ((((((long)((int)blockIdx.y)) & (long)15) >> (long)3) == (long)1)) {
          condval_7 = (((long)15 - (((long)((int)blockIdx.x)) >> (long)3)) - ((((long)((int)blockIdx.y)) & (long)7) * (long)2));
        } else {
          condval_7 = (((((long)((int)blockIdx.y)) & (long)7) * (long)2) + (((long)((int)blockIdx.x)) >> (long)3));
        }
        condval_4 = B[(((((((((long)p) * (long)65536) + (((long)i_3) * (long)8192)) + ((((long)((int)threadIdx.x)) >> (long)4) * (long)2048)) + (condval_7 * (long)128)) + ((((long)((int)threadIdx.x)) & (long)15) * (long)8)) + (long)65536) / (long)8)];
      } else {
        condval_4 = uint4(as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)), as_type<uint>(half2(broadcast_var_1, broadcast_var_1)));
      }
      *(threadgroup uint4*)(b_s + (((((p + 1) & 1) * 4096) + (i_3 * 512)) + (((int)threadIdx.x) * 8))) = condval_4;
    }
    simdgroup_half8x8 A_local[4];
    simdgroup_half8x8 B_local[16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        simdgroup_load(A_local[i_4], (&(a_s[(((((p & 1) * 2048) + (((int)__simd_group_id) * 1024)) + (i_4 * 256)) + (ki * 8))])), 32, 0, (bool)0);
      }
      #pragma clang loop unroll(disable)
      for (int j = 0; j < 16; ++j) {
        simdgroup_load(B_local[j], (&(b_s[((((p & 1) * 4096) + (ki * 1024)) + (j * 8))])), 128, 0, (bool)0);
      }
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        #pragma clang loop unroll(disable)
        for (int j_1 = 0; j_1 < 16; ++j_1) {
          simdgroup_multiply_accumulate(acc[((i_5 * 16) + j_1)], A_local[i_5], B_local[j_1], acc[((i_5 * 16) + j_1)]);
        }
      }
    }
  }
  simdgroup_half8x8 A_local_1[4];
  simdgroup_half8x8 B_local_1[16];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      simdgroup_load(A_local_1[i_6], (&(a_s[((((((int)__simd_group_id) * 1024) + (i_6 * 256)) + (ki_1 * 8)) + 2048)])), 32, 0, (bool)0);
    }
    #pragma clang loop unroll(disable)
    for (int j_2 = 0; j_2 < 16; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[(((ki_1 * 1024) + (j_2 * 8)) + 4096)])), 128, 0, (bool)0);
    }
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      #pragma clang loop unroll(disable)
      for (int j_3 = 0; j_3 < 16; ++j_3) {
        simdgroup_multiply_accumulate(acc[((i_7 * 16) + j_3)], A_local_1[i_7], B_local_1[j_3], acc[((i_7 * 16) + j_3)]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((int)__simd_group_id) * 4096)])), 128, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((int)__simd_group_id) * 4096) + 8)])), 128, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((int)__simd_group_id) * 4096) + 16)])), 128, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((int)__simd_group_id) * 4096) + 24)])), 128, 0, (bool)0);
  simdgroup_store(acc[4], (&(c_s[((((int)__simd_group_id) * 4096) + 32)])), 128, 0, (bool)0);
  simdgroup_store(acc[5], (&(c_s[((((int)__simd_group_id) * 4096) + 40)])), 128, 0, (bool)0);
  simdgroup_store(acc[6], (&(c_s[((((int)__simd_group_id) * 4096) + 48)])), 128, 0, (bool)0);
  simdgroup_store(acc[7], (&(c_s[((((int)__simd_group_id) * 4096) + 56)])), 128, 0, (bool)0);
  simdgroup_store(acc[8], (&(c_s[((((int)__simd_group_id) * 4096) + 64)])), 128, 0, (bool)0);
  simdgroup_store(acc[9], (&(c_s[((((int)__simd_group_id) * 4096) + 72)])), 128, 0, (bool)0);
  simdgroup_store(acc[10], (&(c_s[((((int)__simd_group_id) * 4096) + 80)])), 128, 0, (bool)0);
  simdgroup_store(acc[11], (&(c_s[((((int)__simd_group_id) * 4096) + 88)])), 128, 0, (bool)0);
  simdgroup_store(acc[12], (&(c_s[((((int)__simd_group_id) * 4096) + 96)])), 128, 0, (bool)0);
  simdgroup_store(acc[13], (&(c_s[((((int)__simd_group_id) * 4096) + 104)])), 128, 0, (bool)0);
  simdgroup_store(acc[14], (&(c_s[((((int)__simd_group_id) * 4096) + 112)])), 128, 0, (bool)0);
  simdgroup_store(acc[15], (&(c_s[((((int)__simd_group_id) * 4096) + 120)])), 128, 0, (bool)0);
  simdgroup_store(acc[16], (&(c_s[((((int)__simd_group_id) * 4096) + 1024)])), 128, 0, (bool)0);
  simdgroup_store(acc[17], (&(c_s[((((int)__simd_group_id) * 4096) + 1032)])), 128, 0, (bool)0);
  simdgroup_store(acc[18], (&(c_s[((((int)__simd_group_id) * 4096) + 1040)])), 128, 0, (bool)0);
  simdgroup_store(acc[19], (&(c_s[((((int)__simd_group_id) * 4096) + 1048)])), 128, 0, (bool)0);
  simdgroup_store(acc[20], (&(c_s[((((int)__simd_group_id) * 4096) + 1056)])), 128, 0, (bool)0);
  simdgroup_store(acc[21], (&(c_s[((((int)__simd_group_id) * 4096) + 1064)])), 128, 0, (bool)0);
  simdgroup_store(acc[22], (&(c_s[((((int)__simd_group_id) * 4096) + 1072)])), 128, 0, (bool)0);
  simdgroup_store(acc[23], (&(c_s[((((int)__simd_group_id) * 4096) + 1080)])), 128, 0, (bool)0);
  simdgroup_store(acc[24], (&(c_s[((((int)__simd_group_id) * 4096) + 1088)])), 128, 0, (bool)0);
  simdgroup_store(acc[25], (&(c_s[((((int)__simd_group_id) * 4096) + 1096)])), 128, 0, (bool)0);
  simdgroup_store(acc[26], (&(c_s[((((int)__simd_group_id) * 4096) + 1104)])), 128, 0, (bool)0);
  simdgroup_store(acc[27], (&(c_s[((((int)__simd_group_id) * 4096) + 1112)])), 128, 0, (bool)0);
  simdgroup_store(acc[28], (&(c_s[((((int)__simd_group_id) * 4096) + 1120)])), 128, 0, (bool)0);
  simdgroup_store(acc[29], (&(c_s[((((int)__simd_group_id) * 4096) + 1128)])), 128, 0, (bool)0);
  simdgroup_store(acc[30], (&(c_s[((((int)__simd_group_id) * 4096) + 1136)])), 128, 0, (bool)0);
  simdgroup_store(acc[31], (&(c_s[((((int)__simd_group_id) * 4096) + 1144)])), 128, 0, (bool)0);
  simdgroup_store(acc[32], (&(c_s[((((int)__simd_group_id) * 4096) + 2048)])), 128, 0, (bool)0);
  simdgroup_store(acc[33], (&(c_s[((((int)__simd_group_id) * 4096) + 2056)])), 128, 0, (bool)0);
  simdgroup_store(acc[34], (&(c_s[((((int)__simd_group_id) * 4096) + 2064)])), 128, 0, (bool)0);
  simdgroup_store(acc[35], (&(c_s[((((int)__simd_group_id) * 4096) + 2072)])), 128, 0, (bool)0);
  simdgroup_store(acc[36], (&(c_s[((((int)__simd_group_id) * 4096) + 2080)])), 128, 0, (bool)0);
  simdgroup_store(acc[37], (&(c_s[((((int)__simd_group_id) * 4096) + 2088)])), 128, 0, (bool)0);
  simdgroup_store(acc[38], (&(c_s[((((int)__simd_group_id) * 4096) + 2096)])), 128, 0, (bool)0);
  simdgroup_store(acc[39], (&(c_s[((((int)__simd_group_id) * 4096) + 2104)])), 128, 0, (bool)0);
  simdgroup_store(acc[40], (&(c_s[((((int)__simd_group_id) * 4096) + 2112)])), 128, 0, (bool)0);
  simdgroup_store(acc[41], (&(c_s[((((int)__simd_group_id) * 4096) + 2120)])), 128, 0, (bool)0);
  simdgroup_store(acc[42], (&(c_s[((((int)__simd_group_id) * 4096) + 2128)])), 128, 0, (bool)0);
  simdgroup_store(acc[43], (&(c_s[((((int)__simd_group_id) * 4096) + 2136)])), 128, 0, (bool)0);
  simdgroup_store(acc[44], (&(c_s[((((int)__simd_group_id) * 4096) + 2144)])), 128, 0, (bool)0);
  simdgroup_store(acc[45], (&(c_s[((((int)__simd_group_id) * 4096) + 2152)])), 128, 0, (bool)0);
  simdgroup_store(acc[46], (&(c_s[((((int)__simd_group_id) * 4096) + 2160)])), 128, 0, (bool)0);
  simdgroup_store(acc[47], (&(c_s[((((int)__simd_group_id) * 4096) + 2168)])), 128, 0, (bool)0);
  simdgroup_store(acc[48], (&(c_s[((((int)__simd_group_id) * 4096) + 3072)])), 128, 0, (bool)0);
  simdgroup_store(acc[49], (&(c_s[((((int)__simd_group_id) * 4096) + 3080)])), 128, 0, (bool)0);
  simdgroup_store(acc[50], (&(c_s[((((int)__simd_group_id) * 4096) + 3088)])), 128, 0, (bool)0);
  simdgroup_store(acc[51], (&(c_s[((((int)__simd_group_id) * 4096) + 3096)])), 128, 0, (bool)0);
  simdgroup_store(acc[52], (&(c_s[((((int)__simd_group_id) * 4096) + 3104)])), 128, 0, (bool)0);
  simdgroup_store(acc[53], (&(c_s[((((int)__simd_group_id) * 4096) + 3112)])), 128, 0, (bool)0);
  simdgroup_store(acc[54], (&(c_s[((((int)__simd_group_id) * 4096) + 3120)])), 128, 0, (bool)0);
  simdgroup_store(acc[55], (&(c_s[((((int)__simd_group_id) * 4096) + 3128)])), 128, 0, (bool)0);
  simdgroup_store(acc[56], (&(c_s[((((int)__simd_group_id) * 4096) + 3136)])), 128, 0, (bool)0);
  simdgroup_store(acc[57], (&(c_s[((((int)__simd_group_id) * 4096) + 3144)])), 128, 0, (bool)0);
  simdgroup_store(acc[58], (&(c_s[((((int)__simd_group_id) * 4096) + 3152)])), 128, 0, (bool)0);
  simdgroup_store(acc[59], (&(c_s[((((int)__simd_group_id) * 4096) + 3160)])), 128, 0, (bool)0);
  simdgroup_store(acc[60], (&(c_s[((((int)__simd_group_id) * 4096) + 3168)])), 128, 0, (bool)0);
  simdgroup_store(acc[61], (&(c_s[((((int)__simd_group_id) * 4096) + 3176)])), 128, 0, (bool)0);
  simdgroup_store(acc[62], (&(c_s[((((int)__simd_group_id) * 4096) + 3184)])), 128, 0, (bool)0);
  simdgroup_store(acc[63], (&(c_s[((((int)__simd_group_id) * 4096) + 3192)])), 128, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(disable)
  for (int i_8 = 0; i_8 < 32; ++i_8) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    int condval_8;
    if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
      condval_8 = ((15 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 2));
    } else {
      condval_8 = (((((int)blockIdx.y) & 7) * 2) + (((int)blockIdx.x) >> 3));
    }
    if (condval_8 < 16) {
      c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i_8 * 256) + (((int)threadIdx.x) * 4)));
      C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
      int condval_9;
      if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
        condval_9 = ((15 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 2));
      } else {
        condval_9 = (((((int)blockIdx.y) & 7) * 2) + (((int)blockIdx.x) >> 3));
      }
      if (0 <= condval_9) {
        int condval_10;
        if ((((((int)blockIdx.y) & 15) >> 3) == 1)) {
          condval_10 = ((15 - (((int)blockIdx.x) >> 3)) - ((((int)blockIdx.y) & 7) * 2));
        } else {
          condval_10 = (((((int)blockIdx.y) & 7) * 2) + (((int)blockIdx.x) >> 3));
        }
        C[((((((((((int)blockIdx.y) >> 3) * 1048576) + ((((int)blockIdx.x) & 7) * 131072)) + (i_8 * 4096)) + (((int)__simd_group_id) * 2048)) + (condval_10 * 128)) + ((((int)threadIdx.x) & 31) * 4)) / 4)] = C_local_cast[0];
      }
    }
  }
}


