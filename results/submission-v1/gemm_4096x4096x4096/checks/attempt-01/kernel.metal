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
  threadgroup uchar buf_dyn_shmem[32768];
  threadgroup half* a_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup float* c_s = (threadgroup float*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 0));
  threadgroup half* b_s = (threadgroup half*)((threadgroup void*)((threadgroup char*)buf_dyn_shmem + 16384));
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
    *(threadgroup uint4*)(a_s + ((i * 1024) + (((int)threadIdx.x) * 8))) = A[(((((((int)blockIdx.y) * 524288) + (i * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + ((((int)threadIdx.x) & 3) * 8)) / 8)];
  }
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(threadgroup uint4*)(b_s + ((i_1 * 1024) + (((int)threadIdx.x) * 8))) = B[(((((i_1 * 65536) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) / 8)];
  }
  #pragma clang loop unroll(disable)
  for (int p = 0; p < 127; ++p) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      *(threadgroup uint4*)(a_s + (((((p + 1) & 1) * 4096) + (i_2 * 1024)) + (((int)threadIdx.x) * 8))) = A[(((((((((int)blockIdx.y) * 524288) + (i_2 * 131072)) + ((((int)threadIdx.x) >> 2) * 4096)) + (p * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32) / 8)];
    }
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(threadgroup uint4*)(b_s + (((((p + 1) & 1) * 2048) + (i_3 * 1024)) + (((int)threadIdx.x) * 8))) = B[(((((((p * 131072) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 131072) / 8)];
    }
    simdgroup_half8x8 A_local[8];
    simdgroup_half8x8 B_local[4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ki = 0; ki < 4; ++ki) {
      #pragma clang loop unroll(disable)
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        simdgroup_load(A_local[i_4], (&(a_s[(((((p & 1) * 4096) + ((((int)__simd_group_id) & 1) * 2048)) + (i_4 * 256)) + (ki * 8))])), 32, 0, (bool)0);
      }
      for (int j = 0; j < 4; ++j) {
        simdgroup_load(B_local[j], (&(b_s[(((((p & 1) * 2048) + (ki * 512)) + ((((int)__simd_group_id) >> 1) * 32)) + (j * 8))])), 64, 0, (bool)0);
      }
      #pragma clang loop unroll(disable)
      for (int i_5 = 0; i_5 < 8; ++i_5) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          simdgroup_multiply_accumulate(acc[((i_5 * 4) + j_1)], A_local[i_5], B_local[j_1], acc[((i_5 * 4) + j_1)]);
        }
      }
    }
  }
  simdgroup_half8x8 A_local_1[8];
  simdgroup_half8x8 B_local_1[4];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    #pragma clang loop unroll(disable)
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      simdgroup_load(A_local_1[i_6], (&(a_s[(((((((int)__simd_group_id) & 1) * 2048) + (i_6 * 256)) + (ki_1 * 8)) + 4096)])), 32, 0, (bool)0);
    }
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      simdgroup_load(B_local_1[j_2], (&(b_s[((((ki_1 * 512) + ((((int)__simd_group_id) >> 1) * 32)) + (j_2 * 8)) + 2048)])), 64, 0, (bool)0);
    }
    #pragma clang loop unroll(disable)
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      for (int j_3 = 0; j_3 < 4; ++j_3) {
        simdgroup_multiply_accumulate(acc[((i_7 * 4) + j_3)], A_local_1[i_7], B_local_1[j_3], acc[((i_7 * 4) + j_3)]);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_store(acc[0], (&(c_s[(((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32))])), 64, 0, (bool)0);
  simdgroup_store(acc[1], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 8)])), 64, 0, (bool)0);
  simdgroup_store(acc[2], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 16)])), 64, 0, (bool)0);
  simdgroup_store(acc[3], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 24)])), 64, 0, (bool)0);
  simdgroup_store(acc[4], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 512)])), 64, 0, (bool)0);
  simdgroup_store(acc[5], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 520)])), 64, 0, (bool)0);
  simdgroup_store(acc[6], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 528)])), 64, 0, (bool)0);
  simdgroup_store(acc[7], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 536)])), 64, 0, (bool)0);
  simdgroup_store(acc[8], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1024)])), 64, 0, (bool)0);
  simdgroup_store(acc[9], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1032)])), 64, 0, (bool)0);
  simdgroup_store(acc[10], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1040)])), 64, 0, (bool)0);
  simdgroup_store(acc[11], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1048)])), 64, 0, (bool)0);
  simdgroup_store(acc[12], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1536)])), 64, 0, (bool)0);
  simdgroup_store(acc[13], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1544)])), 64, 0, (bool)0);
  simdgroup_store(acc[14], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1552)])), 64, 0, (bool)0);
  simdgroup_store(acc[15], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 1560)])), 64, 0, (bool)0);
  simdgroup_store(acc[16], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2048)])), 64, 0, (bool)0);
  simdgroup_store(acc[17], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2056)])), 64, 0, (bool)0);
  simdgroup_store(acc[18], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2064)])), 64, 0, (bool)0);
  simdgroup_store(acc[19], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2072)])), 64, 0, (bool)0);
  simdgroup_store(acc[20], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2560)])), 64, 0, (bool)0);
  simdgroup_store(acc[21], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2568)])), 64, 0, (bool)0);
  simdgroup_store(acc[22], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2576)])), 64, 0, (bool)0);
  simdgroup_store(acc[23], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 2584)])), 64, 0, (bool)0);
  simdgroup_store(acc[24], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3072)])), 64, 0, (bool)0);
  simdgroup_store(acc[25], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3080)])), 64, 0, (bool)0);
  simdgroup_store(acc[26], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3088)])), 64, 0, (bool)0);
  simdgroup_store(acc[27], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3096)])), 64, 0, (bool)0);
  simdgroup_store(acc[28], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3584)])), 64, 0, (bool)0);
  simdgroup_store(acc[29], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3592)])), 64, 0, (bool)0);
  simdgroup_store(acc[30], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3600)])), 64, 0, (bool)0);
  simdgroup_store(acc[31], (&(c_s[((((((int)__simd_group_id) & 1) * 4096) + ((((int)__simd_group_id) >> 1) * 32)) + 3608)])), 64, 0, (bool)0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(disable)
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    thread float4 c_s_local_cast_1[1];
    thread half4 C_local_cast[1];
    c_s_local_cast_1[0] = *(threadgroup float4*)(c_s + ((i_8 * 512) + (((int)threadIdx.x) * 4)));
    C_local_cast[0] = ((half4)c_s_local_cast_1[0]);
    C[((((((((int)blockIdx.y) * 524288) + (i_8 * 32768)) + ((((int)threadIdx.x) >> 4) * 4096)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)) / 4)] = C_local_cast[0];
  }
}


