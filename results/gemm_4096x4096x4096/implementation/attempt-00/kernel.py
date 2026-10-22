"""Prewritten GEMM family, parameters chosen by HiGHS. Compilation not yet validated.
API reference: tile-ai/tilelang examples/gemm/example_gemm.py (2026-10-22).
"""
def build(workload, config):
    import tilelang
    import tilelang.language as T
    m, k = workload['inputs']['A']['shape']
    n = workload['inputs']['B']['shape'][1]
    bm, bn, bk = (config[x] for x in ('block_M','block_N','block_K'))
    stages, threads = config['num_stages'], config['threads']

    @tilelang.jit
    def matmul(A, B):
        A: T.Tensor((m, k), T.float16)
        B: T.Tensor((k, n), T.float16)
        C = T.empty((m, n), T.float16)
        with T.Kernel(T.ceildiv(n,bn), T.ceildiv(m,bm), threads=threads) as (bx,by):
            a_s=T.alloc_shared((bm,bk),T.float16)
            b_s=T.alloc_shared((bk,bn),T.float16)
            acc=T.alloc_fragment((bm,bn),T.float32)
            T.clear(acc)
            for p in T.Pipelined(T.ceildiv(k,bk),num_stages=stages):
                T.copy(A[by*bm,p*bk],a_s)
                T.copy(B[p*bk,bx*bn],b_s)
                T.gemm(a_s,b_s,acc)
            T.copy(acc,C[by*bm,bx*bn])
        return C

    kernel=matmul.compile()
    return lambda inputs: {'C':kernel(inputs['A'],inputs['B'])}
