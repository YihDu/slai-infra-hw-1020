"""Prewritten cp.async GEMM family, parameters chosen by HiGHS.
API reference: tile-ai/tilelang examples/gemm/example_gemm.py (2026-10-22).
"""
def define(workload, config):
    import tilelang
    import tilelang.language as T
    m, k = workload['inputs']['A']['shape']
    n = workload['inputs']['B']['shape'][1]
    bm, bn, bk = (config[x] for x in ('block_M','block_N','block_K'))
    stages, threads = config['num_stages'], config['threads']
    panel=config.get('swizzle',0)
    panel_rows=max(1,panel)
    tiles_m=(m+bm-1)//bm;tiles_n=(n+bn-1)//bn

    @tilelang.jit(target={'kind':'cuda','arch':'sm_121a'},
                  pass_configs={'tl.disable_warp_specialized': True})
    def matmul(A, B):
        A: T.Tensor((m, k), T.float16)
        B: T.Tensor((k, n), T.float16)
        C = T.empty((m, n), T.float16)
        with T.Kernel(T.ceildiv(n,bn), T.ceildiv(m,bm), threads=threads) as (px,py):
            linear=py*tiles_n+px
            panel_index=linear//(panel_rows*tiles_n)
            offset=linear%(panel_rows*tiles_n)
            stride=T.min(panel_rows,tiles_m-panel_index*panel_rows)
            by=py if panel==0 else panel_index*panel_rows+offset%stride
            bx=px if panel==0 else T.if_then_else(panel_index%2==1,tiles_n-1-offset//stride,offset//stride)
            a_s=T.alloc_shared((bm,bk),T.float16)
            b_s=T.alloc_shared((bk,bn),T.float16)
            acc=T.alloc_fragment((bm,bn),T.float32)
            T.clear(acc)
            for p in T.Pipelined(T.ceildiv(k,bk),num_stages=stages):
                T.copy(A[by*bm,p*bk],a_s,disable_tma=True)
                T.copy(B[p*bk,bx*bn],b_s,disable_tma=True)
                T.gemm(a_s,b_s,acc)
            T.sync_threads()
            c_s=T.alloc_shared((bm,bn),T.float16)
            for i,j in T.Parallel(bm,bn,coalesced_width=1):
                c_s[i,j]=acc[i,j].astype(T.float16)
            T.sync_threads()
            for i,j in T.Parallel(bm,bn,coalesced_width=1):
                if by*bm+i<m and bx*bn+j<n:
                    C[by*bm+i,bx*bn+j]=c_s[i,j].astype(T.float16)
        return C

    return matmul


def build(workload, config):
    """Execute the same PrimFunc on the input device (CPU tensors use Mac Metal).

    Metal execution checks this source/configuration, not CUDA binary correctness.
    CUDA inputs must reside on a device compatible with the GB10 sm_121a target.
    """
    import tilelang
    import torch
    program=define(workload,config).get_tir()
    compiled={}

    def run(inputs):
        a,b=inputs['A'],inputs['B']
        original_device=a.device
        if a.device!=b.device:raise ValueError('Inputs must use the same device')
        if a.device.type=='cpu':
            if not torch.backends.mps.is_available():
                raise RuntimeError('CPU inputs require the declared Mac Metal execution environment')
            a,b=a.to('mps'),b.to('mps')
        backend='metal' if a.device.type=='mps' else 'cuda' if a.device.type=='cuda' else None
        if backend is None:raise ValueError('Only Metal and GB10 CUDA execution are supported')
        if backend not in compiled:
            if backend=='metal':
                compiled[backend]=tilelang.compile(program,target='metal',execution_backend='torch')
            else:
                compiled[backend]=tilelang.compile(program,target={'kind':'cuda','arch':'sm_121a'},
                                                   pass_configs={'tl.disable_warp_specialized':True})
        if backend=='metal':
            output=torch.full((a.shape[0],b.shape[1]),float('nan'),device=a.device,dtype=torch.float16)
            compiled[backend](a,b,output)
        else:output=compiled[backend](a,b)
        return {'C':output.to(original_device)}
    run.program=program
    run.compiled=compiled
    return run
