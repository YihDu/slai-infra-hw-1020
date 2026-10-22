"""Execute the actual generated TileLang program on Mac Metal, with explicit scope."""
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(workload,config,source,directory):
    import torch
    import tilelang
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise RuntimeError('Mac Metal execution unavailable')
    source=Path(source);directory=Path(directory)
    spec=importlib.util.spec_from_file_location('generated_candidate',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=module.build(workload,config)
    tir=run.program.script();(directory/'input.tir').write_text(tir)
    m,k=workload['inputs']['A']['shape'];n=workload['inputs']['B']['shape'][1]
    cases=[]
    for pattern,seed in [('normal',42),('normal',7),('zero_rows_signed_columns',42)]:
        torch.manual_seed(seed)
        a=(torch.randn(m,k)*.1).half();b=(torch.randn(k,n)*.1).half()
        if pattern=='zero_rows_signed_columns':
            a[::7]=0;b[:,::2]=-b[:,::2].abs();b[:,1::2]=b[:,1::2].abs()
        reference=(a.double()@b.double()).half()
        start=time.monotonic()
        outputs=run({'A':a.to('mps'),'B':b.to('mps')})
        actual=outputs['C'].cpu()
        elapsed=round((time.monotonic()-start)*1000)
        contract=set(outputs)=={'C'} and actual.shape==reference.shape and actual.dtype==torch.float16
        if contract:
            errors=(actual.float()-reference.float()).abs()
            limit=workload['tolerance']['atol']+workload['tolerance']['rtol']*reference.float().abs()
            bad=~torch.isfinite(actual)|(errors>limit)
            finite=torch.isfinite(errors).all().item()
            maximum=errors.max().item() if finite else None
            mismatches=int(bad.sum().item())
        else:maximum=None;mismatches=m*n
        cases.append(dict(pattern=pattern,seed=seed,passed=contract and mismatches==0,
                          max_abs_error=maximum,mismatches=mismatches,elapsed_ms=elapsed,
                          timing_source='Metal compile/execution/copy wall time; not GB10 performance'))
    shader=run.compiled['metal'].get_kernel_source()
    (directory/'kernel.metal').write_text(shader)
    return dict(status='passed' if all(c['passed'] for c in cases) else 'failed',passed=all(c['passed'] for c in cases),
                checked_target='metal',cuda_binary_executed=False,
                scope='Exact TileLang source/configuration executed on Metal; CUDA backend numerical execution not performed',
                source_sha256=digest(source),config=config,tir_sha256=hashlib.sha256(tir.encode()).hexdigest(),
                metal_sha256=hashlib.sha256(shader.encode()).hexdigest(),reference='CPU PyTorch float64 full matmul rounded to float16',
                hardware=platform.platform(),torch_version=torch.__version__,tilelang_version=tilelang.__version__,
                tolerance=workload['tolerance'],cases=cases)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workload',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    try:result=check(json.loads(args.workload.read_text()),json.loads(args.config.read_text()),args.source,args.output.parent)
    except Exception as exc:
        result=dict(status='error',passed=False,error_type=type(exc).__name__,reason=str(exc)[:2000],checked_target='metal',cuda_binary_executed=False)
    result.update(config_sha256=digest(args.config),elapsed_ms=round((time.monotonic()-start)*1000))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
