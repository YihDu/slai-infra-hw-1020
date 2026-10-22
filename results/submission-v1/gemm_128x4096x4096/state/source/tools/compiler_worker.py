#!/usr/bin/env python3
"""Run inside the pinned Linux compiler image, without GPU access."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workload',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    report=dict(status='error',target='sm_121a',source_sha256=sha(args.source),config_sha256=sha(args.config),
                gpu_execution=False,stage='import')
    try:
        import tilelang
        from tilelang import tvm
        from tilelang.env import TILELANG_TEMPLATE_PATH,CUTLASS_INCLUDE_DIR,CUDA_HOME
        report['versions']={n:importlib.metadata.version(n) for n in ['tilelang','torch','nvidia-cuda-nvcc','nvidia-cuda-nvdisasm']}
        w=json.loads(args.workload.read_text());config=json.loads(args.config.read_text())
        spec=importlib.util.spec_from_file_location('generated_kernel',args.source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        program=module.define(w,config).get_tir()
        (out/'input.tir').write_text(program.script())
        report['stage']='lower'
        target=tvm.target.Target({'kind':'cuda','arch':'sm_121a'})
        with target, tvm.transform.PassContext(config={'tl.disable_warp_specialized':True}):
            artifact=tilelang.lower(program,target=target,enable_device_compile=False,enable_host_codegen=False)
        source=artifact.kernel_source
        (out/'kernel.cu').write_text(source)
        (out/'device.tir').write_text(artifact.device_mod.script())
        functions=list(artifact.device_mod.functions.values())
        resources=[]
        for fn in functions:
            attrs=fn.attrs
            resources.append({str(k):str(attrs[k]) for k in attrs.keys()})
        report['device_attributes']=resources
        dynamic=max((int(f.attrs.get('dyn_shared_memory_buf',0)) for f in functions),default=0)
        report['stage']='nvcc'
        cmd=['nvcc','-std=c++20','-arch=sm_121a','-cubin',str(out/'kernel.cu'),'-o',str(out/'kernel.cubin'),
             '-Xptxas','-v','-O3','--expt-relaxed-constexpr',f'-I{TILELANG_TEMPLATE_PATH}',f'-I{CUTLASS_INCLUDE_DIR}',f'-I{CUDA_HOME}/include']
        report['command']=cmd
        process=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        (out/'nvcc.log').write_text(process.stdout+process.stderr)
        if process.returncode:raise RuntimeError('nvcc failed; see nvcc.log')
        regs=re.search(r'Used (\d+) registers',process.stderr)
        smem=re.search(r'(\d+) bytes smem',process.stderr)
        symbol=re.search(r"Compiling entry function '([^']+)'",process.stderr)
        threads=re.search(r'__launch_bounds__\((\d+)',source)
        if not regs or not symbol or not threads:raise RuntimeError('Could not extract actual kernel resources')
        report['stage']='nvdisasm'
        process=subprocess.run(['nvdisasm','-hex',str(out/'kernel.cubin')],capture_output=True,text=True,timeout=30)
        (out/'nvdisasm.log').write_text(process.stderr)
        if process.returncode:raise RuntimeError('nvdisasm failed')
        (out/'kernel.sass').write_text(process.stdout)
        actual_threads=int(threads.group(1));static=int(smem.group(1)) if smem else 0
        report.update(status='compiled',stage='complete',kernel=symbol.group(1),registers=int(regs.group(1)),
                      threads=actual_threads,static_smem_bytes=static,dynamic_smem_bytes=dynamic,smem_bytes=static+dynamic,
                      cubin_sha256=sha(out/'kernel.cubin'),sass_sha256=sha(out/'kernel.sass'))
        if 'UTMALDG' in process.stdout:raise RuntimeError('Unexpected TMA mapping in cp.async family')
        if actual_threads!=config['threads']:raise RuntimeError('Unexpected launch thread count')
        if static+dynamic>101376 or int(regs.group(1))*actual_threads>65536:
            report['status']='resource_invalid'
        m,k=w['inputs']['A']['shape'];n=w['inputs']['B']['shape'][1]
        request=dict(path='cpasync',sass='kernel.sass',kernel=symbol.group(1),shape=dict(M=m,N=n,K=k),
                     holes={**config,'swizzle':0},regs=int(regs.group(1)),smem_bytes=static+dynamic,warm=True,split=1)
        (out/'simulation-input.json').write_text(json.dumps(request,indent=2)+'\n')
    except Exception as exc:
        report.update(status='error',error_type=type(exc).__name__,reason=str(exc))
    report['elapsed_ms']=round((time.monotonic()-start)*1000)
    (out/'compilation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['status']=='compiled' else 1


if __name__=='__main__':raise SystemExit(main())
