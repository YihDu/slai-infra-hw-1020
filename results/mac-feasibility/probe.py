import hashlib,importlib.util,json,time,platform
from pathlib import Path
import torch,tilelang
root=Path.cwd();out=root/'.cache/mac-feasibility';torch.set_num_threads(4)
spec=importlib.util.spec_from_file_location('candidate',out/'kernel.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
c=dict(block_M=32,block_N=64,block_K=32,num_stages=2,threads=128)
(out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
reports=[]
for path in sorted((root/'third_party/gb10/workloads').glob('*.json')):
 w=json.loads(path.read_text());m,k=w['inputs']['A']['shape'];n=w['inputs']['B']['shape'][1]
 program=module.define(w,c).get_tir();tir=program.script();(out/(w['id']+'.tir')).write_text(tir)
 start=time.monotonic();kernel=tilelang.compile(program,target='metal',execution_backend='torch')
 (out/(w['id']+'.metal')).write_text(kernel.get_kernel_source())
 compile_seconds=time.monotonic()-start;cases=[]
 for pattern in ['normal_seed42','zero_rows_signed_columns']:
  start=time.monotonic();torch.manual_seed(42)
  a=(torch.randn(m,k)*.1).half();b=(torch.randn(k,n)*.1).half()
  if pattern=='zero_rows_signed_columns':
   a[::7]=0;b[:,::2]=-b[:,::2].abs();b[:,1::2]=b[:,1::2].abs()
  reference=(a.double()@b.double()).half()
  actual=torch.full((m,n),float('nan'),device='mps',dtype=torch.float16)
  kernel(a.to('mps'),b.to('mps'),actual);actual=actual.cpu()
  error=(actual.float()-reference.float()).abs()
  torch.testing.assert_close(actual,reference,rtol=w['tolerance']['rtol'],atol=w['tolerance']['atol'])
  cases.append(dict(pattern=pattern,passed=True,max_abs_error=error.max().item(),seconds=time.monotonic()-start))
 r=dict(workload=w['id'],configuration=c,source_sha256=hashlib.sha256((out/'kernel.py').read_bytes()).hexdigest(),
        tir_sha256=hashlib.sha256(tir.encode()).hexdigest(),checked_target='metal',cuda_binary_executed=False,
        compile_seconds=compile_seconds,cases=cases,versions=dict(torch=torch.__version__,tilelang=tilelang.__version__,platform=platform.platform()))
 reports.append(r);(out/'numeric.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(r),flush=True)
