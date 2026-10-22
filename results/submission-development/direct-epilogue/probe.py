"""Bounded development hypotheses, not an exhaustive performance table or formal Agent run."""
import os,sys,json,shutil,time,hashlib
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from optimizer.checker import check_candidate
from optimizer.compiler import compile_candidate,simulate_candidate
os.environ.update(TILELANG_DISABLE_CACHE='1',TILELANG_AUTO_TUNING_DISABLE_CACHE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
out=Path(__file__).resolve().parent
source=out/'kernel.py'
if not source.exists():shutil.copyfile(root/'templates/gemm_tilelang.py',source)
# Test independent hypotheses: smaller M/more CTAs; wider N; larger K; fewer/more threads.
configs=[(128,128,32,2,128),(64,128,32,2,128)]
for idx,(bm,bn,bk,st,th) in enumerate(configs):
 d=out/f'probe-{idx:02d}'
 if (d/'probe.json').exists():continue
 d.mkdir(exist_ok=True);w=root/'third_party/gb10/workloads/gemm_4096x4096x4096.json'
 c=dict(block_M=bm,block_N=bn,block_K=bk,num_stages=st,threads=th,swizzle=16)
 cp=d/'config.json';cp.write_text(json.dumps(c,indent=2)+'\n')
 r=dict(config=c,development_only=True)
 r['numeric']=json.loads((d/'checks/numeric.json').read_text()) if (d/'checks/numeric.json').exists() else check_candidate(w,cp,source,d/'checks',120)
 if r['numeric'].get('passed'):
  r['compile']=json.loads((d/'compilation/compilation.json').read_text()) if (d/'compilation/compilation.json').exists() else compile_candidate(w,cp,source,d/'compilation',180)
  if r['compile']['status']=='compiled':
   assert hashlib.sha256((d/'compilation/input.tir').read_bytes()).hexdigest()==r['numeric']['tir_sha256']
   r['simulation']=simulate_candidate(d/'compilation',120)
 (d/'probe.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(dict(probe=idx,config=c,numeric=r['numeric']['status'],compile=r.get('compile',{}).get('status'),simulation=r.get('simulation',{}))),flush=True)
