"""Development batch; each independent workload has its own 1200-second budget."""
import subprocess,pathlib,json,time
root=pathlib.Path.cwd();out=root/'results/exploration-v3'
order=['gemm_32x4096x4096','gemm_128x4096x4096','gemm_2048x2048x2048','gemm_4096x4096x1024','gemm_4096x4096x4096']
for name in order:
 print('START '+name,flush=True)
 run=subprocess.run(['./run.sh','--workload',str(root/'third_party/gb10/workloads'/f'{name}.json'),'--model-config',str(root/'configs/deepseek.json'),'--output',str(out/name),'--rounds','12'])
 if run.returncode:raise SystemExit(run.returncode)
 r=json.loads((out/name/'result.json').read_text());e=json.loads((out/name/'evaluation.json').read_text())
 print(json.dumps(dict(case=name,status=r['status'],best_us=e['value'],elapsed_ms=r['elapsed_ms'])),flush=True)
