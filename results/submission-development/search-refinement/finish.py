"""Run acceptance serially after the expensive exploratory batch has completed."""
import json,subprocess,time
from pathlib import Path
root=Path.cwd();out=Path(__file__).resolve().parent
trace=root/'results/exploration-v3/gemm_4096x4096x4096/trace.jsonl'
while True:
 try:
  if json.loads(trace.read_text().splitlines()[-1])['event']=='run_completed':break
 except (OSError,ValueError,IndexError):pass
 time.sleep(2)
with (out/'integration-tests-serial.log').open('w') as log:
 r=subprocess.run([str(root/'.venv/bin/python'),'-m','unittest','discover','-s','tests'],stdout=log,stderr=subprocess.STDOUT)
print('Serial test return code:',r.returncode,flush=True)
if r.returncode:raise SystemExit(r.returncode)
r=subprocess.run(['./run.sh','--workload','third_party/gb10/workloads/gemm_4096x4096x4096.json','--model-config','configs/deepseek.json','--output',str(out/'staged-big-run'),'--rounds','2'])
raise SystemExit(r.returncode)
