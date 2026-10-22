"""Compare independent searches against the frozen v2 best, never just a weak new baseline."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from validate_submission import verify_run
out=Path(__file__).resolve().parent
cases=[]
for workload in sorted((root/'third_party/gb10/workloads').glob('*.json')):
 name=json.loads(workload.read_text())['id'];path=out/name
 if not (path/'trace.jsonl').exists():continue
 events=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
 if not events or events[-1]['event']!='run_completed':continue
 new=verify_run(path);old=verify_run(root/'results/submission-v2'/name)
 cases.append(dict(workload=name,previous_best_us=old['final']['simulated_us'],exploration_best_us=new['final']['simulated_us'],
                   improvement_ratio=old['final']['simulated_us']/new['final']['simulated_us'],
                   keep='exploration-v3' if new['final']['simulated_us']<old['final']['simulated_us'] else 'submission-v2',
                   new_search=new))
probes=[]
for family in ['expanded-family','half-epilogue','half-epilogue-scalar','direct-epilogue','large-panels','warp-policy']:
 for p in sorted((root/'results/submission-development'/family).glob('probe-*/probe.json')):
  r=json.loads(p.read_text());probes.append(dict(family=family,probe=p.parent.name,config=r['config'],numeric=r['numeric']['status'],
   numeric_reason=r['numeric'].get('reason'),simulated_us=r.get('simulation',{}).get('value'),
   path=p.relative_to(root).as_posix()))
summary=dict(scope='Development exploration; Metal numerical execution and GB10 simulated timing, no CUDA execution',
             cases=cases,development_probes=probes)
(out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(completed_cases=len(cases),probes=len(probes),comparisons=[{k:v for k,v in c.items() if k!='new_search'} for c in cases]),indent=2))
