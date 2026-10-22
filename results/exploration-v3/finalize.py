"""Finalize exploration evidence; do not overwrite the frozen submission ZIP."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from validate_submission import verify_run,read,digest
out=Path(__file__).resolve().parent
ref=root/'results/submission-development/search-refinement'
staged=verify_run(ref/'staged-big-run');smoke=verify_run(ref/'run')
manifest=read(ref/'staged-big-run/state/manifest.json')
for name,sha in manifest['resources'].items():assert digest(root/name)==sha,name
rows=[];runs=[]
for w in sorted((root/'third_party/gb10/workloads').glob('*.json')):
 name=read(w)['id'];old=root/'results/submission-v2'/name;new=out/name
 previous=verify_run(old);explored=verify_run(new);runs.append(explored)
 choices=[(previous,old),(explored,new)]
 if staged['workload']==name:choices.append((staged,ref/'staged-big-run'))
 best,path=min(choices,key=lambda x:x[0]['final']['simulated_us'])
 rows.append(dict(workload=name,previous_best_us=previous['final']['simulated_us'],best_known_us=best['final']['simulated_us'],
                  speedup_over_previous=previous['final']['simulated_us']/best['final']['simulated_us'],
                  result=(path/'result.json').relative_to(root).as_posix(),selected_attempt=best['final']['attempt']))
usage={}
for key in runs[0]['usage']:
 values=[r['usage'][key] for r in [*runs,smoke,staged]]
 usage[key]=None if any(v is None for v in values) else sum(values)
serial=(ref/'integration-tests-serial.log').read_text();assert 'Ran 21 tests' in serial and '\nOK\n' in serial
frozen=root/'dist/slai-infra-hw-1020.zip';frozen_report=read(frozen.with_suffix('.json'));assert digest(frozen)==frozen_report['sha256']
report=dict(scope='Metal numerical execution and GB10 SASS timing simulation; no CUDA execution',best_known=rows,
            exploration_usage=usage,public_searches=[dict(workload=r['workload'],elapsed_seconds=r['elapsed_seconds'],numeric_test_cases=r['numeric_test_cases']) for r in runs],
            staged_search=staged,refined_smoke=smoke,tests=21,serial_tests='passed',
            serial_log_sha256=digest(ref/'integration-tests-serial.log'),current_runtime_matches_staged_snapshot=True,
            frozen_zip_unchanged=True,frozen_zip_sha256=digest(frozen),course_upload=False)
(out/'best-known.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(best_known=rows,usage=usage,staged_seconds=staged['elapsed_seconds'],tests=21),indent=2))
