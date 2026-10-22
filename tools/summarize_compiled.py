#!/usr/bin/env python3
"""Verify the saved compiled runs before reporting simulator estimates."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rows=[]
    for directory in sorted((ROOT/'results/compiled-baseline').glob('gemm_*')):
        result=json.loads((directory/'result.json').read_text())
        manifest=json.loads((directory/'state/manifest.json').read_text())['resources']
        for name,digest in manifest.items():
            assert sha(directory/'state/source'/name)==digest,name
        for line in (directory/'trace.jsonl').read_text().splitlines():
            for ref in json.loads(line)['artifacts']:
                path=(directory/ref).resolve()
                assert path.is_relative_to(directory.resolve()) and path.is_file(),ref
        evaluations=json.loads((directory/'evaluation.json').read_text())['attempts']
        checks=json.loads((directory/'checks.json').read_text())['attempts']
        attempts=[]
        for item in evaluations:
            if 'kernel_sha256' not in item:
                attempts.append(item);continue
            tag=item['attempt'];impl=directory/'implementation'/tag
            assert sha(impl/'kernel.py')==item['kernel_sha256']
            assert sha(impl/'config.json')==item['config_sha256']
            comp=item['compilation'];sim=item['simulation'];folder=directory/'compilation'/tag
            if comp['status']=='compiled':
                assert sha(folder/'kernel.py')==comp['source_sha256']==item['kernel_sha256']
                assert sha(folder/'config.json')==comp['config_sha256']==item['config_sha256']
                assert sha(folder/'kernel.cubin')==comp['cubin_sha256']
                assert sha(folder/'kernel.sass')==comp['sass_sha256']
            if sim['status']=='completed':
                assert sim['raw']['sass_sha256']==comp['sass_sha256']
                assert sim['audit'] and all(a['warps']==a['warps_done'] for a in sim['audit'])
            attempts.append(dict(attempt=tag,configuration=item['configuration'],compilation=comp['status'],
                                 simulation=sim['status'],simulated_us=sim.get('value'),
                                 numerical_correctness=item['correctness']['status']))
        assert result['status']=='no_valid_result' and result['implementation'] is None
        rows.append(dict(workload=result['workload_id'],seconds=result['elapsed_ms']/1000,
                         usage=result['usage'],cpu_shadow_passed=all(c['passed'] for c in checks),attempts=attempts))
    assert len(rows)==5,'Expected all five public workloads'
    summary=dict(scope='Compiled SASS timing estimates; exact GPU numerical correctness unverified',cases=rows)
    (ROOT/'results/compiled-baseline/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
