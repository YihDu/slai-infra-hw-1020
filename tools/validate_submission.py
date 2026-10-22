#!/usr/bin/env python3
"""Validate identities, source snapshots and real-run evidence before packaging."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    def reject(value):raise ValueError('Non-finite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(directory,relative):
    p=(directory/relative).resolve()
    assert p.is_relative_to(directory.resolve()) and p.is_file(),relative
    return p


def verify_run(directory):
    result=read(directory/'result.json');manifest=read(directory/'state/manifest.json')
    for name,sha in manifest['resources'].items():assert digest(inside(directory/'state/source',name))==sha,name
    events=[json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines()]
    for event in events:
        for name in event['artifacts']:inside(directory,name)
    assert result['usage']['model_calls']==sum(e['event']=='model_started' for e in events)
    assert events[-1]['event']=='run_completed'
    evaluations=read(directory/'evaluation.json');search=read(directory/'state/search.json')
    assert result['status']=='checked' and result['checked_target']=='metal'
    assert result['implementation']['factory']=='build'
    assert result['cuda_binary_executed'] is False
    assert result['elapsed_ms']<=1200000
    selected=next(a for a in evaluations['attempts'] if a['attempt']==result['selected_attempt'])
    assert selected['configuration']==result['implementation']['config']
    assert selected['identity']==result['identity']
    assert digest(inside(directory,result['implementation']['path']))==result['identity']['source_sha256']
    usable=[];cases=0
    for attempt in evaluations['attempts']:
        if 'configuration' not in attempt:continue
        tag=attempt['attempt'];impl=directory/'implementation'/tag;check=directory/'checks'/tag
        identity=attempt['identity'];numeric=attempt['correctness'];comp=attempt['compilation'];sim=attempt['simulation']
        assert digest(impl/'kernel.py')==identity['source_sha256']
        assert digest(impl/'config.json')==identity['config_sha256']
        solver=read(directory/'formulations'/tag/'solver.json')
        assignment=solver['candidates'][solver['selected_index']]
        assert assignment['config']==attempt['configuration']
        assert assignment['smem_bytes_estimate']<=solver['effective_shared_memory_limit']
        if numeric.get('passed'):
            assert numeric['checked_target']=='metal' and numeric['cuda_binary_executed'] is False
            assert numeric['source_sha256']==identity['source_sha256'] and numeric['config_sha256']==identity['config_sha256']
            assert digest(check/'input.tir')==numeric['tir_sha256']
            assert digest(check/'kernel.metal')==numeric['metal_sha256']
            assert len(numeric['cases'])>=3 and all(c['passed'] and c['mismatches']==0 for c in numeric['cases'])
            if 'disk_kernel_cache_disabled' in numeric:assert numeric['disk_kernel_cache_disabled'] is True
            cases+=len(numeric['cases'])
        if comp['status']=='compiled':
            folder=directory/'compilation'/tag
            assert comp['target']=='sm_121a' and comp['gpu_execution'] is False
            assert comp['source_sha256']==identity['source_sha256'] and comp['config_sha256']==identity['config_sha256']
            assert digest(folder/'kernel.py')==identity['source_sha256'] and digest(folder/'config.json')==identity['config_sha256']
            for file,key in [('kernel.cubin','cubin_sha256'),('kernel.sass','sass_sha256')]:assert digest(folder/file)==comp[key]
            assert digest(folder/'input.tir')==numeric['tir_sha256']
            request=read(folder/'simulation-input.json')
            for key,value in attempt['configuration'].items():assert request['holes'][key]==value
            assert request['regs']==comp['registers'] and request['smem_bytes']==comp['smem_bytes']
        if sim['status']=='completed':
            assert numeric['passed'] and attempt['identical_input_tir'] and comp['status']=='compiled'
            assert sim['raw']['sass_sha256']==comp['sass_sha256']
            assert sim['audit'] and all(a['warps']==a['warps_done'] for a in sim['audit'])
            assert sim['value']>0 and sim['unit']=='us'
            usable.append(dict(attempt=tag,configuration=attempt['configuration'],simulated_us=sim['value'],features=assignment['features'],identity=identity))
    assert usable
    best=min(usable,key=lambda x:x['simulated_us'])
    assert evaluations['value']==best['simulated_us']==search['best']['simulated_us']
    assert selected['simulation']['value']==best['simulated_us']
    initial=next(x for x in usable if x['attempt']==search['initial']['attempt'])
    weights=read(directory/'formulations'/initial['attempt']/'plan.json')['weights']
    predicted=min(usable,key=lambda x:sum(weights[k]*v for k,v in x['features'].items()))
    return dict(workload=result['workload_id'],elapsed_seconds=result['elapsed_ms']/1000,
                initial=initial,final=best,simulated_speedup=initial['simulated_us']/best['simulated_us'],
                initial_proxy_preferred_attempt=predicted['attempt'],initial_proxy_preferred_us=predicted['simulated_us'],
                tested_candidates=len(usable),numeric_test_cases=cases,usage=result['usage'],
                checked_target='metal',cuda_binary_executed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=(ROOT/'results/submission-v2' if (ROOT/'results/submission-v2').exists() else ROOT/'results'));a=p.parse_args()
    expected=sorted(read(w)['id'] for w in (ROOT/'third_party/gb10/workloads').glob('*.json'))
    rows=[verify_run(a.results/name) for name in expected]
    source=read(ROOT/'third_party/gb10/SOURCE.json')
    for name,record in source['files'].items():assert digest(ROOT/'third_party/gb10'/name)==record['sha256']
    summary=dict(scope='TileLang Metal numerical checks + identical input IR + GB10 SASS simulation; CUDA binary not executed',cases=rows)
    (a.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
