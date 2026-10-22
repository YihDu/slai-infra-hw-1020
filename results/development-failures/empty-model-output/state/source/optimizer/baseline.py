"""Development baseline: real Agent/MIP with explicitly limited CPU diagnostics."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from optimizer.formulation import solve, validate, index_proof
from optimizer.model import ModelConfig, generate
from optimizer.trace import Trace

ROOT=Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape_of(w):
    if w.get('operation')!='C = A @ B':raise ValueError('Baseline supports only declared C = A @ B')
    if set(w['inputs'])!={'A','B'} or set(w['outputs'])!={'C'}:raise ValueError('Unsupported tensor contract')
    a,b,c=w['inputs']['A'],w['inputs']['B'],w['outputs']['C']
    if any(x['dtype']!='float16' or len(x['shape'])!=2 for x in [a,b,c]):raise ValueError('Expected rank-2 float16 tensors')
    if any(type(v) is not int or v<=0 for x in [a,b,c] for v in x['shape']):raise ValueError('Invalid shape')
    m,k=a['shape'];kb,n=b['shape']
    if kb!=k or c['shape']!=[m,n] or w['accumulation_dtype']!='float32':raise ValueError('Invalid GEMM contract')
    if w['evaluation']['scope']!='whole_workload':raise ValueError('Unsupported evaluation scope')
    for key in ['atol','rtol']:
        v=w['tolerance'][key]
        if not isinstance(v,(int,float)) or not 0<=v<1:raise ValueError('Invalid tolerance')
    return m,n,k


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workload',required=True,type=Path)
    parser.add_argument('--model-config',type=Path,default=ROOT/'configs/deepseek.json')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--budget-seconds',type=int,default=1200)
    args=parser.parse_args()
    if not 1<=args.rounds<=5 or not 1<=args.budget_seconds<=1200:parser.error('rounds 1..5; budget 1..1200 seconds')
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):parser.error('Output must be new or empty; preserve previous evidence')
    out.mkdir(parents=True,exist_ok=True)
    for folder in ['implementation','formulations','state','agent','checks']: (out/folder).mkdir()
    start=time.monotonic();deadline=start+args.budget_seconds;trace=Trace(out)
    usage=dict(model_calls=0,input_tokens=0,output_tokens=0,cached_tokens=0,reasoning_tokens=0,evaluations=0,
               cpu_diagnostics=0,compilations=0,simulations=0)
    result=dict(schema_version='agentic-result-v1',workload_id=None,status='no_valid_result',implementation=None,
                evidence={},elapsed_ms=0,usage=usage,reason='No numerically checked TileLang implementation yet')
    checks=[];attempts=[];feedback=None;exit_code=0
    def checkpoint():
        result['elapsed_ms']=round((time.monotonic()-start)*1000)
        dump(out/'checks.json',dict(scope='CPU shadow only',attempts=checks))
        dump(out/'evaluation.json',dict(attempts=attempts,performance_source='none',unit=None,value=None))
        result['evidence']=dict(correctness='checks.json',evaluation='evaluation.json')
        dump(out/'result.json',result)
    def remaining():
        if time.monotonic()>=deadline:raise TimeoutError('Wall-clock budget exhausted')
        return deadline-time.monotonic()
    def alarm(signum,frame):raise TimeoutError('Wall-clock budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,args.budget_seconds)
    checkpoint()
    try:
        w=json.loads(args.workload.read_text());shape=shape_of(w);result['workload_id']=w['id']
        model=ModelConfig.load(args.model_config)
        dump(out/'state/workload.json',w)
        dump(out/'state/model-config.json',dict(protocol=model.protocol,model=model.model,base_url=model.base_url,
                                               api_key_env=model.api_key_env,max_tokens_field=model.max_tokens_field,trust_env=model.trust_env))
        prompt=(ROOT/'prompts/formulate.txt').read_text()
        (out/'state/formulate.txt').write_text(prompt)
        manifest={str(p.relative_to(ROOT)):digest(p) for p in [*sorted((ROOT/'optimizer').glob('*.py')),ROOT/'templates/gemm_tilelang.py',ROOT/'prompts/formulate.txt']}
        dump(out/'state/manifest.json',dict(resources=manifest,python=sys.version,prebuilt='CPU shadow, kernel family, MIP builder, bounds checker',
                                          runtime_changes='Agent domains/resource constraints/objective; selected configuration; generated implementation artifacts',
                                          graph_optimization='Single GEMM: graph fusion not applicable; no TVM graph transformation claimed'))
        trace.record('workload_loaded',artifacts=['state/workload.json','state/model-config.json','state/manifest.json'])
        for round_id in range(args.rounds):
            remaining();tag=f'attempt-{round_id:02d}'
            request=prompt+'\nWorkload:\n'+json.dumps(w)+'\nFeedback:\n'+json.dumps(feedback)
            (out/'agent'/f'{tag}-request.txt').write_text(request)
            usage['model_calls']+=1
            trace.record('model_started',attempt=tag,artifacts=[f'agent/{tag}-request.txt'])
            try:
                response=generate(model,request,timeout=min(120,remaining()),max_output_tokens=4096)
                dump(out/'agent'/f'{tag}-response.json',response)
                raw=response.get('usage') or {}
                for target,keys in [('input_tokens',['input_tokens','prompt_tokens']),('output_tokens',['output_tokens','completion_tokens'])]:
                    val=next((raw[k] for k in keys if k in raw),None)
                    usage[target]=None if val is None or usage[target] is None else usage[target]+val
                for target,value in [('cached_tokens',raw.get('prompt_tokens_details',raw.get('input_tokens_details',{})).get('cached_tokens')),
                                     ('reasoning_tokens',raw.get('completion_tokens_details',raw.get('output_tokens_details',{})).get('reasoning_tokens'))]:
                    usage[target]=None if value is None or usage[target] is None else usage[target]+value
                text=response['text'].strip()
                if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
                plan=validate(json.loads(text))
            except TimeoutError:raise
            except Exception as exc:
                # Do not save HTTP headers, credentials or potentially sensitive server bodies.
                feedback=dict(stage='model_or_schema',error_type=type(exc).__name__)
                trace.record('model_or_schema_failed',attempt=tag,**feedback)
                attempts.append(dict(attempt=tag,**feedback));checkpoint();continue
            directory=out/'formulations'/tag;directory.mkdir()
            dump(directory/'plan.json',plan)
            config,solver=solve(plan,shape,directory,min(20,remaining()))
            dump(directory/'solver.json',solver)
            trace.record('highs_completed',attempt=tag,artifacts=[f'agent/{tag}-response.json',f'formulations/{tag}/plan.json',f'formulations/{tag}/model.lp',f'formulations/{tag}/model.mps',f'formulations/{tag}/solver.json'])
            if config is None:
                feedback=dict(stage='MIP',status=solver['status']);attempts.append(dict(attempt=tag,**feedback));checkpoint();continue
            impl=out/'implementation'/tag;impl.mkdir()
            shutil.copyfile(ROOT/'templates/gemm_tilelang.py',impl/'kernel.py')
            shutil.copyfile(ROOT/'optimizer/cpu_shadow.py',impl/'cpu_shadow.py')
            dump(impl/'config.json',config)
            ast.parse((impl/'kernel.py').read_text())
            kernel_hash=digest(impl/'kernel.py');config_hash=digest(impl/'config.json')
            proof=index_proof(shape,config,kernel_hash);proof['config_sha256']=config_hash
            dump(out/'checks'/f'{tag}-smt.json',proof)
            trace.record('implementation_generated',attempt=tag,artifacts=[f'implementation/{tag}/kernel.py',f'implementation/{tag}/config.json',f'checks/{tag}-smt.json'],kernel_sha256=kernel_hash,config_sha256=config_hash)
            check_path=out/'checks'/f'{tag}-cpu.json'
            cmd=[sys.executable,'-m','optimizer.numeric','--workload',str(out/'state/workload.json'),'--config',str(impl/'config.json'),
                 '--source',str(impl/'cpu_shadow.py'),'--output',str(check_path)]
            try:
                process=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=max(.01,remaining()-2))
                (out/'checks'/f'{tag}-cpu.log').write_text(process.stdout+process.stderr)
                cpu=json.loads(check_path.read_text()) if process.returncode==0 else dict(passed=False,error='CPU process failed')
                usage['cpu_diagnostics']+=1
            except subprocess.TimeoutExpired:
                cpu=dict(passed=False,error='CPU process timeout');dump(check_path,cpu)
            checks.append(dict(attempt=tag,**cpu))
            availability=dict(tilelang=importlib.util.find_spec('tilelang') is not None,nvcc=shutil.which('nvcc'),nvdisasm=shutil.which('nvdisasm'))
            evaluation=dict(attempt=tag,kernel_sha256=kernel_hash,config_sha256=config_hash,configuration=config,
                            compilation=dict(status='unavailable',tools=availability,reason='sm_121a compile/export adapter not integrated'),
                            correctness=dict(status='not_run',reason='CPU shadow does not execute TileLang'),
                            simulation=dict(status='not_run',reason='No compiled encoded SASS or actual resource metadata',source='gb10_instruction_simulator',value=None,unit='us'),
                            cpu_shadow_passed=cpu['passed'])
            attempts.append(evaluation);dump(out/'checks'/f'{tag}-evaluation.json',evaluation)
            trace.record('diagnostics_completed',attempt=tag,artifacts=[f'checks/{tag}-cpu.json',f'checks/{tag}-evaluation.json'],cpu_shadow_passed=cpu['passed'])
            feedback=dict(previous_plan=plan,selection=config,solver_status=solver['status'],cost=solver['incumbent'],
                          cpu_shadow_passed=cpu['passed'],compilation=evaluation['compilation'],simulation=evaluation['simulation'],
                          instruction='Revise domains/weights only when justified. CPU timing is not GB10 timing; compilation is still blocked.')
            dump(out/'state/latest-feedback.json',feedback)
            result['reason']='CPU development baseline completed; exact TileLang compilation/numerical execution and GB10 simulation remain unavailable'
            checkpoint()
    except TimeoutError:
        result['reason']='Wall-clock budget reached without a checked TileLang implementation'
        trace.record('deadline_reached')
    except Exception as exc:
        result.update(status='error',reason=type(exc).__name__+': '+str(exc)[:300]);exit_code=1
        trace.record('execution_error',error_type=type(exc).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        checkpoint();trace.record('run_completed',status=result['status'],artifacts=['result.json','checks.json','evaluation.json'])
    print(json.dumps(dict(status=result['status'],workload_id=result['workload_id'],elapsed_ms=result['elapsed_ms'],usage=usage)))
    return exit_code


if __name__=='__main__':raise SystemExit(main())
