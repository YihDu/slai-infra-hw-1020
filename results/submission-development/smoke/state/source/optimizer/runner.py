"""Budgeted Agent/MIP search with same-IR Metal checks and GB10 simulation."""
import argparse
import json
import shutil
import signal
import sys
import time
from pathlib import Path

from optimizer.baseline import ROOT, digest, dump, shape_of
from optimizer.checker import check_candidate
from optimizer.compiler import compile_candidate, simulate_candidate
from optimizer.formulation import index_proof, solve, validate
from optimizer.model import ModelConfig, generate
from optimizer.trace import Trace


def add_usage(totals,raw):
    values={
        'input_tokens':raw.get('input_tokens',raw.get('prompt_tokens')),
        'output_tokens':raw.get('output_tokens',raw.get('completion_tokens')),
        'cached_tokens':raw.get('cache_read_input_tokens',raw.get('prompt_tokens_details',raw.get('input_tokens_details',{})).get('cached_tokens')),
        'reasoning_tokens':raw.get('completion_tokens_details',raw.get('output_tokens_details',{})).get('reasoning_tokens'),
        'cache_creation_tokens':raw.get('cache_creation_input_tokens'),
    }
    for name,value in values.items():
        totals[name]=None if value is None or totals[name] is None else totals[name]+value


def choose_best(best,candidate):
    """Keep a validated source even if simulation fails; rank only valid estimates."""
    if not candidate['eligible']:return best
    if best is None:return candidate
    value=candidate.get('simulated_us');previous=best.get('simulated_us')
    return candidate if value is not None and (previous is None or value<previous) else best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',type=Path,required=True)
    parser.add_argument('--model-config',type=Path,default=ROOT/'configs/deepseek.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--budget-seconds',type=int,default=1200)
    args=parser.parse_args()
    if not 1<=args.rounds<=20 or not 1<=args.budget_seconds<=1200:parser.error('rounds 1..20, budget 1..1200')
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):parser.error('Output must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+args.budget_seconds;trace=Trace(out)
    usage=dict(model_calls=0,input_tokens=0,output_tokens=0,cached_tokens=0,reasoning_tokens=0,cache_creation_tokens=0,
               evaluations=0,compilations=0,simulations=0,numerical_checks=0,solver_calls=0)
    attempts=[];history=[];excluded=[];best=None;initial=None;exit_code=0
    result=dict(schema_version='agentic-result-v1',workload_id=None,status='no_valid_result',implementation=None,
                evidence=dict(correctness='checks.json',evaluation='evaluation.json'),usage=usage,
                reason='No validated source/configuration with matching CUDA input IR yet')

    def checkpoint():
        result['elapsed_ms']=round((time.monotonic()-start)*1000)
        if best is not None:
            result.update(status='checked',implementation=dict(path=best['implementation'],factory='build',config=best['configuration']),
                          checked_target='metal',cuda_binary_executed=False,
                          validation_scope='Same TileLang source/configuration/IR tested on Metal; GB10 SASS timing simulated, CUDA binary not executed',
                          identity=best['identity'],selected_attempt=best['attempt'],initial_checked_attempt=initial['attempt'])
            result.pop('reason',None)
        dump(out/'checks.json',dict(scope='Actual TileLang Metal execution; not CUDA binary execution',attempts=[dict(attempt=a['attempt'],**a['correctness']) for a in attempts if 'correctness' in a]))
        dump(out/'evaluation.json',dict(source='gb10_instruction_simulator',unit='us',value=best.get('simulated_us') if best else None,
                                       selected_attempt=best['attempt'] if best else None,attempts=attempts))
        dump(out/'state/search.json',dict(history=history,excluded=excluded,best=best,initial=initial))
        dump(out/'result.json',result)

    def remaining():
        seconds=deadline-time.monotonic()
        if seconds<=0:raise TimeoutError('Wall-clock budget exhausted')
        return seconds

    def alarm(signum,frame):raise TimeoutError('Wall-clock budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,args.budget_seconds)
    checkpoint()
    try:
        workload=json.loads(args.workload.read_text());shape=shape_of(workload)
        result['workload_id']=workload['id'];model=ModelConfig.load(args.model_config)
        dump(out/'state/workload.json',workload)
        dump(out/'state/model-config.json',model.__dict__)
        case_files=[]
        for name in ['reference.py','baseline.py','README.md']:
            path=args.workload.parent/name
            if path.is_file():
                destination=out/'state/case'/name;destination.parent.mkdir(exist_ok=True,parents=True)
                shutil.copyfile(path,destination);case_files.append(dict(path='state/case/'+name,sha256=digest(path)))
        prompt=(ROOT/'prompts/formulate.txt').read_text()
        files=[*sorted((ROOT/'optimizer').glob('*.py')),ROOT/'templates/gemm_tilelang.py',ROOT/'prompts/formulate.txt',
               ROOT/'tools/compiler_worker.py',ROOT/'tools/simulate_audited.py',ROOT/'setup/compiler.Dockerfile',
               ROOT/'setup/compiler-python.lock',ROOT/'setup/mac-numeric.lock',ROOT/'requirements-cpu.lock',ROOT/'run.sh']
        resources={str(p.relative_to(ROOT)):digest(p) for p in files}
        for name in resources:
            target=out/'state/source'/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(ROOT/name,target)
        dump(out/'state/manifest.json',dict(resources=resources,case_files=case_files,
             prebuilt='Kernel family, numerical tests, MIP feature/constraint builder, Z3 bounds checks, compiler/simulator adapters',
             runtime_decisions='Agent domains/resource cap/objective weights; HiGHS configuration; feedback revisions; best candidate',
             verification='Metal source-level check plus identical CUDA input TIR; no CUDA binary execution',
             case_policy='GEMM semantics checked using independent CPU float64 reference; supplied baseline retained but replaced by disclosed family',
             graph_optimization='Single GEMM: no graph fusion opportunity; TileLang uses its bundled TVM lowering'))
        trace.record('workload_loaded',artifacts=['state/workload.json','state/model-config.json','state/manifest.json'])
        checkpoint()
        for i in range(args.rounds):
            remaining();tag=f'attempt-{i:02d}'
            agent=out/'agent';agent.mkdir(exist_ok=True)
            request=prompt+'\nWorkload:\n'+json.dumps(workload)+'\nObserved history:\n'+json.dumps(history)
            request_path=agent/(tag+'-request.txt');request_path.write_text(request)
            usage['model_calls']+=1
            trace.record('model_started',attempt=tag,artifacts=[str(request_path.relative_to(out))])
            response_path=agent/(tag+'-response.json')
            try:
                response=generate(model,request,timeout=min(100,remaining()),max_output_tokens=4096)
                dump(response_path,response);add_usage(usage,response.get('usage') or {})
                text=response['text'].strip()
                if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
                plan=validate(json.loads(text))
            except TimeoutError:
                add_usage(usage,{})
                raise
            except Exception as exc:
                if not response_path.exists():add_usage(usage,{})
                observation=dict(attempt=tag,stage='model_or_schema',error_type=type(exc).__name__)
                history.append(observation);attempts.append(observation)
                trace.record('model_failed',**observation,artifacts=[str(request_path.relative_to(out))]+([str(response_path.relative_to(out))] if response_path.exists() else []))
                checkpoint();continue
            directory=out/'formulations'/tag;directory.mkdir(parents=True)
            dump(directory/'plan.json',plan);usage['solver_calls']+=1
            try:config,solver=solve(plan,shape,directory,min(20,remaining()),excluded=excluded,hardware_limit=32768)
            except ValueError as exc:
                observation=dict(attempt=tag,stage='formulation',reason=str(exc),plan=plan)
                history.append(observation);attempts.append(observation)
                trace.record('formulation_rejected',attempt=tag,artifacts=[f'formulations/{tag}/plan.json'],reason=str(exc))
                checkpoint();continue
            dump(directory/'solver.json',solver)
            trace.record('highs_completed',attempt=tag,artifacts=[f'agent/{tag}-response.json',*[f'formulations/{tag}/{f}' for f in ['plan.json','model.lp','model.mps','solver.json']]])
            if config is None:
                history.append(dict(attempt=tag,stage='MIP',status=solver['status'],plan=plan));checkpoint();continue
            excluded.append(config)
            impl=out/'implementation'/tag;impl.mkdir(parents=True)
            shutil.copyfile(ROOT/'templates/gemm_tilelang.py',impl/'kernel.py');dump(impl/'config.json',config)
            identity=dict(source_sha256=digest(impl/'kernel.py'),config_sha256=digest(impl/'config.json'))
            check_dir=out/'checks'/tag;check_dir.mkdir(parents=True)
            proof=index_proof(shape,config,identity['source_sha256']);proof['config_sha256']=identity['config_sha256']
            dump(check_dir/'smt.json',proof)
            smt_ok=all(x['status']=='unsat' for x in proof['results'])
            evaluation=dict(attempt=tag,configuration=config,identity=identity,smt=proof,
                            compilation=dict(status='not_run'),simulation=dict(status='not_run',value=None,unit='us'),
                            correctness=dict(status='not_run',passed=False))
            attempts.append(evaluation)
            trace.record('implementation_generated',attempt=tag,artifacts=[f'implementation/{tag}/kernel.py',f'implementation/{tag}/config.json',f'checks/{tag}/smt.json'])
            if smt_ok:
                usage['numerical_checks']+=1
                evaluation['correctness']=check_candidate(out/'state/workload.json',impl/'config.json',impl/'kernel.py',check_dir,min(180,remaining()))
                trace.record('numerical_check_completed',attempt=tag,status=evaluation['correctness']['status'],artifacts=[f'checks/{tag}/numeric.json'])
                checkpoint()
                if evaluation['correctness'].get('passed'):
                    usage['compilations']+=1
                    compiled_dir=out/'compilation'/tag
                    evaluation['compilation']=compile_candidate(out/'state/workload.json',impl/'config.json',impl/'kernel.py',compiled_dir,min(180,remaining()))
                    trace.record('compilation_completed',attempt=tag,status=evaluation['compilation']['status'],artifacts=[f'compilation/{tag}/compilation.json'])
                    comp=evaluation['compilation'];numeric=evaluation['correctness']
                    match=comp['status']=='compiled' and digest(compiled_dir/'input.tir')==numeric['tir_sha256'] and comp['source_sha256']==numeric['source_sha256']==identity['source_sha256'] and comp['config_sha256']==numeric['config_sha256']==identity['config_sha256']
                    evaluation['identical_input_tir']=match
                    if match:
                        identity.update(tir_sha256=numeric['tir_sha256'],sass_sha256=comp['sass_sha256'],cubin_sha256=comp['cubin_sha256'])
                        candidate=dict(attempt=tag,eligible=True,implementation=f'implementation/{tag}/kernel.py',configuration=config,identity=identity,simulated_us=None)
                        if initial is None:initial=candidate.copy()
                        best=choose_best(best,candidate);checkpoint()
                        usage['simulations']+=1;usage['evaluations']+=1
                        simulation=simulate_candidate(compiled_dir,min(120,remaining()));evaluation['simulation']=simulation
                        if simulation['status']=='completed':candidate['simulated_us']=simulation['value']
                        if initial['attempt']==tag:initial=candidate.copy()
                        best=choose_best(best,candidate)
                        trace.record('simulation_completed',attempt=tag,status=simulation['status'],artifacts=[f'compilation/{tag}/simulation.json'])
            observation=dict(attempt=tag,plan=plan,selection=config,proxy_cost=solver['incumbent'],solver_status=solver['status'],
                             smt_bounds_passed=smt_ok,numerical_status=evaluation['correctness']['status'],
                             numerical_failure=evaluation['correctness'].get('reason'),
                             compilation_status=evaluation['compilation']['status'],compilation_reason=evaluation['compilation'].get('reason'),
                             actual_resources={k:evaluation['compilation'].get(k) for k in ['registers','smem_bytes','threads']},
                             simulation_status=evaluation['simulation']['status'],simulated_us=evaluation['simulation'].get('value'),
                             best_simulated_us=best.get('simulated_us') if best else None)
            history.append(observation);dump(out/'state/latest-feedback.json',observation)
            dump(check_dir/'evaluation.json',evaluation)
            trace.record('feedback_saved',attempt=tag,artifacts=['state/latest-feedback.json',f'checks/{tag}/evaluation.json'])
            checkpoint()
    except TimeoutError:
        result['stop_reason']='Wall-clock budget reached';trace.record('deadline_reached')
    except Exception as exc:
        result['stop_reason']=type(exc).__name__+': '+str(exc)[:500]
        if best is None:result.update(status='error',reason=result['stop_reason']);exit_code=1
        trace.record('execution_error',error_type=type(exc).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        checkpoint();trace.record('run_completed',status=result['status'],artifacts=['result.json','checks.json','evaluation.json','state/search.json'])
    print(json.dumps(dict(status=result['status'],workload_id=result['workload_id'],elapsed_ms=result['elapsed_ms'],usage=usage)))
    return exit_code


if __name__=='__main__':raise SystemExit(main())
