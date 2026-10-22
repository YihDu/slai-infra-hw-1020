"""Invoke a prepared offline compiler container; never install during a workload."""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def compile_candidate(workload,config,source,output,timeout=180):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Compiler output must be new or empty; preserve earlier evidence')
    output.mkdir(parents=True,exist_ok=True)
    # Stage inputs in output so callers can use arbitrary workload/output paths.
    import shutil
    for src,name in [(workload,'workload.json'),(config,'config.json'),(source,'kernel.py')]:
        shutil.copyfile(src,output/name)
    context=os.environ.get('SLAI_DOCKER_CONTEXT','colima-slai-compiler')
    image=os.environ.get('SLAI_COMPILER_IMAGE','slai-gb10-compiler:0.1.15')
    name='slai-compile-'+uuid.uuid4().hex
    image_id=None
    cmd=['docker','--context',context,'run','--rm','--name',name,'--pull=never','--network=none','--memory=4g','--cpus=4',
         '--cap-drop=ALL','--mount',f'type=bind,src={ROOT},dst=/work,readonly',
         '--mount',f'type=bind,src={output},dst=/out',image,'python','/work/tools/compiler_worker.py',
         '--workload','/out/workload.json','--config','/out/config.json','--source','/out/kernel.py','--output','/out']
    try:
        inspection=subprocess.run(['docker','--context',context,'image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=min(10,timeout))
        image_id=inspection.stdout.strip() or None
        process=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        (output/'container.log').write_text(process.stdout+process.stderr)
        path=output/'compilation.json'
        report=json.loads(path.read_text()) if path.exists() else dict(status='error',reason='Compiler container failed',exit_code=process.returncode)
    except (subprocess.TimeoutExpired,OSError) as exc:
        report=dict(status='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'unavailable',reason=type(exc).__name__)
    finally:
        # Killing the Docker client alone does not stop its container.
        try:subprocess.run(['docker','--context',context,'rm','-f',name],capture_output=True,timeout=10)
        except (subprocess.TimeoutExpired,OSError):pass
    report['container_image']=image
    report['container_image_id']=image_id
    (output/'compilation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def simulate_candidate(output,timeout=120):
    output=Path(output)
    if (output/'simulation.json').exists():
        raise ValueError('Simulation evidence already exists; use a new candidate directory')
    cmd=[sys.executable,str(ROOT/'tools/simulate_audited.py'),'--input',str(output/'simulation-input.json'),
         '--output',str(output/'simulation.json')]
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        (output/'simulation.log').write_text(p.stdout+p.stderr)
        path=output/'simulation.json'
        report=json.loads(path.read_text()) if path.exists() else dict(status='error',reason='Simulator failed',value=None,unit='us')
    except subprocess.TimeoutExpired:
        report=dict(status='timeout',reason='Simulator wall-clock limit',value=None,unit='us')
    (output/'simulation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
