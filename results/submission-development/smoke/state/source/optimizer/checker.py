"""Isolate the native Metal numerical runtime from the lightweight Agent process."""
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check_candidate(workload,config,source,directory,timeout):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    output=directory/'numeric.json'
    if output.exists():raise ValueError('Numerical evidence already exists')
    python=os.environ.get('SLAI_NUMERIC_PYTHON',str(ROOT/'.venv-metal/bin/python'))
    cmd=[python,'-m','optimizer.tilelang_numeric','--workload',str(workload),'--config',str(config),
         '--source',str(source),'--output',str(output)]
    try:
        p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
        (directory/'numeric.log').write_text(p.stdout+p.stderr)
        report=json.loads(output.read_text()) if output.exists() else dict(status='error',passed=False,reason='Numerical subprocess failed',exit_code=p.returncode)
    except (subprocess.TimeoutExpired,OSError) as exc:
        report=dict(status='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'unavailable',passed=False,reason=type(exc).__name__)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report
