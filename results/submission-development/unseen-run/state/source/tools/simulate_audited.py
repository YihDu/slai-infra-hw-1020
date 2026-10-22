#!/usr/bin/env python3
"""Call the unmodified course simulator and reject incomplete warp execution."""
import argparse
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();course=ROOT/'third_party/gb10'
    source=json.loads((course/'SOURCE.json').read_text())
    for name,record in source['files'].items():
        if hashlib.sha256((course/name).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Modified simulator source: '+name)
    sys.path.insert(0,str(course))
    from agentic.gb10.sim import kernel
    from agentic.gb10.sim.sm import SM,kernel_sass
    audits=[]
    class AuditedSM(SM):
        def run(self,*args,**kwargs):
            result=super().run(*args,**kwargs)
            audit=dict(warps=len(self.warps),warps_done=sum(w.done for w in self.warps),cycles=result['cycles'])
            audits.append(audit)
            if audit['warps_done']!=audit['warps']:raise RuntimeError('Simulator stopped with unfinished warps')
            return result
    kernel.SM=AuditedSM
    spec=importlib.util.spec_from_file_location('course_simulate',course/'simulate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    request=json.loads(args.input.read_text())
    report=dict(status='error',source='gb10_instruction_simulator',unit='us',value=None,
                source_commit=source['commit'],audit=audits,scope='Timing simulation only; not numerical correctness')
    try:
        if request['path']!='cpasync':raise ValueError('Only cpasync audit integrated')
        ins=kernel_sass(args.input.parent/request['sass'],request['kernel'])
        report['opcode_counts']=dict(collections.Counter(x.base for x in ins))
        raw=module.simulate(request,args.input.parent)
        report.update(status='completed',value=raw['simulation']['us'],raw=raw)
    except Exception as exc:report.update(error_type=type(exc).__name__,reason=str(exc))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],value=report['value'],unit='us',audit=audits)))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
