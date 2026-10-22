#!/usr/bin/env python3
"""Build an audited ZIP with public runs at the course-required root layout."""
import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path
from validate_submission import ROOT, verify_run, read


def build(results,output):
    expected=sorted(read(p)['id'] for p in (ROOT/'third_party/gb10/workloads').glob('*.json'))
    for name in expected:verify_run(results/name)
    entries={}
    def add(source,name):
        assert source.is_file() and not source.is_symlink(),str(source)
        path=Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert name not in entries,name
        entries[name]=source
    for name in ['run.sh','README.md','report.md','requirements-cpu.lock','requirements-cpu.txt','homework-agentic-optimizer.md']:
        add(ROOT/name,name)
    for folder in ['optimizer','templates','prompts','configs','third_party','tests']:
        for source in (ROOT/folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':add(source,source.relative_to(ROOT).as_posix())
    for name in ['verification-scope.md','iteration-log.md']:
        add(ROOT/'docs'/name,'docs/'+name)
    for name in ['cpu.sh','mac-numeric.sh','mac-numeric.lock','compiler-build.sh','compiler.Dockerfile','compiler-python.lock','toolchain.json','wheels-macos-arm64.json','acceptance.json','submission-tests.log']:
        add(ROOT/'setup'/name,'setup/'+name)
    for name in ['compiler_worker.py','simulate_audited.py','doctor.py','fetch_wheels.py','check_model_config.py','validate_submission.py','build_submission.py','write_submission_report.py']:
        add(ROOT/'tools'/name,'tools/'+name)
    for case in expected:
        for source in (results/case).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':
                add(source,'results/'+case+'/'+source.relative_to(results/case).as_posix())
    if (results/'summary.json').exists():add(results/'summary.json','results/summary.json')
    for folder in ['development-failures','submission-development','mac-feasibility','submission-v1']:
        for source in (ROOT/'results'/folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':
                add(source,'results/history/'+folder+'/'+source.relative_to(ROOT/'results'/folder).as_posix())
    patterns=[rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'sk-[A-Za-z0-9_-]{24,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    sizes=0;hashes={}
    for name,path in entries.items():
        data=path.read_bytes();sizes+=len(data)
        assert not any(re.search(pattern,data) for pattern in patterns),'Credential-like content: '+name
        hashes[name]=hashlib.sha256(data).hexdigest()
    assert sizes<1024**3 and len(entries)+1<=10000
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Refusing to overwrite an existing submission ZIP')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,path in sorted(entries.items()):
            info=zipfile.ZipInfo(name);info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3;info.external_attr=(stat.S_IFREG|(0o755 if name=='run.sh' else 0o644))<<16
            archive.writestr(info,path.read_bytes())
        archive.writestr('SUBMISSION-MANIFEST.json',json.dumps(dict(files=hashes,verification='Metal numerical tests; same input TIR; GB10 SASS simulation; no CUDA binary execution'),indent=2)+'\n')
    assert output.stat().st_size<=256*1024**2
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert {'run.sh','README.md','report.md'}<=set(archive.namelist())
        for item in archive.infolist():
            assert not item.flag_bits&1 and not stat.S_ISLNK(item.external_attr>>16)
            assert not Path(item.filename).is_absolute() and '..' not in Path(item.filename).parts
        assert sum(i.file_size for i in archive.infolist())<=1024**3
    report=dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),compressed_bytes=output.stat().st_size,
                content_bytes=sizes,entries=len(entries)+1,archive_validation='passed',public_cases=expected)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=(ROOT/'results/submission-v2' if (ROOT/'results/submission-v2').exists() else ROOT/'results'))
    parser.add_argument('--output',type=Path,default=ROOT/'dist/slai-infra-hw-1020.zip');a=parser.parse_args()
    print(json.dumps(build(a.results.resolve(),a.output.resolve()),indent=2))
