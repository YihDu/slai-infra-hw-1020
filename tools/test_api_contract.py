#!/usr/bin/env python3
"""TEST ONLY: local protocol fixtures + real kernel/check/compiler/simulator loop.

This server returns a fixed test plan, not an optimization model or benchmark.
It is never used by the production entry point and requires no provider credentials.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from validate_submission import verify_run

PLAN = dict(domains=dict(block_M=[32],block_N=[64],block_K=[32],num_stages=[2],threads=[128],swizzle=[0]),
            shared_memory_limit=32768,weights=dict(traffic=1,launches=1,pipeline=1,footprint=1),rationale='Protocol contract fixture, not Agent optimization evidence')
ROUTES = {'responses':'openai_responses','chat':'openai_chat_completions',
          'messages':'anthropic_messages','chat-legacy':'openai_chat_completions'}
FIXTURE_KEY = 'contract-test-placeholder'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):parser.error('Output must be new or empty')
    out.mkdir(parents=True,exist_ok=True);requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,status,payload):
            data=json.dumps(payload).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_POST(self):
            try:
                route=self.path.split('/')[1];protocol=ROUTES[route]
                expected='/'+route+'/v1'+{'openai_responses':'/responses','openai_chat_completions':'/chat/completions','anthropic_messages':'/messages'}[protocol]
                assert self.path==expected,'Incorrect endpoint'
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert body['model']=='ta-model-'+route,'Configured model ignored'
                if protocol=='anthropic_messages':
                    assert self.headers.get('x-api-key')==FIXTURE_KEY,'Configured credential ignored'
                    assert self.headers.get('anthropic-version')=='2023-06-01','Missing API version'
                    assert type(body['max_tokens']) is int
                else:assert self.headers.get('Authorization')=='Bearer '+FIXTURE_KEY,'Configured credential ignored'
                prompt=body['input'] if protocol=='openai_responses' else body['messages'][0]['content']
                assert isinstance(prompt,str) and prompt
                field=next(f for f in ['max_output_tokens','max_completion_tokens','max_tokens'] if f in body)
                reject=route=='chat-legacy' and field=='max_completion_tokens'
                record=dict(route=route,protocol=protocol,path=self.path,model=body['model'],auth_verified=True,token_field=field,status_code=400 if reject else 200)
                requests.append(record)
                with (out/'http-requests.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if reject:
                    self.respond(400,dict(error=dict(code='unsupported_parameter',param='max_completion_tokens',message='Unsupported parameter')));return
                text=json.dumps(PLAN)
                usage=dict(input_tokens=11,output_tokens=7)
                if protocol=='openai_responses':payload=dict(id='fixture-responses',status='completed',output=[dict(type='message',content=[dict(type='output_text',text=text)])],usage=usage)
                elif protocol=='openai_chat_completions':payload=dict(id='fixture-chat',choices=[dict(message=dict(role='assistant',content=text),finish_reason='stop')],usage=dict(prompt_tokens=11,completion_tokens=7))
                else:payload=dict(id='fixture-messages',content=[dict(type='text',text=text)],stop_reason='end_turn',usage=usage)
                self.respond(200,payload)
            except Exception as exc:
                self.respond(400,dict(error=dict(message='Contract assertion failed: '+type(exc).__name__)))
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env={k:os.environ[k] for k in ['PATH','HOME','TMPDIR','LANG','LC_ALL','SLAI_DOCKER_CONTEXT','SLAI_COMPILER_IMAGE','SLAI_NUMERIC_PYTHON'] if k in os.environ}
    env['TA_TEST_API_KEY']=FIXTURE_KEY
    reports=[]
    try:
        with tempfile.TemporaryDirectory() as foreign_cwd:
            for route,protocol in ROUTES.items():
                d=out/route;d.mkdir();config=d/'model.json'
                config.write_text(json.dumps(dict(protocol=protocol,base_url=f'http://127.0.0.1:{server.server_port}/{route}/v1',model='ta-model-'+route,api_key_env='TA_TEST_API_KEY'),indent=2)+'\n')
                prompt=d/'prompt.txt';prompt.write_text('Return a test JSON object.')
                cli=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'tools/model_api.py'),'--model-config',str(config),'--prompt-file',str(prompt)],cwd=foreign_cwd,env=env,capture_output=True,text=True,timeout=30)
                (d/'cli.stdout.json').write_text(cli.stdout);(d/'cli.stderr.log').write_text(cli.stderr)
                assert cli.returncode==0,cli.stderr
                reply=json.loads(cli.stdout);assert json.loads(reply['text'])==PLAN
                workload=json.loads((ROOT/'third_party/gb10/workloads/gemm_32x4096x4096.json').read_text())
                workload.update(id='api_contract_'+route)
                workload['inputs']['A']['shape']=[32,64];workload['inputs']['B']['shape']=[64,64];workload['outputs']['C']['shape']=[32,64]
                path=d/'workload.json';path.write_text(json.dumps(workload,indent=2)+'\n')
                with (d/'runner.log').open('w') as log:
                    run=subprocess.run([str(ROOT/'run.sh'),'--workload',str(path),'--model-config',str(config),'--output',str(d/'run'),'--rounds','1'],cwd=foreign_cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
                assert run.returncode==0,(d/'runner.log').read_text()
                summary=verify_run(d/'run');result=json.loads((d/'run/result.json').read_text())
                assert result['usage']['http_requests']==(2 if route=='chat-legacy' else 1)
                assert result['usage']['model_calls']==1
                saved=json.loads((d/'run/state/model-config.json').read_text())
                assert saved['model']=='ta-model-'+route and saved['api_key_env']=='TA_TEST_API_KEY'
                reports.append(dict(route=route,protocol=protocol,status='passed',numeric_test_cases=summary['numeric_test_cases'],http_requests=result['usage']['http_requests']))
                print(json.dumps(reports[-1]),flush=True)
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)
    report=dict(fixture_only=True,not_performance_evidence=True,server='local HTTP protocol fixture; not an actual TA/provider endpoint',
                credential_source='TA_TEST_API_KEY only; DEEPSEEK_API_KEY absent',cases=reports,all_passed=len(reports)==4,
                numerical_backend='Metal',cuda_binary_executed=False,http_requests=len(requests))
    (out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
