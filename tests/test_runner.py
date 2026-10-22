import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from optimizer.baseline import digest, dump
from optimizer.runner import main, choose_best, add_usage
from optimizer.model import ModelAPIError
from test_baseline import PLAN


class RunnerTests(unittest.TestCase):
    def test_failed_api_attempts_and_available_usage_are_saved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            workload=json.loads((Path(__file__).resolve().parents[1]/'third_party/gb10/workloads/gemm_32x4096x4096.json').read_text())
            dump(root/'workload.json',workload)
            error=ModelAPIError('Model endpoint returned HTTP 400',[dict(status_code=400),dict(status_code=400)],dict(input_tokens=3,output_tokens=0))
            argv=['runner','--workload',str(root/'workload.json'),'--output',str(root/'out'),'--rounds','1']
            with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-placeholder'}),patch('sys.argv',argv),patch('optimizer.runner.generate',side_effect=error):
                self.assertEqual(main(),0)
            result=json.loads((root/'out/result.json').read_text())
            self.assertEqual(result['usage']['model_calls'],1)
            self.assertEqual(result['usage']['http_requests'],2)
            self.assertEqual(result['usage']['input_tokens'],3)
            saved=json.loads((root/'out/agent/attempt-00-error.json').read_text())
            self.assertEqual(saved['request_count'],2)

    def test_schema_failure_reason_reaches_next_model_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            workload=json.loads((Path(__file__).resolve().parents[1]/'third_party/gb10/workloads/gemm_32x4096x4096.json').read_text())
            dump(root/'workload.json',workload)
            requests=[]
            def generate(config,request,**kwargs):
                requests.append(request)
                if len(requests)==2:raise TimeoutError('test ends after inspecting feedback')
                return dict(text=json.dumps(dict(PLAN,shared_memory_limit=65536)),usage={})
            argv=['runner','--workload',str(root/'workload.json'),'--output',str(root/'out'),'--rounds','2']
            with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-placeholder'}),patch('sys.argv',argv),patch('optimizer.runner.generate',side_effect=generate):
                self.assertEqual(main(),0)
            self.assertEqual(len(requests),2)
            self.assertIn('Invalid resource budget',requests[1])

    def test_real_deadline_interrupts_model_wait(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            workload=json.loads((Path(__file__).resolve().parents[1]/'third_party/gb10/workloads/gemm_32x4096x4096.json').read_text())
            dump(root/'workload.json',workload)
            argv=['runner','--workload',str(root/'workload.json'),'--output',str(root/'out'),'--budget-seconds','1']
            start=time.monotonic()
            with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-placeholder'}),patch('sys.argv',argv),patch('optimizer.runner.generate',side_effect=lambda *a,**k:time.sleep(3)):
                self.assertEqual(main(),0)
            self.assertLess(time.monotonic()-start,1.5)
            result=json.loads((root/'out/result.json').read_text())
            self.assertEqual(result['status'],'no_valid_result')
            self.assertIsNone(result['usage']['input_tokens'])
            self.assertIn('budget',result['stop_reason'])

    def test_ranking_requires_validation_and_preserves_best(self):
        a=dict(eligible=True,simulated_us=10)
        self.assertIs(choose_best(a,dict(eligible=False,simulated_us=1)),a)
        self.assertIs(choose_best(a,dict(eligible=True,simulated_us=None)),a)
        b=dict(eligible=True,simulated_us=9)
        self.assertIs(choose_best(a,b),b)

    def test_usage_marks_missing_and_reads_anthropic_cache(self):
        counts=dict.fromkeys(['input_tokens','output_tokens','cached_tokens','reasoning_tokens','cache_creation_tokens'],0)
        add_usage(counts,dict(input_tokens=12,output_tokens=3,cache_read_input_tokens=7,cache_creation_input_tokens=4))
        self.assertEqual(counts['cached_tokens'],7);self.assertEqual(counts['cache_creation_tokens'],4)
        self.assertIsNone(counts['reasoning_tokens'])
        add_usage(counts,{})
        self.assertTrue(all(v is None for v in counts.values()))

    def test_checkpoint_survives_later_timeout_and_binds_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);workload=dict(id='small',operation='C = A @ B',inputs=dict(A=dict(shape=[32,64],dtype='float16'),B=dict(shape=[64,64],dtype='float16')),
                outputs=dict(C=dict(shape=[32,64],dtype='float16')),accumulation_dtype='float32',tolerance=dict(rtol=.02,atol=.002),evaluation=dict(scope='whole_workload'),baseline_config={})
            dump(root/'workload.json',workload)
            def check(w,c,s,d,t):
                d.mkdir(exist_ok=True,parents=True);(d/'input.tir').write_text('test-only IR')
                report=dict(status='passed',passed=True,checked_target='metal',source_sha256=digest(s),config_sha256=digest(c),tir_sha256=digest(d/'input.tir'),cases=[])
                dump(d/'numeric.json',report);return report
            def compile(w,c,s,d,t):
                d.mkdir(exist_ok=True,parents=True);(d/'input.tir').write_text('test-only IR')
                report=dict(status='compiled',source_sha256=digest(s),config_sha256=digest(c),sass_sha256='mock-test-only',cubin_sha256='mock-test-only')
                dump(d/'compilation.json',report);return report
            calls=[]
            def simulate(d,t):
                calls.append(d)
                if len(calls)>1:raise TimeoutError('injected budget alarm')
                report=dict(status='completed',value=10,unit='us');dump(d/'simulation.json',report);return report
            plan=dict(PLAN,shared_memory_limit=32768)
            response=dict(text=json.dumps(plan),usage=dict(prompt_tokens=1,completion_tokens=2))
            argv=['runner','--workload',str(root/'workload.json'),'--output',str(root/'out'),'--rounds','2']
            with patch.dict('os.environ',{'DEEPSEEK_API_KEY':'test-placeholder'}),patch('sys.argv',argv),patch('optimizer.runner.generate',return_value=response),patch('optimizer.runner.check_candidate',side_effect=check),patch('optimizer.runner.compile_candidate',side_effect=compile),patch('optimizer.runner.simulate_candidate',side_effect=simulate):
                self.assertEqual(main(),0)
            r=json.loads((root/'out/result.json').read_text())
            self.assertEqual(r['status'],'checked');self.assertEqual(r['selected_attempt'],'attempt-00')
            self.assertEqual(r['checked_target'],'metal');self.assertFalse(r['cuda_binary_executed'])
            self.assertEqual(r['identity']['source_sha256'],digest(root/'out'/r['implementation']['path']))
            self.assertIn('budget',r['stop_reason'])
            for line in (root/'out/trace.jsonl').read_text().splitlines():
                for artifact in json.loads(line)['artifacts']:self.assertTrue((root/'out'/artifact).exists(),artifact)


if __name__=='__main__':unittest.main()
