import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from optimizer.cpu_shadow import build
from optimizer.formulation import validate, solve, index_proof
from optimizer.baseline import shape_of
from optimizer.baseline import main
from optimizer.model import ModelConfig


PLAN=dict(domains=dict(block_M=[32,64],block_N=[64],block_K=[32],num_stages=[1,2],threads=[128]),
          shared_memory_limit=16384,weights=dict(traffic=1,launches=1,pipeline=1,footprint=1),rationale='test')


class BaselineTests(unittest.TestCase):
    def test_driver_does_not_promote_shadow_to_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            workload=dict(id='small',operation='C = A @ B',inputs=dict(A=dict(shape=[32,32],dtype='float16'),B=dict(shape=[32,64],dtype='float16')),
                          outputs=dict(C=dict(shape=[32,64],dtype='float16')),accumulation_dtype='float32',tolerance=dict(rtol=.02,atol=.002),
                          evaluation=dict(scope='whole_workload'),baseline_config={})
            (root/'workload.json').write_text(json.dumps(workload))
            response=dict(text=json.dumps(PLAN),usage=dict(prompt_tokens=1,completion_tokens=1))
            argv=['baseline','--workload',str(root/'workload.json'),'--output',str(root/'out'),'--rounds','1']
            with patch('sys.argv',argv),patch('optimizer.baseline.generate',return_value=response):
                self.assertEqual(main(),0)
            result=json.loads((root/'out/result.json').read_text())
            self.assertEqual(result['status'],'no_valid_result');self.assertIsNone(result['implementation'])
            checks=json.loads((root/'out/checks.json').read_text())
            self.assertTrue(checks['attempts'][0]['passed'])
            argv=['baseline','--workload',str(root/'workload.json'),'--output',str(root/'timeout'),'--rounds','1']
            with patch('sys.argv',argv),patch('optimizer.baseline.generate',side_effect=TimeoutError):
                self.assertEqual(main(),0)
            result=json.loads((root/'timeout/result.json').read_text())
            self.assertEqual(result['status'],'no_valid_result')
            self.assertIn('budget',result['reason'])

    def test_real_mip_exports_and_feasible_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            config,report=solve(PLAN,(32,64,64),Path(d))
            self.assertEqual(report['status'],'Optimal')
            self.assertEqual(config['block_M'],32)
            self.assertTrue((Path(d)/'model.mps').is_file())
            selected=report['candidates'][report['selected_index']]
            self.assertLessEqual(selected['smem_bytes_estimate'],PLAN['shared_memory_limit'])
            self.assertEqual(report['gap'],0)

    def test_reject_invalid_model_decision(self):
        p=copy.deepcopy(PLAN);p['weights']['traffic']=float('nan')
        with self.assertRaises(ValueError):validate(p)
        p=copy.deepcopy(PLAN);p['domains']['block_M']=[33]
        with self.assertRaises(ValueError):validate(p)

    def test_smt_detects_unguarded_tail(self):
        c=dict(block_M=32,block_N=64,block_K=32)
        good=index_proof((32,64,64),c,'test')
        bad=index_proof((33,64,64),c,'test')
        self.assertTrue(all(r['status']=='unsat' for r in good['results']))
        self.assertEqual(bad['results'][0]['status'],'sat')

    def test_shadow_full_tail_and_cancellation(self):
        rng=np.random.default_rng(7)
        a=rng.normal(0,.1,(35,67)).astype(np.float16)
        b=rng.normal(0,.1,(67,70)).astype(np.float16)
        c=build({},dict(block_M=32,block_N=64,block_K=32))({'A':a,'B':b})['C']
        ref=(a.astype(np.float64)@b.astype(np.float64)).astype(np.float16)
        np.testing.assert_allclose(c,ref,atol=.002,rtol=.02)
        self.assertEqual(c.dtype,np.float16)

    def test_infeasible_mip_has_no_assignment(self):
        p=copy.deepcopy(PLAN);p['domains'].update(block_M=[128],block_N=[128],block_K=[64],num_stages=[3])
        p['shared_memory_limit']=8192
        with tempfile.TemporaryDirectory() as d:
            config,report=solve(p,(128,128,64),Path(d))
        self.assertIsNone(config);self.assertEqual(report['status'],'Infeasible')


if __name__=='__main__':unittest.main()
