"""Integration test: a real Metal shader must fail after an arithmetic mutation."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT/'.venv-metal/bin/python').exists(),'Requires declared Mac Metal setup')
class ActualNumericTests(unittest.TestCase):
    def test_generated_program_and_corrupted_program_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            d=Path(temporary)
            w=json.loads((ROOT/'third_party/gb10/workloads/gemm_32x4096x4096.json').read_text())
            w['inputs']['A']['shape']=[32,64];w['inputs']['B']['shape']=[64,64];w['outputs']['C']['shape']=[32,64]
            (d/'workload.json').write_text(json.dumps(w))
            (d/'config.json').write_text(json.dumps(dict(block_M=32,block_N=64,block_K=32,num_stages=2,threads=128)))
            source=(ROOT/'templates/gemm_tilelang.py').read_text()
            for name,code,passed in [('good',source,True),('corrupted',source.replace('=c_s[i,j].astype(T.float16)','=c_s[i,j].astype(T.float16)+1'),False)]:
                path=d/(name+'.py');path.write_text(code);out=d/name/'numeric.json'
                process=subprocess.run([str(ROOT/'.venv-metal/bin/python'),'-m','optimizer.tilelang_numeric','--workload',str(d/'workload.json'),
                    '--config',str(d/'config.json'),'--source',str(path),'--output',str(out)],cwd=ROOT,capture_output=True,text=True,timeout=45)
                self.assertEqual(process.returncode,0,process.stderr)
                report=json.loads(out.read_text());self.assertEqual(report['passed'],passed,report)
                self.assertEqual(report['status'],'passed' if passed else 'failed')
                self.assertEqual(len(report['cases']),3)
                self.assertFalse(report['cuda_binary_executed'])


if __name__=='__main__':unittest.main()
