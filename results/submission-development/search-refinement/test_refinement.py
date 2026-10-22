import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('refined',Path(__file__).with_name('formulation_refined.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
class RefinementTests(unittest.TestCase):
 def test_canonical_equivalence_against_full_launch_mapping(self):
  for rows in [1,2,3,9,32,64,128]:
   for cols in [1,3,64]:
    for panel in [0,4,8,16,32,64]:
     shape=(rows*16,cols*32,64);c=dict(block_M=16,block_N=32,swizzle=panel)
     canonical=f.canonical_swizzle(shape,c)
     self.assertEqual([f.launch_coordinate(i,rows,cols,panel) for i in range(rows*cols)],
                      [f.launch_coordinate(i,rows,cols,canonical) for i in range(rows*cols)])
 def test_equivalent_launches_are_excluded_before_real_highs_solve(self):
  plan=dict(domains=dict(block_M=[32],block_N=[64],block_K=[32],num_stages=[1],threads=[128],swizzle=[0,4,8,16,32,64]),shared_memory_limit=32768,weights=dict(traffic=1,launches=1,pipeline=1,footprint=1),rationale='test')
  with tempfile.TemporaryDirectory() as d:
   selected,r=f.solve(plan,(32,64,64),Path(d));self.assertEqual(len(r['candidates']),1)
   with self.assertRaises(ValueError):f.solve(plan,(32,64,64),Path(d),excluded=[dict(selected,swizzle=64)])
 def test_constant_divisor_proof_handles_tail_and_oversize_panels(self):
  for rows in [1,2,3,9,32,64,128]:
   for panel in [4,8,16,32,64]:
    p=f.index_proof((rows*16,96,64),dict(block_M=16,block_N=32,block_K=32,swizzle=panel),'test-only')
    self.assertTrue(all(r['status']=='unsat' for r in p['results']),p)
    self.assertTrue(p['finite_mapping_check']['bijective'])
if __name__=='__main__':unittest.main()
