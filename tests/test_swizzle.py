import sys
import unittest
from pathlib import Path
from optimizer.formulation import launch_coordinate,index_proof

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/gb10'))
from agentic.gb10.model import launch_order


class SwizzleTests(unittest.TestCase):
    def test_exact_order_matches_unmodified_course_model_including_partial_panels(self):
        for rows in [1,2,3,9,32,128]:
            for cols in [1,3,64]:
                for panel in [0,4,8,16]:
                    mapping=[launch_coordinate(i,rows,cols,panel) for i in range(rows*cols)]
                    self.assertEqual(mapping,list(launch_order(rows,cols,panel)))
                    self.assertEqual(len(set(mapping)),rows*cols)

    def test_partial_panel_has_smt_bounds_and_complete_bijection_check(self):
        proof=index_proof((288,448,64),dict(block_M=32,block_N=64,block_K=32,swizzle=4),'test-only')
        self.assertTrue(proof['finite_mapping_check']['bijective'])
        self.assertTrue(all(r['status']=='unsat' for r in proof['results']))
        self.assertIn('swizzled_tile_bounds',[r['dimension'] for r in proof['results']])


if __name__=='__main__':unittest.main()
