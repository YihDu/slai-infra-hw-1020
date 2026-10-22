import unittest
from optimizer.formulation import index_proof

class SMTFeedbackTests(unittest.TestCase):
    def test_nondivisible_shape_retains_concrete_witness(self):
        proof = index_proof((33, 64, 64), dict(block_M=32, block_N=32, block_K=32, swizzle=0), 'test')
        failure = next(item for item in proof['results'] if item['status'] == 'sat')
        witness = failure['counterexample']
        self.assertGreaterEqual(int(witness['block_M']) * 32 + int(witness['local_M']), 33)
        self.assertIn('violated_property', failure)
