import unittest,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'experiments/exp1_factor_reuse_ablation'))
from frozen_cases import load
from linear_system import _kkt_apply
class ArtifactTests(unittest.TestCase):
    def test_all_frozen_inputs(self):
        from verify_artifact import check
        self.assertEqual(check(),96)
    def test_kkt_action_symmetric(self):
        c=load(ROOT/'experiments/exp1_factor_reuse_ablation/data/cartpole/seed_000.npz')
        rng=np.random.default_rng(17);x=rng.standard_normal(c.total_dimension);y=rng.standard_normal(c.total_dimension)
        # Independent inner-product identity checks the actual recorded operator.
        a=_kkt_apply(c,x[:,None]).ravel()
        b=_kkt_apply(c,y[:,None]).ravel()
        self.assertAlmostEqual(float(y@a),float(x@b),places=9)
    def test_lazy_binding(self):
        from rapidoc import solver
        self.assertIsNone(solver._library)
        self.assertIn('factor_reuse_solver',solver.DLL.name)
    def test_paper_and_video(self):
        self.assertTrue((ROOT/'paper/RapidDOC.pdf').read_bytes().startswith(b'%PDF'))
        self.assertIn(b'ftyp',(ROOT/'assets/RapidDOC_demo.mp4').read_bytes()[:64])
        self.assertIn('https://youtu.be/b53gcrjdS7c',(ROOT/'README.md').read_text(encoding='utf-8'))
if __name__=='__main__':unittest.main()
