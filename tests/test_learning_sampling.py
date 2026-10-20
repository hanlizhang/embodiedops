import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from learning_sampling import build_plan,motion_regimes


class SamplingTests(unittest.TestCase):
    def test_balanced_order_and_reproducibility(self):
        eps=np.repeat(np.arange(9),30);labels=np.tile(np.repeat([0,1,2],10),9)
        a=build_plan(eps,500,42);b=build_plan(eps,500,42,'motion_mixture',labels)
        self.assertEqual([r['episode'] for r in a],[r['episode'] for r in b])
        self.assertEqual(b,build_plan(eps,500,42,'motion_mixture',labels))
        self.assertEqual(len({r['episode'] for r in b[:9]}),9)

    def test_observed_response_not_contact(self):
        a=np.zeros((20,12));s=np.zeros((20,16));a[:,5]=.2
        self.assertTrue(np.all(motion_regimes(a,s)[:-1]==1))
        s[:,7]=np.arange(20)*.01
        self.assertTrue(np.all(motion_regimes(a,s)==2))
        a[:]=0
        self.assertTrue(np.all(motion_regimes(a,s)[:-1]==0))

    def test_invalid_shapes_and_nan_rejected(self):
        with self.assertRaises(ValueError):motion_regimes(np.zeros((2,11)),np.zeros((2,16)))
        a=np.zeros((2,12));a[0,0]=np.nan
        with self.assertRaises(ValueError):motion_regimes(a,np.zeros((2,16)))


if __name__=='__main__':unittest.main()
