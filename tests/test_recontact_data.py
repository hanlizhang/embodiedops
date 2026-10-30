"""Causal indexing and geometry category checks; no simulator or model execution."""
import sys
from pathlib import Path
import unittest
import math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recontact_self_imitation import contacts,eligible,advance
from recontact_data import window_test


class RecontactTests(unittest.TestCase):
    def test_anchor_past_only(self):
        seen={'door':False};gap={'door':0}
        for _ in range(20):advance(seen,gap,{'door':False})
        self.assertEqual(eligible(seen,gap,False),[])
        advance(seen,gap,{'door':True})
        for _ in range(4):advance(seen,gap,{'door':False})
        self.assertEqual(eligible(seen,gap,False),[])
        advance(seen,gap,{'door':False})
        self.assertEqual(eligible(seen,gap,False),['door'])
        self.assertEqual(eligible(seen,gap,True),[])

    def test_exact_target_and_active_force(self):
        cs=[]
        for ids,f,active in [([1,5],.9,0),([1,5],9,-1),([2,5],40,0),([1,6],30,0),([1,2],70,0)]:
            cs.append(dict(geom_ids=ids,force_N_contact_frame=[f,0,0],constraint_address=active))
        x=contacts([dict(time=1.,robot_contacts=cs)],{1,2},{5,6},{'door':{5}},{1})
        self.assertFalse(x['loaded']['door']);self.assertEqual(x['peaks']['self'],70)
        self.assertEqual(x['peaks']['other_fridge'],40)
        cs.append(dict(geom_ids=[5,1],force_N_contact_frame=[1,0,0],constraint_address=0))
        x=contacts([dict(time=1.,robot_contacts=cs)],{1,2},{5,6},{'door':{5}},{1})
        self.assertTrue(x['loaded']['door'])

    def test_window_excludes_unexecuted_future(self):
        records=[dict(fridge=dict(qpos={'door':math.radians(80-i)},success=i==162)) for i in range(163)]
        loaded=[{'door':i==106} for i in range(163)]
        x=window_test(records,loaded,101,'door')
        self.assertEqual((x['start'],x['end'],x['first_recontact']),(101,150,107))
        self.assertFalse(x['success_in_window'])
        x=window_test(records,loaded,151,'door')
        self.assertEqual(x['executed_actions'],13)
        self.assertTrue(x['success_in_window']);self.assertFalse(x['positive'])


if __name__=='__main__':unittest.main()
