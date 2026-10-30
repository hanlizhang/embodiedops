"""Targeted preflight checks; no robot episode, rendering or policy execution."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recovery_v1b import peak_contacts, check_direction


class RecoveryV1bTest(unittest.TestCase):
    def test_reversed_approach_is_rejected(self):
        action=np.zeros(12); action[5]=.2
        decision=dict(stage='approach',nearest_points=[0,0,0,1,0,0])
        check_direction(action,decision,np.eye(3))
        with self.assertRaisesRegex(AssertionError,'Reversed'):
            check_direction(-action,decision,np.eye(3))

    def test_substep_peak_and_first_loaded_door_contact(self):
        def contact(ids,force,address=0):
            return dict(geom_ids=ids,force_N_contact_frame=[force,0,0],constraint_address=address)
        samples=[dict(time=25.51,robot_contacts=[contact([1,3],.5),contact([1,2],100)]),
                 dict(time=25.512,robot_contacts=[contact([1,3],30),contact([1,4],900,-1)])]
        peaks,first=peak_contacts(samples,{1,2},{3,4},{3},{1})
        self.assertEqual(peaks,dict(all=100,fridge=30,self=100,door=30))
        self.assertEqual(first,25.512)

    def test_no_contact_has_no_contact_time(self):
        peaks,first=peak_contacts([],set(),set(),set(),set())
        self.assertIsNone(first)
        self.assertTrue(all(v==0 for v in peaks.values()))


if __name__=='__main__': unittest.main()
