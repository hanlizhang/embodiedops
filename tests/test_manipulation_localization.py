"""No simulator or policy execution: reject false contact and image diagnoses."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from localize_manipulation import black_views, loaded_steps


class LocalizationTests(unittest.TestCase):
    def test_title_does_not_hide_black_cameras(self):
        frame=np.zeros((288,768,3),dtype=np.uint8)
        frame[:32]=255
        self.assertEqual(black_views(frame),[True,True,True])
        frame[100,300,0]=1
        self.assertEqual(black_views(frame),[True,False,True])

    def test_proximity_and_inactive_constraint_are_not_loaded_contact(self):
        contacts=[{'step':i,'samples':[{'robot_contacts':[{'constraint_address':address,
                    'force_N_contact_frame':[force,0,0],'distance_m':-.001}]}]}
                  for i,(address,force) in enumerate([(-1,10),(0,0),(0,10)],1)]
        self.assertEqual(loaded_steps(contacts),{3})


if __name__=='__main__':
    unittest.main()
