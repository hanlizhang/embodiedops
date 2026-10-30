"""Fail-closed numerical and rendering comparison without simulator execution."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recovery_feasibility import max_error, camera_error
from recovery_branches import world_delta_to_public
from recovery_geometry_audit import ordered_geometry_distance, XML

class RecoveryGateTest(unittest.TestCase):
    def test_geometry_direction_both_orders_and_ccd_paths(self):
        import mujoco
        for disabled in (False,True):
            model=mujoco.MjModel.from_xml_string(XML)
            if disabled: model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            data=mujoco.MjData(model); mujoco.mj_forward(model,data)
            for first,second,expected in [(0,1,1),(1,0,-1)]:
                distance,points=ordered_geometry_distance(model,data,first,second,10.)
                self.assertAlmostEqual(distance,2.)
                self.assertEqual(np.sign(points[3]-points[0]),expected)
                np.testing.assert_allclose(np.sort([points[0],points[3]]),[1.,3.],atol=1e-6)
    def test_world_base_rotation_and_action_contract(self):
        rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        action=world_delta_to_public([0,.02,0],rotation)
        np.testing.assert_allclose(action[5:8],[.2,0,0])
        np.testing.assert_allclose(rotation@(action[5:8]*.05),[0,.01,0],atol=1e-8)
        np.testing.assert_array_equal(action[:4],np.zeros(4))
        self.assertEqual(action[4],-1); self.assertEqual(action[11],-1)
        with self.assertRaises(AssertionError): world_delta_to_public([0,0,0],np.zeros((3,3)))
    def test_nonfinite_and_shape_rejected(self):
        for actual, expected in [([float('nan')],[0]),([0,1],[0])]:
            with self.assertRaises(AssertionError): max_error(actual,expected)

    def test_drift_measured(self):
        self.assertAlmostEqual(max_error([0,1],[0,1.001]),.001)

    def test_camera_unsigned_difference_and_black(self):
        a=np.full((256,256,3),255,dtype=np.uint8)
        b=a.copy(); b[0,0,0]=1
        self.assertEqual(camera_error(a,b)['max'],254)
        with self.assertRaises(AssertionError): camera_error(a,np.zeros_like(a))

if __name__=='__main__': unittest.main()
