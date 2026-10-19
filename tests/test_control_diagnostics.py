"""Physics instrumentation must not perturb dynamics or swallow exceptions."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from control_diagnostics import ContactRecorder


class Sim:
    def __init__(self):
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body name="robot0_test" pos="0 0 .101"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        self.model = SimpleNamespace(_model=m)
        self.data = SimpleNamespace(_data=mujoco.MjData(m), time=0.)

    def step(self):
        mujoco.mj_step(self.model._model, self.data._data)
        self.data.time = self.data._data.time

    def step2(self):
        mujoco.mj_step2(self.model._model, self.data._data)
        self.data.time = self.data._data.time


class DiagnosticsTest(unittest.TestCase):
    def test_dynamics_identical_and_loaded_contacts(self):
        a, b = Sim(), Sim()
        with tempfile.TemporaryDirectory() as temp:
            recorder = ContactRecorder(b, Path(temp))
            with recorder.step(1):
                for _ in range(25):
                    a.step()
                    b.step()
            np.testing.assert_array_equal(a.data._data.qpos, b.data._data.qpos)
            np.testing.assert_array_equal(a.data._data.qvel, b.data._data.qvel)
            rows = json.loads((Path(temp)/'contacts.jsonl').read_text())
            self.assertEqual(len(rows['samples']), 25)
            self.assertTrue(any(c['force_N_contact_frame'][0] > 0 for s in rows['samples'] for c in s['robot_contacts']))

    def test_restore_on_exception(self):
        sim = Sim()
        original = sim.step
        with tempfile.TemporaryDirectory() as temp:
            recorder = ContactRecorder(sim, Path(temp))
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                with recorder.step(1):
                    raise RuntimeError('probe')
            self.assertEqual(sim.step, original)


if __name__ == '__main__':
    unittest.main()
