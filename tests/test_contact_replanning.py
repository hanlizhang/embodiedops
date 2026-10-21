"""Controlled feedback and installed native queue regression; no policy inference."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contact_replanning import Rule, Supervisor, interrupt, contact_evidence, tracking_evidence, valid_cameras
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class FeedbackTest(unittest.TestCase):
    def decision(self, s, i, loaded=True, motion=0., safe=True):
        return s.update(i, dict(target_error_m=.02, achieved_motion_m=motion),
                        dict(peak_normal_N=dict(robot_fridge=2. if loaded else 0.)), 30, safe)

    def test_detection_hysteresis_and_determinism(self):
        logs = []
        for _ in range(2):
            s = Supervisor(Rule())
            log = [self.decision(s, i) for i in range(1, 101)]
            self.assertEqual([r['step'] for r in log if r['trigger']], [5])
            log += [self.decision(s, i, loaded=False) for i in range(101, 106)]
            log += [self.decision(s, i) for i in range(106, 111)]
            self.assertTrue(log[-1]['trigger'])
            logs.append(json.dumps(log, sort_keys=True))
        self.assertEqual(*logs)

    def test_free_space_movement_and_safety(self):
        for kwargs in (dict(loaded=False), dict(motion=.002), dict(safe=False)):
            s = Supervisor(Rule())
            self.assertFalse(any(self.decision(s, i, **kwargs)['trigger'] for i in range(1, 100)))
        self.assertFalse(valid_cameras({'pixels': {'left': np.zeros((3, 3, 3))}}))
        self.assertFalse(valid_cameras({'pixels': {'left': np.full((3, 3, 3), np.nan)}}))

    def test_event_cap_and_cooldown(self):
        s = Supervisor(Rule(max_events=2))
        fired = []
        for i in range(1, 200):
            d = self.decision(s, i, loaded=(i % 10 < 5))
            if d['trigger']:
                fired.append(i)
        self.assertEqual(len(fired), 2)
        self.assertGreaterEqual(fired[1]-fired[0], 20)

    def test_contacts(self):
        cs = [dict(geom_ids=p, constraint_address=0, force_N_contact_frame=[f, 0, 0])
              for p, f in [([1, 2], 3), ([1, 3], 4), ([1, 4], 5)]]
        cs.append(dict(geom_ids=[1, 3], constraint_address=-1, force_N_contact_frame=[999, 0, 0]))
        result = contact_evidence([{'robot_contacts': cs}], {1, 2}, {3})
        self.assertEqual(result['peak_normal_N'], dict(robot_self=3, robot_fridge=4, robot_environment=5))

    def test_frame_and_actual_target(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        t = dict(origin_ori=rotation, origin_pos=[10., 10, 0], goal_pos=[.1, 0, 0], _goal_update_mode='desired')
        trace = dict(targets={'right': t}, before={'robot0_eef_pos': [10., 10, 0]},
                     after={'robot0_eef_pos': [10., 10.001, 0]})
        e = tracking_evidence(trace, np.zeros(12))
        np.testing.assert_allclose(e['achieved_delta_m'], [.001, 0, 0], atol=1e-12)
        self.assertAlmostEqual(e['target_error_m'], .099)

    def test_native_queue_h50_and_interruption(self):
        for enabled in (False, True):
            calls = []
            def chunk(batch, noise):
                calls.append(batch['observation.state'].item())
                return torch.arange(50).reshape(1, 50, 1)
            p = SimpleNamespace(config=SimpleNamespace(n_action_steps=50), _rtc_enabled=lambda: False,
                                eval=lambda: None, _prepare_batch=lambda b: b, _get_action_chunk=chunk)
            SmolVLAPolicy.reset(p)
            p._check_get_actions_condition = lambda: not p._queues['action']
            for i in range(101):
                v = SmolVLAPolicy.select_action(p, {'observation.state': torch.tensor(i)})
                if enabled and i == 4:
                    interrupt(p, dict(trigger=True, actions_discarded=45))
                if not enabled:
                    self.assertEqual(v.item(), i % 50)
            self.assertEqual(calls, [0, 5, 55] if enabled else [0, 50, 100])

    def test_runner_guard_order(self):
        import inspect
        import phase1d_rollout
        source = inspect.getsource(phase1d_rollout.main)
        self.assertLess(source.index("invalid_observation_black_camera"), source.index('policy.select_action(batch)'))
        self.assertIn("safe = not trace['guard_failures'] and feedback.valid_cameras(obs)", source)
        self.assertIn('gate = action_gate(env, action)', source)
        self.assertIn('feedback.interrupt(policy, decision)', source)


if __name__ == '__main__':
    unittest.main()
