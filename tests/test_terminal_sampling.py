import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from terminal_sampling import anchor_pool, replace_anchors
from terminal_window_experiment import pilot_decision, validate_worker_completion


class TerminalSamplingTests(unittest.TestCase):
    def test_successful_but_late_worker_cannot_continue(self):
        validate_worker_completion("reload", 0, 800, 900)
        with self.assertRaises(TimeoutError):
            validate_worker_completion("reload", 0, 1414.422, 900)
        with self.assertRaises(RuntimeError):
            validate_worker_completion("training", 1, 100, 900)
        validate_worker_completion("physical_guard", 2, 100, 420, allowed=(0, 2))

    def test_reward_transition_and_padding_boundaries(self):
        reward = np.r_[np.zeros(100), np.ones(16)]
        pool = anchor_pool(reward)
        np.testing.assert_array_equal(pool, np.arange(51, 67))
        for frame in pool:
            self.assertEqual(len(reward[frame:frame+50]), 50)
            self.assertEqual(reward[frame], 0)
            self.assertIn(100, range(frame, frame+50))
        self.assertEqual(len(anchor_pool(np.ones(60))), 0)
        self.assertEqual(len(anchor_pool(np.zeros(60))), 0)
        self.assertEqual(len(anchor_pool(np.r_[np.zeros(2), np.ones(16)])), 0)
        self.assertEqual(len(anchor_pool(np.r_[np.zeros(50), np.ones(3), [0], np.ones(16)])), 0)
        with self.assertRaises(ValueError):
            anchor_pool([0, np.nan, 1])

    def test_fallback_does_not_consume_target_rng(self):
        groups = {0: np.arange(100), 1: np.arange(100, 200)}
        pools = {0: np.arange(20, 36), 1: np.array([], dtype=int)}
        original = [dict(index=110, episode=1), dict(index=1, episode=0),
                    dict(index=2, episode=0), dict(index=120, episode=1)]
        plan = replace_anchors(original, groups, pools)
        self.assertEqual([r["episode"] for r in plan], [r["episode"] for r in original])
        self.assertEqual(plan[0]["index"], 110)
        self.assertEqual(plan[1]["index"], 1)
        self.assertEqual(plan[2]["index"], int(np.random.default_rng(242).choice(np.arange(20, 36))))
        self.assertEqual(plan[3]["index"], 120)

    def test_loss_or_average_progress_cannot_hide_robot_regressions(self):
        plan = json.loads((ROOT/"environment/terminal-window-experiment-v1.json").read_text())
        policy = dict(max_door_progress_regression_deg=1,
                      max_peak_force_ratio=1.25, max_peak_force_slack_N=25)
        def metrics(success, angles):
            return dict(valid=True, success=success, guard_failure=False,
                        closing_deg=180-sum(angles), final_angles_deg=angles,
                        peak_contact_normal_N=100)
        pairs = [dict(seed=s, matched_reset=True,
                      baseline=metrics(s == 44, [0, 0] if s == 44 else [20, 20]),
                      candidate=metrics(s in (43, 44), [0, 0] if s in (43, 44) else [20, 20]))
                 for s in (43, 44, 45, 46)]
        self.assertTrue(pilot_decision(pairs, pairs, .36, .3, .4, plan, policy)["passed"])
        # Better total motion still fails if the other door regresses.
        pairs[2]["candidate"] = metrics(False, [0, 22])
        self.assertIn("individual_door_regression",
                      pilot_decision(pairs, pairs, .36, .3, .4, plan, policy)["reasons"])
        pairs[0]["candidate"]["success"] = False
        self.assertIn("incomplete_closing_cases_not_repaired",
                      pilot_decision(pairs, pairs, .2, .2, .4, plan, policy)["reasons"])
        pairs[0]["candidate"]["success"] = True
        pairs[2]["candidate"] = metrics(False, [20, 20])
        self.assertIn("global_validation_regression",
                      pilot_decision(pairs, pairs, .38, .3, .4, plan, policy)["reasons"])
        pairs[0]["candidate"]["peak_contact_normal_N"] = 151
        self.assertIn("native:contact_force_regression",
                      pilot_decision(pairs, pairs, .36, .3, .4, plan, policy)["reasons"])


if __name__ == "__main__":
    unittest.main()
