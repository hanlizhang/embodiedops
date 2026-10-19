import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from agent_workflow import DeterministicBackend, acceptance, read, ROOT


class WorkflowTests(unittest.TestCase):
    def test_invalid_observation_stops_before_repair(self):
        d=DeterministicBackend().decide({'results':{'observation_contract':{'valid':False}}},['stop'])
        self.assertEqual(d['tool'],'stop')

    def test_evidence_changes_repair_route_without_seed(self):
        for mismatch,expected in [(True,'configuration_repair'),(False,'trajectory_contact')]:
            state={'results':{'observation_contract':{'valid':True},'instruction_probe':{'mismatch':mismatch}}}
            self.assertEqual(DeterministicBackend().decide(state,[] )['tool'],expected)

    def test_no_failure_stops_learning(self):
        state={'results':{'observation_contract':{'valid':True},'instruction_probe':{'mismatch':False},
                          'trajectory_contact':{'failures':0,'historical_training_sparse':True}}}
        self.assertEqual(DeterministicBackend().decide(state,[])['tool'],'stop')

    def test_success_cannot_override_invalid_or_collision_regression(self):
        policy=read(ROOT/'environment/agent-workflow-v1.json')['acceptance']
        b=dict(valid=True,success=False,closing_deg=1,guard_failure=False,peak_contact_normal_N=10)
        c=dict(b,success=True,closing_deg=80,peak_contact_normal_N=100)
        pair=dict(baseline=b,candidate=c,matched_reset=True)
        self.assertIn('contact_force_regression',acceptance([pair],policy)['reasons'])
        c.update(peak_contact_normal_N=10,valid=False)
        self.assertFalse(acceptance([pair],policy)['passed'])
        c['valid']=True
        self.assertTrue(acceptance([pair],policy)['passed'])
        c['peak_contact_normal_N']=None
        self.assertIn('contact_measurements_unavailable',acceptance([pair],policy)['reasons'])

    def test_no_success_gain_rejected(self):
        p=read(ROOT/'environment/agent-workflow-v1.json')['acceptance']
        b=dict(valid=True,success=False,closing_deg=1,guard_failure=False,peak_contact_normal_N=10)
        self.assertFalse(acceptance([dict(baseline=b,candidate=dict(b,closing_deg=20),matched_reset=True)],p)['passed'])


if __name__=='__main__':unittest.main()
