import sys
import unittest
from pathlib import Path

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fp32_master import FP32Masters


class MasterTests(unittest.TestCase):
    def test_autograd_accumulates_and_export_remains_native(self):
        model = nn.Sequential(nn.Linear(1, 1, bias=False, dtype=torch.bfloat16))
        with torch.no_grad():
            model[0].weight.fill_(1)
        original = model[0].weight
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        masters = FP32Masters(model)
        self.assertIs(original, opt.param_groups[0]["params"][0])
        for _ in range(64):
            opt.zero_grad()
            loss = model(torch.ones(1, 1, dtype=torch.bfloat16)).float().square().sum()
            loss.backward()
            self.assertEqual(original.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(original.grad).all())
            opt.step()
        self.assertNotEqual(original.item(), 1)
        self.assertNotEqual(model[0].weight.item(), 1)
        self.assertEqual(model[0].weight.dtype, torch.bfloat16)
        self.assertEqual(opt.state[original]["exp_avg"].dtype, torch.float32)
        exported = model.state_dict()
        self.assertEqual(list(exported), ["0.weight"])
        fresh = nn.Sequential(nn.Linear(1, 1, bias=False, dtype=torch.bfloat16))
        fresh.load_state_dict(exported)
        self.assertTrue(torch.equal(fresh[0].weight, model[0].weight))

    def test_initial_forward_and_frozen_parameters_unchanged(self):
        model = nn.Sequential(nn.Linear(2, 2, dtype=torch.bfloat16))
        model[0].bias.requires_grad_(False)
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        before = model(x).detach()
        bias = model[0].bias.clone()
        masters = FP32Masters(model)
        self.assertEqual(set(masters.masters), {"0.weight"})
        self.assertTrue(torch.equal(before, model(x)))
        self.assertTrue(torch.equal(bias, model[0].bias))
        self.assertEqual(model[0].bias.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
