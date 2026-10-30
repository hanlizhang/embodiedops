"""Exercise installed native reduction without constructing a learned model."""
import unittest
from types import SimpleNamespace
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class NativeMaskTests(unittest.TestCase):
    def test_time_and_dimension_masks_have_zero_gradient(self):
        loss_tensor=torch.full((1,50,32),1000.,requires_grad=True)
        with torch.no_grad():loss_tensor[:,:13,:12]=1.
        dummy=SimpleNamespace(config=SimpleNamespace(adapt_to_pi_aloha=False,
            action_feature=SimpleNamespace(shape=(12,)),max_action_dim=32),
            prepare_images=lambda b:([],[]),prepare_state=lambda b:None,
            prepare_action=lambda b:torch.zeros((1,50,32)),
            model=SimpleNamespace(forward=lambda *a:loss_tensor))
        batch={'observation.language.tokens':None,'observation.language.attention_mask':None,
            'action_is_pad':torch.arange(50)[None]>=13}
        value,_=SmolVLAPolicy.forward(dummy,batch)
        self.assertEqual(value.item(),1.)
        value.backward()
        self.assertEqual(torch.count_nonzero(loss_tensor.grad[:,13:]).item(),0)
        self.assertEqual(torch.count_nonzero(loss_tensor.grad[:,:,12:]).item(),0)
        self.assertEqual(torch.count_nonzero(loss_tensor.grad).item(),13*12)


if __name__=='__main__':unittest.main()
