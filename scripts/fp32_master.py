"""Autograd-connected FP32 masters with native forward-weight precision.

Only trainable BF16 parameters are parametrized. Casting is differentiable;
gradients and Adam moments accumulate in FP32. Policy exports retain the original
names/dtypes, while master values are saved separately for training resumption.
"""
import torch
from torch import nn
from torch.nn.utils import parametrize
from safetensors.torch import load_file, save_file


class NativeCast(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, master):
        return master.to(self.dtype)


class FP32Masters:
    def __init__(self, policy):
        self.policy = policy
        self.parameters = dict(policy.named_parameters())
        self.native_dtypes = {n: p.dtype for n, p in self.parameters.items()}
        self.masters = {}
        self.modules = {}
        for name, parameter in self.parameters.items():
            if not parameter.requires_grad or parameter.dtype != torch.bfloat16:
                continue
            module_path, field = name.rsplit(".", 1)
            module = policy.get_submodule(module_path)
            # Keep Parameter identity: existing optimizer groups remain valid.
            parameter.data = parameter.detach().float()
            parametrize.register_parametrization(module, field, NativeCast(torch.bfloat16), unsafe=True)
            assert module.parametrizations[field].original is parameter
            self.masters[name] = parameter
            self.modules[name] = (module, field)

        def export_hook(module, state, prefix, metadata):
            for name in self.masters:
                parent, field = name.rsplit(".", 1)
                value = state.pop(prefix + parent + ".parametrizations." + field + ".original")
                state[prefix + name] = value.to(self.native_dtypes[name])

        self.export_hook = policy.register_state_dict_post_hook(export_hook)

    def forward_weight(self, name):
        if name in self.modules:
            module, field = self.modules[name]
            return getattr(module, field)
        return self.parameters[name]

    def state(self):
        return {n: p.detach() for n, p in self.masters.items()}

    def save(self, path):
        save_file({n: p.detach().cpu().contiguous() for n, p in self.masters.items()}, str(path))

    def load(self, path):
        values = load_file(str(path), device="cpu")
        assert set(values) == set(self.masters)
        with torch.no_grad():
            for n, value in values.items():
                assert value.dtype == torch.float32 and value.shape == self.masters[n].shape
                native_before = self.forward_weight(n).detach().cpu()
                assert torch.equal(value.to(self.native_dtypes[n]), native_before), n
                self.masters[n].copy_(value)
