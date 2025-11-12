import collections
from modules.framework.tensor import Tensor

class Parameter(Tensor):
    """A kind of Tensor that is considered a module parameter."""
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)

class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        self.training = True
        self._modules = collections.OrderedDict()
        self._parameters = collections.OrderedDict()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            yield from module.parameters()

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad._data *= 0 # Efficiently zero out the gradient data

    def to(self, device_type):
        for name, param in self._parameters.items():
            self._parameters[name] = param.to(device_type)
        for name, module in self._modules.items():
            module.to(device_type)
        return self

    def train(self):
        self.training = True
        for name, module in self._modules.items():
            module.train()

    def eval(self):
        self.training = False
        for name, module in self._modules.items():
            module.eval()

    def state_dict(self):
        state = collections.OrderedDict()
        for name, param in self._parameters.items():
            state[name] = param.data
        for name, module in self._modules.items():
            for sub_name, sub_param in module.state_dict().items():
                state[f"{name}.{sub_name}"] = sub_param
        return state

    def load_state_dict(self, state_dict):
        for name, param in self._parameters.items():
            if name in state_dict:
                param._data = state_dict[name]
            else:
                raise KeyError(f"Parameter '{name}' not found in state_dict.")
        for name, module in self._modules.items():
            sub_state_dict = collections.OrderedDict()
            prefix = f"{name}."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    sub_state_dict[key[len(prefix):]] = value
            module.load_state_dict(sub_state_dict)