from collections import defaultdict
from modules.framework.tensor import Tensor
from modules.framework.device import device

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad._data *= 0

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p._data -= self.lr * p.grad.data

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = defaultdict(lambda: None) # First moment estimates
        self.v = defaultdict(lambda: None) # Second moment estimates
        self.t = 0 # Timestep

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is not None:
                grad = p.grad.data
                
                if self.m[p] is None:
                    self.m[p] = device.backend.zeros_like(grad)
                    self.v[p] = device.backend.zeros_like(grad)

                self.m[p] = self.betas[0] * self.m[p] + (1 - self.betas[0]) * grad
                self.v[p] = self.betas[1] * self.v[p] + (1 - self.betas[1]) * (grad * grad)

                m_hat = self.m[p] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[p] / (1 - self.betas[1] ** self.t)

                p._data -= self.lr * m_hat / (device.backend.sqrt(v_hat) + self.eps)