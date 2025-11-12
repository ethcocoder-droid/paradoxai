import numpy as np
from modules.framework.device import device


class _Function:
    """Base class for autograd functions."""
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []
        self.needs_input_grad = None

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

class Tensor:
    def __init__(self, data, device_type=None, requires_grad=False, _ctx=None):
        if device_type is None:
            device_type = device.current_device

        # If data is already a Tensor, extract its underlying data
        if isinstance(data, Tensor):
            data = data._data

        if isinstance(data, (list, tuple)):
            data = device.backend.array(data)
        elif not isinstance(data, (device.backend.ndarray, type(device.backend.array([])))):
            # Handle cases where data might be a CuPy array if device_type is 'gpu' but backend is numpy
            if device_type == 'gpu' and device.backend.__name__ == 'numpy':
                if device.gpu_backend and isinstance(data, device.gpu_backend.ndarray):
                    pass # Already a CuPy array, no conversion needed if device_type is 'gpu'
                else:
                    data = device.backend.array(data)
            else:
                data = device.backend.array(data)

        self._data = data
        self.device = device_type
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx # Context for backward pass

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"Tensor(data={self.data}, device='{self.device}', requires_grad={self.requires_grad})"

    def zero_grad(self):
        if self.grad is not None:
            self.grad._data = device.backend.zeros_like(self.data)
        else:
            self.grad = Tensor(device.backend.zeros_like(self.data), device_type=self.device, requires_grad=False)

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return

        if grad_output is None:
            # For scalar output, grad_output is 1
            if self.data.size == 1:
                grad_output = Tensor(device.backend.array(1.0, dtype=self.dtype), device_type=self.device)
            else:
                raise RuntimeError("grad_output must be specified for non-scalar tensors")

        if self.grad is None:
            self.grad = grad_output
        else:
            self.grad._data += grad_output.data

        if self._ctx:
            grads = self._ctx.backward(grad_output.data)
            if not isinstance(grads, tuple):
                grads = (grads,)

            for i, parent in enumerate(self._ctx.parents):
                if parent.requires_grad:
                    if isinstance(grads[i], Tensor):
                        parent.backward(grads[i])
                    else:
                        parent.backward(Tensor(grads[i], device_type=parent.device))

    def to(self, device_type):
        if device_type == self.device:
            return self

        if device_type == 'cpu':
            new_data = device.to_cpu(self.data)
        elif device_type == 'gpu':
            new_data = device.to_gpu(self.data)
        else:
            raise ValueError(f"Unknown device type: {device_type}")

        return Tensor(new_data, device_type=device_type, requires_grad=self.requires_grad)

    # Placeholder for operations, will be implemented in math_ops.py
    def _ensure_tensor(self, other):
        if not isinstance(other, Tensor):
            return Tensor(other, device_type=self.device, requires_grad=False)
        return other

    def __add__(self, other):
        from modules.framework.ops.math_ops import Add
        other = self._ensure_tensor(other)
        return Add.apply(self, other)

    def __sub__(self, other):
        from modules.framework.ops.math_ops import Sub
        other = self._ensure_tensor(other)
        return Sub.apply(self, other)

    def __mul__(self, other):
        from modules.framework.ops.math_ops import Mul
        other = self._ensure_tensor(other)
        return Mul.apply(self, other)

    def __truediv__(self, other):
        from modules.framework.ops.math_ops import Div
        other = self._ensure_tensor(other)
        return Div.apply(self, other)

    def __pow__(self, other):
        from modules.framework.ops.math_ops import Pow
        # For power, 'other' can be a scalar and doesn't need to be a Tensor for the backward pass to work correctly
        # as it's saved as a scalar. So we only ensure 'self' is a Tensor.
        return Pow.apply(self, other)

    def __matmul__(self, other):
        from modules.framework.ops.math_ops import MatMul
        other = self._ensure_tensor(other)
        return MatMul.apply(self, other)

    def sum(self, axis=None, keepdims=False):
        from modules.framework.ops.math_ops import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def reshape(self, shape):
        from modules.framework.ops.math_ops import Reshape
        return Reshape.apply(self, shape)

    def transpose(self, axes=None):
        from modules.framework.ops.math_ops import Transpose
        return Transpose.apply(self, axes=axes)

    def exp(self):
        from modules.framework.ops.math_ops import Exp
        return Exp.apply(self)

    def log(self):
        from modules.framework.ops.math_ops import Log
        return Log.apply(self)

    def relu(self):
        from modules.framework.ops.math_ops import ReLU
        return ReLU.apply(self)

    @staticmethod
    def uniform(*shape, a=0.0, b=1.0, requires_grad=False):
        data = np.random.uniform(a, b, size=shape if len(shape) > 1 else shape[0])
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def normal(*shape, mean=0.0, std=1.0, requires_grad=False):
        data = np.random.normal(mean, std, size=shape)
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        data = np.ones(shape)
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def zeros(*shape, requires_grad=False):
        data = np.zeros(shape)
        return Tensor(data, requires_grad=requires_grad)

    @staticmethod
    def eye(size, requires_grad=False):
        data = np.eye(size)
        return Tensor(data, requires_grad=requires_grad)

    @property
    def T(self):
        from modules.framework.ops.math_ops import Transpose
        return Transpose.apply(self)