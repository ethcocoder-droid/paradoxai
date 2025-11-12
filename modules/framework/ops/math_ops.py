from modules.framework.tensor import Tensor, _Function
from modules.framework.device import device

class Add(_Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data.shape, b.data.shape)
        return a.data + b.data

    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        grad_a, grad_b = None, None

        if self.needs_input_grad[0]:
            _grad_a = grad_output
            # Sum gradients along broadcasted dimensions and reshape
            if _grad_a.shape != shape_a:
                # Determine which dimensions were broadcasted and need to be summed
                for i in range(len(_grad_a.shape) - len(shape_a)):
                    _grad_a = device.backend.sum(_grad_a, axis=0)
                for i, dim_a in enumerate(shape_a):
                    if dim_a == 1:
                        _grad_a = device.backend.sum(_grad_a, axis=i, keepdims=True)
                _grad_a = device.backend.reshape(_grad_a, shape_a)
            grad_a = _grad_a

        if self.needs_input_grad[1]:
            _grad_b = grad_output
            # Sum gradients along broadcasted dimensions and reshape
            if _grad_b.shape != shape_b:
                # Determine which dimensions were broadcasted and need to be summed
                for i in range(len(_grad_b.shape) - len(shape_b)):
                    _grad_b = device.backend.sum(_grad_b, axis=0)
                for i, dim_b in enumerate(shape_b):
                    if dim_b == 1:
                        _grad_b = device.backend.sum(_grad_b, axis=i, keepdims=True)
                _grad_b = device.backend.reshape(_grad_b, shape_b)
            grad_b = _grad_b

        return grad_a, grad_b

    @staticmethod
    def apply(a, b):
        ctx = Add(a, b)
        ctx.needs_input_grad = (a.requires_grad, b.requires_grad)
        ctx.save_for_backward(a.shape, b.shape)
        return Tensor(device.backend.add(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad, _ctx=ctx)

class MatMul(_Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data, b.data)
        return device.backend.matmul(a.data, b.data)

    def backward(self, grad):
        a, b = self.saved_tensors
        grad_a, grad_b = None, None
        if self.needs_input_grad[0]:
            grad_a = Tensor(device.backend.matmul(grad.data, device.backend.transpose(b)))
        if self.needs_input_grad[1]:
            grad_b = Tensor(device.backend.matmul(device.backend.transpose(a), grad.data))
        return grad_a, grad_b

    @staticmethod
    def apply(a, b):
        ctx = MatMul(a, b)
        ctx.needs_input_grad = (a.requires_grad, b.requires_grad)
        ctx.save_for_backward(a, b)
        return Tensor(device.backend.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad, _ctx=ctx)

class ReLU(_Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x.data) # Save x for backward pass
        return device.backend.maximum(0, x.data)

    def backward(self, grad_output):
        x_data = self.saved_tensors[0]
        grad_x = grad_output * (x_data > 0).astype(x_data.dtype)
        return grad_x

    @staticmethod
    def apply(x):
        ctx = ReLU(x)
        result_data = ReLU.forward(ctx, x)
        result_tensor = Tensor(result_data, device_type=x.device, requires_grad=x.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Sub(_Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data.shape, b.data.shape)
        return a.data - b.data

    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        grad_a = grad_output
        grad_b = -grad_output

        if grad_a.shape != shape_a:
            for i in range(len(grad_a.shape) - len(shape_a)):
                grad_a = device.backend.sum(grad_a, axis=0)
            for i, dim_a in enumerate(shape_a):
                if dim_a == 1:
                    grad_a = device.backend.sum(grad_a, axis=i, keepdims=True)
            grad_a = device.backend.reshape(grad_a, shape_a)

        if grad_b.shape != shape_b:
            for i in range(len(grad_b.shape) - len(shape_b)):
                grad_b = device.backend.sum(grad_b, axis=0)
            for i, dim_b in enumerate(shape_b):
                if dim_b == 1:
                    grad_b = device.backend.sum(grad_b, axis=i, keepdims=True)
            grad_b = device.backend.reshape(grad_b, shape_b)

        return (grad_a, grad_b)

    @staticmethod
    def apply(a, b):
        ctx = Sub(a, b)
        result_data = Sub.forward(ctx, a, b)
        result_tensor = Tensor(result_data, device_type=a.device, requires_grad=a.requires_grad or b.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Mul(_Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data, b.data)
        return a.data * b.data

    def backward(self, grad_output):
        a_data, b_data = self.saved_tensors
        grad_a = grad_output * b_data
        grad_b = grad_output * a_data
        return (grad_a, grad_b)

    @staticmethod
    def apply(a, b):
        ctx = Mul(a, b)
        result_data = Mul.forward(ctx, a, b)
        result_tensor = Tensor(result_data, device_type=a.device, requires_grad=a.requires_grad or b.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Div(_Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data, b.data)
        return a.data / b.data

    def backward(self, grad_output):
        a_data, b_data = self.saved_tensors
        grad_a = grad_output / b_data
        grad_b = grad_output * (-a_data / (b_data ** 2))
        return (grad_a, grad_b)

    @staticmethod
    def apply(a, b):
        ctx = Div(a, b)
        result_data = Div.forward(ctx, a, b)
        result_tensor = Tensor(result_data, device_type=a.device, requires_grad=a.requires_grad or b.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Pow(_Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.data, b) # Save base tensor data and scalar exponent
        return device.backend.power(a.data, b)

    def backward(self, grad_output):
        a_data, b = self.saved_tensors
        # Derivative of a^b with respect to a is b * a^(b-1)
        grad_a = grad_output * b * device.backend.power(a_data, b - 1)
        # Derivative of a^b with respect to b is a^b * log(a)
        # This is only needed if b was a tensor and required grad, but here b is a scalar
        return (grad_a,)

    @staticmethod
    def apply(a, b):
        # If b is a Tensor, it should be handled as a parent. If it's a scalar, it's saved.
        if isinstance(b, Tensor):
            ctx = Pow(a, b)
            result_data = Pow.forward(ctx, a, b.data)
            result_tensor = Tensor(result_data, device_type=a.device, requires_grad=a.requires_grad or b.requires_grad)
        else:
            ctx = Pow(a) # Only 'a' is a parent tensor
            result_data = Pow.forward(ctx, a, b)
            result_tensor = Tensor(result_data, device_type=a.device, requires_grad=a.requires_grad)

        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Exp(_Function):
    @staticmethod
    def forward(ctx, x):
        result = device.backend.exp(x.data)
        ctx.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result = self.saved_tensors[0]
        grad_x = grad_output * result
        return (grad_x,)

    @staticmethod
    def apply(x):
        ctx = Exp(x)
        result_data = Exp.forward(ctx, x)
        result_tensor = Tensor(result_data, device_type=x.device, requires_grad=x.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Log(_Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x.data)
        return device.backend.log(x.data)

    def backward(self, grad_output):
        x_data = self.saved_tensors[0]
        grad_x = grad_output / x_data
        return (grad_x,)

    @staticmethod
    def apply(x):
        ctx = Log(x)
        result_data = Log.forward(ctx, x)
        result_tensor = Tensor(result_data, device_type=x.device, requires_grad=x.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Sum(_Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.save_for_backward(x.data.shape, axis, keepdims)
        return device.backend.sum(x.data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        x_shape, axis, keepdims = self.saved_tensors
        if axis is None:
            grad_x = device.backend.full(x_shape, grad_output)
        else:
            if not keepdims:
                grad_output = device.backend.expand_dims(grad_output, axis=axis)
            grad_x = device.backend.broadcast_to(grad_output, x_shape)
        return (grad_x,)

    @staticmethod
    def apply(x, axis=None, keepdims=False):
        ctx = Sum(x) # Only 'x' is a parent tensor
        result_data = Sum.forward(ctx, x, axis, keepdims)
        result_tensor = Tensor(result_data, device_type=x.device, requires_grad=x.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Reshape(_Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.data.shape)
        return device.backend.reshape(x.data, shape)

    def backward(self, grad_output):
        original_shape = self.saved_tensors[0]
        grad_x = device.backend.reshape(grad_output, original_shape)
        return (grad_x,)

    @staticmethod
    def apply(x, shape):
        ctx = Reshape(x) # Only 'x' is a parent tensor
        result_data = Reshape.forward(ctx, x, shape)
        result_tensor = Tensor(result_data, device_type=x.device, requires_grad=x.requires_grad)
        if result_tensor.requires_grad:
            result_tensor._ctx = ctx
        return result_tensor

class Transpose(_Function):
    @staticmethod
    def forward(ctx, x, axes=None):
        ctx.save_for_backward(x.data.shape, axes)
        transposed_data = device.backend.transpose(x.data, axes=axes)
        return transposed_data

    def backward(self, grad):
        original_shape, axes = self.saved_tensors
        if axes is None:
            grad_x = device.backend.transpose(grad)
        else:
            # To reverse the transpose, we apply transpose with the inverse of the original axes
            inverse_axes = tuple(device.backend.argsort(axes))
            grad_x = device.backend.transpose(grad, axes=inverse_axes)
        return Tensor(grad_x)

    @staticmethod
    def apply(x, axes=None):
        ctx = Transpose(x)
        ctx.needs_input_grad = (x.requires_grad,)
        ctx.save_for_backward(x.shape, axes)
        return Tensor(device.backend.transpose(x.data, axes=axes), requires_grad=x.requires_grad, _ctx=ctx)