import pytest
import numpy as np
from modules.framework.tensor import Tensor
from modules.framework.nn import Linear
from tests.test_framework_ops import finite_difference_check

def test_linear_forward():
    in_features = 10
    out_features = 5
    batch_size = 3

    linear_layer = Linear(in_features, out_features)
    input_tensor = Tensor(np.random.randn(batch_size, in_features))

    output_tensor = linear_layer(input_tensor)

    assert output_tensor.shape == (batch_size, out_features)
    assert isinstance(output_tensor, Tensor)

def test_linear_backward():
    in_features = 4
    out_features = 3
    batch_size = 2

    linear_layer = Linear(in_features, out_features)
    input_tensor = Tensor(np.random.randn(batch_size, in_features), requires_grad=True)

    # Test weight gradients
    def f_weight(w):
        linear_layer.weight = w
        return linear_layer(input_tensor)

    numeric_grads_weight = finite_difference_check(f_weight, linear_layer.weight)
    linear_layer.zero_grad() # Reset gradients before analytical weight gradient calculation
    output = linear_layer(input_tensor)
    output.backward(Tensor(np.ones_like(output.data)))

    assert np.allclose(linear_layer.weight.grad.data, numeric_grads_weight[0].data, atol=1e-3)

    # Test bias gradients (if bias exists)
    if linear_layer.bias is not None:
        def f_bias(b):
            linear_layer.bias = b
            return linear_layer(input_tensor)

        numeric_grads_bias = finite_difference_check(f_bias, linear_layer.bias)
        linear_layer.zero_grad() # Reset gradients before analytical bias gradient calculation
        output = linear_layer(input_tensor)
        output.backward(Tensor(np.ones_like(output.data)))

        assert np.allclose(linear_layer.bias.grad.data, numeric_grads_bias[0].data, atol=1e-3)

    # Test input gradients
    def f_input(x):
        return linear_layer(x)

    numeric_grads_input = finite_difference_check(f_input, input_tensor)
    linear_layer.zero_grad() # Reset gradients before analytical input gradient calculation
    output = linear_layer(input_tensor)
    output.backward(Tensor(np.ones_like(output.data)))

    assert np.allclose(input_tensor.grad.data, numeric_grads_input[0].data, atol=1e-3)