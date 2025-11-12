import pytest
import numpy as np
from modules.framework.tensor import Tensor
from modules.framework.ops.math_ops import Add, MatMul, ReLU
from modules.framework.device import device

# Helper function for finite difference checking
def finite_difference_check(func, *inputs, epsilon=1e-4, tol=1e-4):
    numeric_grads = []
    for i, x in enumerate(inputs):
        if not x.requires_grad:
            numeric_grads.append(None)
            continue

        grad_x = device.backend.zeros_like(x.data)
        for idx, _ in np.ndenumerate(x.data):
            original_value = x.data[idx]

            x.data[idx] = original_value + epsilon
            output_plus = func(*inputs)
            if output_plus.data.size > 1:
                output_plus = Tensor(device.backend.sum(output_plus.data), device_type=output_plus.device)

            x.data[idx] = original_value - epsilon
            output_minus = func(*inputs)
            if output_minus.data.size > 1:
                output_minus = Tensor(device.backend.sum(output_minus.data), device_type=output_minus.device)

            x.data[idx] = original_value # Restore original value

            numeric_grad = (output_plus.data - output_minus.data) / (2 * epsilon)
            grad_x[idx] = numeric_grad
        numeric_grads.append(Tensor(grad_x, device_type=x.device))
    return numeric_grads

class TestMathOps:

    def test_add_forward(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        c = a + b
        assert np.allclose(c.data, np.array([4.0, 6.0]))

    def test_add_backward(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)

        def func(a_in, b_in):
            return a_in + b_in

        c = func(a, b)
        c.backward(Tensor(np.array([1.0, 1.0])))

        numeric_grads = finite_difference_check(func, a, b)


        assert np.allclose(a.grad.data, numeric_grads[0].data, atol=1e-4)
        assert np.allclose(b.grad.data, numeric_grads[1].data, atol=1e-4)

    def test_matmul_forward(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c.data, expected)

    def test_matmul_backward(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)

        def func(a_in, b_in):
            return a_in @ b_in

        c = func(a, b)
        c.backward(Tensor(np.array([[1.0, 1.0], [1.0, 1.0]])))

        # Test grad_a
        numeric_grad_a = finite_difference_check(lambda x: func(x, b), a)[0]
        assert np.allclose(a.grad.data, numeric_grad_a.data, atol=1e-3)

        # Test grad_b
        numeric_grad_b = finite_difference_check(lambda x: func(a, x), b)[0]
        assert np.allclose(b.grad.data, numeric_grad_b.data, atol=1e-3)

    def test_relu_forward(self):
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
        y = x.relu()
        assert np.allclose(y.data, np.array([0.0, 0.0, 1.0]))

    def test_relu_backward(self):
        x = Tensor(np.array([-1.0, 0.1, 1.0]), requires_grad=True)

        def func(x_in):
            return x_in.relu()

        y = func(x)
        y.backward(Tensor(np.array([1.0, 1.0, 1.0])))

        numeric_grads = finite_difference_check(func, x)

        assert np.allclose(x.grad.data, numeric_grads[0].data, atol=1e-4)

    def test_add_broadcast_backward(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array(3.0), requires_grad=True) # Scalar

        def func(a_in, b_in):
            return a_in + b_in

        c = func(a, b)
        c.backward(Tensor(np.array([1.0, 1.0])))

        numeric_grads = finite_difference_check(func, a, b)

        assert np.allclose(a.grad.data, numeric_grads[0].data, atol=1e-4)
        assert np.allclose(b.grad.data, numeric_grads[1].data, atol=1e-4)

    def test_add_broadcast_backward_2(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([10.0, 20.0]), requires_grad=True) # Row vector

        def func(a_in, b_in):
            return a_in + b_in

        c = func(a, b)
        c.backward(Tensor(np.array([[1.0, 1.0], [1.0, 1.0]])))

        numeric_grads = finite_difference_check(func, a, b)

        assert np.allclose(a.grad.data, numeric_grads[0].data, atol=1e-4)
        assert np.allclose(b.grad.data, numeric_grads[1].data, atol=1e-4)

    def test_add_broadcast_backward_3(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        b = Tensor(np.array([[10.0], [20.0]]), requires_grad=True) # Column vector

        def func(a_in, b_in):
            return a_in + b_in

        c = func(a, b)
        c.backward(Tensor(np.array([[1.0, 1.0], [1.0, 1.0]])))

        numeric_grads = finite_difference_check(func, a, b)

        assert np.allclose(a.grad.data, numeric_grads[0].data, atol=1e-4)
        assert np.allclose(b.grad.data, numeric_grads[1].data, atol=1e-4)

    def test_sub_forward(self):
        a = Tensor(np.array([1, 2, 3]))
        b = Tensor(np.array([3, 2, 1]))
        c = a - b
        np.testing.assert_array_equal(c.data, np.array([-2, 0, 2]))

    def test_sub_backward(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 2.0, 1.0]), requires_grad=True)
        c = a - b
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: (x - b).sum(), a)
        numeric_grads_b = finite_difference_check(lambda x: (a - x).sum(), b)

        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)
        np.testing.assert_allclose(b.grad.data, numeric_grads_b[0].data, atol=1e-4)

    def test_mul_forward(self):
        a = Tensor(np.array([1, 2, 3]))
        b = Tensor(np.array([3, 2, 1]))
        c = a * b
        np.testing.assert_array_equal(c.data, np.array([3, 4, 3]))

    def test_mul_backward(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 2.0, 1.0]), requires_grad=True)
        c = a * b
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: (x * b).sum(), a)
        numeric_grads_b = finite_difference_check(lambda x: (a * x).sum(), b)

        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)
        np.testing.assert_allclose(b.grad.data, numeric_grads_b[0].data, atol=1e-4)

    def test_div_forward(self):
        a = Tensor(np.array([6, 4, 2]))
        b = Tensor(np.array([3, 2, 1]))
        c = a / b
        np.testing.assert_array_equal(c.data, np.array([2, 2, 2]))

    def test_div_backward(self):
        a = Tensor(np.array([6.0, 4.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 2.0, 1.0]), requires_grad=True)
        c = a / b
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: (x / b).sum(), a)
        numeric_grads_b = finite_difference_check(lambda x: (a / x).sum(), b)

        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)
        np.testing.assert_allclose(b.grad.data, numeric_grads_b[0].data, atol=1e-4)

    def test_pow_forward(self):
        a = Tensor(np.array([2, 3, 4]))
        b = 2
        c = a ** b
        np.testing.assert_array_equal(c.data, np.array([4, 9, 16]))

    def test_pow_backward(self):
        a = Tensor(np.array([2.0, 3.0, 4.0]), requires_grad=True)
        b = 2.0
        c = a ** b
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: (x ** b).sum(), a)

        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)

    def test_exp_forward(self):
        a = Tensor(np.array([0, 1, 2]))
        c = a.exp()
        np.testing.assert_allclose(c.data, np.array([np.exp(0), np.exp(1), np.exp(2)]))

    def test_exp_backward(self):
        a = Tensor(np.array([0.1, 1.0, 2.0]), requires_grad=True)
        c = a.exp()
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: x.exp().sum(), a)

        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)

    def test_log_forward(self):
        a = Tensor(np.array([1, np.exp(1), np.exp(2)]))
        c = a.log()
        np.testing.assert_allclose(c.data, np.array([0, 1, 2]))

    def test_log_backward(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        c = a.log()
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: x.log().sum(), a)

        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)

    def test_sum_forward(self):
        a = Tensor(np.array([[1, 2], [3, 4]]))
        c = a.sum()
        np.testing.assert_array_equal(c.data, np.array(10))

        c_axis0 = a.sum(axis=0)
        np.testing.assert_array_equal(c_axis0.data, np.array([4, 6]))

        c_axis1 = a.sum(axis=1)
        np.testing.assert_array_equal(c_axis1.data, np.array([3, 7]))

        c_keepdims = a.sum(axis=0, keepdims=True)
        np.testing.assert_array_equal(c_keepdims.data, np.array([[4, 6]]))

    def test_sum_backward(self):
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        c = a.sum()
        c.backward()

        numeric_grads_a = finite_difference_check(lambda x: x.sum(), a)
        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)

        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        c_axis0 = a.sum(axis=0)
        c_axis0.sum().backward()

        numeric_grads_a_axis0 = finite_difference_check(lambda x: x.sum(axis=0).sum(), a)
        np.testing.assert_allclose(a.grad.data, numeric_grads_a_axis0[0].data, atol=1e-4)

        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        c_axis1 = a.sum(axis=1)
        c_axis1.sum().backward()

        numeric_grads_a_axis1 = finite_difference_check(lambda x: x.sum(axis=1).sum(), a)
        np.testing.assert_allclose(a.grad.data, numeric_grads_a_axis1[0].data, atol=1e-4)

        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        c_keepdims = a.sum(axis=0, keepdims=True)
        c_keepdims.sum().backward()

        numeric_grads_a_keepdims = finite_difference_check(lambda x: x.sum(axis=0, keepdims=True).sum(), a)
        np.testing.assert_allclose(a.grad.data, numeric_grads_a_keepdims[0].data, atol=1e-4)

    def test_reshape_forward(self):
        a = Tensor(np.array([1, 2, 3, 4, 5, 6]))
        c = a.reshape((2, 3))
        np.testing.assert_array_equal(c.data, np.array([[1, 2, 3], [4, 5, 6]]))

    def test_reshape_backward(self):
        a = Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), requires_grad=True)
        c = a.reshape((2, 3))
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: x.reshape((2, 3)).sum(), a)
        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)

    def test_transpose_forward(self):
        a = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        c = a.transpose()
        np.testing.assert_array_equal(c.data, np.array([[1, 4], [2, 5], [3, 6]]))

        a = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
        c_axes = a.transpose(axes=(0, 2, 1))
        np.testing.assert_array_equal(c_axes.data, np.array([[[1, 3], [2, 4]], [[5, 7], [6, 8]]]))

    def test_transpose_backward(self):
        a = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
        c = a.transpose()
        c.sum().backward()

        numeric_grads_a = finite_difference_check(lambda x: x.transpose().sum(), a)
        np.testing.assert_allclose(a.grad.data, numeric_grads_a[0].data, atol=1e-4)

        a = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]), requires_grad=True)
        c_axes = a.transpose(axes=(0, 2, 1))
        c_axes.sum().backward()

        numeric_grads_a_axes = finite_difference_check(lambda x: x.transpose(axes=(0, 2, 1)).sum(), a)
        np.testing.assert_allclose(a.grad.data, numeric_grads_a_axes[0].data, atol=1e-4)