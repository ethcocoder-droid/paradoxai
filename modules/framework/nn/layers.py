import math
from ..module import Module, Parameter
from ..tensor import Tensor
from ..ops.math_ops import MatMul, Add

class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.uniform(in_features, out_features, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features)))
        if bias:
            self.bias = Parameter(Tensor.uniform(out_features, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features)))
        else:
            self.bias = None

    def forward(self, input):
        output = MatMul.apply(input, self.weight)
        if self.bias is not None:
            output = Add.apply(output, self.bias)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )