from torch import nn, empty, Tensor, matmul
from math import sqrt

class MultiHeadLinear(nn.Module):
    """
    Applies a batched linear transformation to incoming data. For each batch, compute `y = xA^T + b`.
    
    Expects an input of size (batch_size, num_heads, seq_len, in_features).
    Returns an output of size (batch_size, num_heads, seq_len, out_features).
    """
    def __init__(self, num_heads: int, in_features: int, out_features: int, 
                 bias: bool=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(empty((num_heads, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(empty((num_heads, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        out = matmul(input, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out