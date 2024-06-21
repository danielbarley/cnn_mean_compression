import math
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Parameter, Module
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
from torch.nn.functional import grad as G


class Conv2dFunction(Function):
    def __init__(self):
        super(Conv2dFunction, self).__init__()

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: int = 1,
        padding: str | int | Tuple[int, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        block_size: Tuple[int, int] = (1, 1),
        pooling: str = "avg",
    ) -> Tensor:
        with torch.no_grad():
            if pooling == "avg":
                sparse = torch.nn.functional.avg_pool2d(
                    input=input, kernel_size=block_size
                )
            elif pooling == "max":
                sparse = torch.nn.functional.max_pool2d(
                    input=input, kernel_size=block_size
                )
            else:
                raise ValueError("illegal argument to pooling")
            output = F.conv2d(
                input=input,
                weight=weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        ctx.save_for_backward(sparse, weight, bias)
        ctx.shape = input.shape
        ctx.block_size = block_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        del input
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor):
        input, weight, bias = ctx.saved_tensors
        kernel_size = ctx.block_size
        input = input.repeat_interleave(
            kernel_size[0], dim=2
        ).repeat_interleave(kernel_size[1], dim=3)
        # Pad input to correct shape if original was not divisible evenly
        if input.shape != ctx.shape:
            a = abs(input.shape[2] - ctx.shape[2])
            b = abs(input.shape[3] - ctx.shape[3])
            pad = torch.nn.ConstantPad2d(padding=(b, 0, a, 0), value=0)
            input = pad(input)
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = G.conv2d_input(
                input_size=input.shape,
                weight=weight,
                grad_output=grad_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = G.conv2d_weight(
                input=input,
                weight_size=weight.shape,
                grad_output=grad_out,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if ctx.needs_input_grad[2]:
            grad_bias = grad_out.sum((0, 2, 3)).squeeze(0)

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Conv2d(Module):
    __constants__ = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: int
    padding: str | Tuple[int, int]
    dilation: int
    groups: int
    block_size: Tuple[int, int]
    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: str | int | Tuple[int, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        block_size: int | Tuple[int, int] = 1,
        pooling: str = "avg",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(Conv2d, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError(
                "Expected block_size to be int or Tuple[int, int], "
                f"got {type(block_size).__name__}"
            )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if isinstance(block_size, int):
            self.block_size = (block_size, block_size)
        elif isinstance(block_size, tuple):
            self.block_size = block_size
        else:
            raise TypeError(
                "Expected block_size to be int or Tuple[int, int], "
                f"got {type(block_size).__name__}"
            )
        self.weight = Parameter(
            torch.empty(
                (
                    self.out_channels,
                    self.in_channels // self.groups,
                    *self.kernel_size,
                ),
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = Parameter(
                torch.zeros(self.out_channels, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # here be scary ghosts...
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight
            )
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return Conv2dFunction.apply(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.block_size,
        )

    def extra_repr(self) -> str:
        repr = """SpConv2D:
            in_channels={in_channels}, out_channels={out_channels}
            stride={stride}, padding={padding}, dilation={dilation}, groups={groups}
            block_size={block_size}""".format(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            block_size=self.block_size,
        )
        return repr
