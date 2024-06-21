from typing import Tuple, Optional, Callable, Type, List, Any
import os
import torch
from torch import nn
from torch import Tensor
from torchvision.models._api import register_model

import layers.convolution as comp_conv


def pack_hook(x: Tensor) -> Tensor:
    return x.to_sparse_bsr()


def unpack_hook(x: Tensor) -> Tensor:
    return x.to_dense()


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    block_size: int | Tuple[int, int] = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> comp_conv.Conv2d:
    return comp_conv.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    block_size: int | Tuple[int, int] = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> comp_conv.Conv2d:
    return comp_conv.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block_size: int | Tuple[int, int] = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            # Why give the option anyways?
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        self.conv1 = conv3x3(
            in_planes=inplanes,
            out_planes=planes,
            stride=stride,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        self.bn1 = norm_layer(
            num_features=planes,
            device=device,
            dtype=dtype,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            in_planes=planes,
            out_planes=planes,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        self.bn2 = norm_layer(
            num_features=planes,
            device=device,
            dtype=dtype,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block_size: int | Tuple[int, int] = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(
            in_planes=inplanes,
            out_planes=width,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        self.bn1 = norm_layer(
            width,
            device=device,
            dtype=dtype,
        )
        self.conv2 = conv3x3(
            in_planes=width,
            out_planes=width,
            stride=stride,
            groups=groups,
            dilation=dilation,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        self.bn2 = norm_layer(
            num_features=width,
            device=device,
            dtype=dtype,
        )
        self.conv3 = conv1x1(
            in_planes=width,
            out_planes=planes * self.expansion,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        self.bn3 = norm_layer(
            num_features=planes * self.expansion,
            device=device,
            dtype=dtype,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock | Bottleneck],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block_size: int | Tuple[int, int] = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None"
                f" or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        if isinstance(block_size, list):
            if len(block_size) != len(layers):
                raise ValueError(
                    "block_size list has to have the same dimensions as layers!,"
                    f" got block_size: {len(block_size)}, layers: {len(layers)}"
                )
            self.block_size = block_size
        elif isinstance(block_size, int):
            self.block_size = [block_size for _ in range(len(layers))]
        else:
            raise TypeError(
                "Expect block_size to be of type int or List[int]"
                f", got: {block_size.__class__.__name__}"
            )
        self.conv1 = comp_conv.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            block_size=self.block_size[0],
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block=block,
            planes=64,
            blocks=layers[0],
            block_size=self.block_size[0],
        )
        self.layer2 = self._make_layer(
            block=block,
            planes=128,
            blocks=layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            block_size=self.block_size[1],
        )
        self.layer3 = self._make_layer(
            block=block,
            planes=256,
            blocks=layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            block_size=self.block_size[2],
        )
        self.layer4 = self._make_layer(
            block=block,
            planes=512,
            blocks=layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            block_size=self.block_size[3],
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(
            in_features=512 * block.expansion, out_features=num_classes
        )

        for m in self.modules():
            if isinstance(m, comp_conv.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, torch.nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        block_size: int | Tuple[int, int] = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    in_planes=self.inplanes,
                    out_planes=planes * block.expansion,
                    stride=stride,
                    block_size=block_size,
                    device=device,
                    dtype=dtype,
                ),
                norm_layer(
                    num_features=planes * block.expansion,
                    device=device,
                    dtype=dtype,
                ),
            )
        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                block_size=block_size,
                device=device,
                dtype=dtype,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    block_size=block_size,
                    device=device,
                    dtype=dtype,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[BasicBlock | Bottleneck],
    layers: List[int],
    block_size: int | Tuple[int, int] = 1,
    load_pretrained: Optional[os.PathLike] = None,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(
        block=block,
        layers=layers,
        block_size=block_size,
        **kwargs,
    )
    if load_pretrained is not None:
        state = torch.load(load_pretrained)
        model.load_state_dict(state_dict=state.state_dict)

    return model


@register_model()
def compressed_resnet18(
    *,
    block_size: int | Tuple[int, int] = 1,
    load_pretrained: Optional[os.PathLike] = None,
    **kwargs: Any,
) -> ResNet:
    return _resnet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        block_size=block_size,
        load_pretrained=load_pretrained,
        **kwargs,
    )


@register_model()
def compressed_resnet34(
    *,
    block_size: int | Tuple[int, int] = 1,
    load_pretrained: Optional[os.PathLike] = None,
    **kwargs: Any,
) -> ResNet:
    return _resnet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        block_size=block_size,
        load_pretrained=load_pretrained,
        **kwargs,
    )


@register_model()
def compressed_resnet50(
    *,
    block_size: int | Tuple[int, int] = 1,
    load_pretrained: Optional[os.PathLike] = None,
    **kwargs: Any,
) -> ResNet:
    return _resnet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        block_size=block_size,
        load_pretrained=load_pretrained,
        **kwargs,
    )


@register_model()
def compressed_resnet101(
    *,
    block_size: int | Tuple[int, int] = 1,
    load_pretrained: Optional[os.PathLike] = None,
    **kwargs: Any,
) -> ResNet:
    return _resnet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        block_size=block_size,
        load_pretrained=load_pretrained,
        **kwargs,
    )


@register_model()
def compressed_resnet152(
    *,
    block_size: int | Tuple[int, int] = 1,
    load_pretrained: Optional[os.PathLike] = None,
    **kwargs: Any,
) -> ResNet:
    return _resnet(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        block_size=block_size,
        load_pretrained=load_pretrained,
        **kwargs,
    )
