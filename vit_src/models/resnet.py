from .stoch_depth import StochDepthBasicBlock, StochDepthBottleneck, conv1x1
from typing import Type, Any, Callable, Union, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch import Tensor

import logging


logger = logging.getLogger(__name__)
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def init_weight(net: nn.Module, zero_init_residual: bool = False):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
        for m in net.modules():
            if isinstance(m, StochDepthBottleneck):
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
            elif isinstance(m, StochDepthBasicBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


class ResNet(nn.Module):
    """The commonly used ResNet architecture, but with stochastic depth and consistency loss.
    """
    def __init__(
        self,
        block: Type[Union[StochDepthBasicBlock, StochDepthBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        probs: Tuple[float, float] = (1, 0.5),
    ) -> None:

        super().__init__()
        self.prob_now = probs[0]
        self.prob_step = (probs[0] - probs[1]) / (sum(layers) - 1)
        logger.info(f"prob_keep starts from {probs[0]} to {probs[1]}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.ce = nn.CrossEntropyLoss()

        # initialize weight
        init_weight(self, zero_init_residual=zero_init_residual)

    def _make_layer(self, block: Type[Union[StochDepthBasicBlock, StochDepthBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            keep_prob=self.prob_now))
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, keep_prob=self.prob_now))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, labels: Tensor = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        x = self.fc(z)

        if labels is not None:
            loss = self.ce(x, labels)
            return loss
        else:
            return x, z


class ThreeLayerResNet(nn.Module):
    """The three-layer ResNet specialized for CIFAR10/100 dataset.
    """
    def __init__(
        self,
        block: Type[Union[StochDepthBasicBlock, StochDepthBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        probs: Tuple[float, float] = (1, 0.5),
    ) -> None:

        super().__init__()
        self.prob_now = probs[0]
        self.prob_step = (probs[0] - probs[1]) / (sum(layers) - 1)
        logger.info(f"prob_keep starts from {probs[0]} to {probs[1]}")
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.ce = nn.CrossEntropyLoss()

        # initialize weight
        init_weight(self, zero_init_residual=zero_init_residual)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, keep_prob=self.prob_now))
            self.prob_now = self.prob_now - self.prob_step
            self.inplanes = planes * block.expansion
            downsample = None  # only the first block needs downsampling

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, labels: Tensor = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        x = self.fc(z)

        if labels is not None:
            loss = self.ce(x, labels)
            return loss
        else:
            return x, z


def _resnet(
    block: Type[Union[StochDepthBasicBlock, StochDepthBottleneck]],
    layers: List[int],
    pretrained_url: str = None,
    num_classes: int = 1000,
    replace_fc: bool = False,
    **kwargs,
) -> Union[ResNet, ThreeLayerResNet]:

    if len(layers) == 3:
        model = ThreeLayerResNet(block, layers, **kwargs)
    else:
        model = ResNet(block, layers, **kwargs)
    if pretrained_url is not None:
        logger.warning("Pretrained model will be downloaded and used!")
        state_dict = model_zoo.load_url(pretrained_url)
        model.load_state_dict(state_dict)

    if num_classes != 1000 or replace_fc:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def wide_resnet101_2(
    pretrained: bool = False,
    **kwargs: Any,
) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    kwargs['width_per_group'] = 64 * 2
    kwargs['pretrained_url'] = model_urls['wide_resnet101_2'] if pretrained else None
    model = _resnet(StochDepthBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(
    pretrained: bool = True,
    **kwargs: Any,
) -> ResNet:

    kwargs['pretrained_url'] = model_urls['resnet152'] if pretrained else None
    model = _resnet(StochDepthBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet50(
    pretrained: bool = False,
    **kwargs: Any,
) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    kwargs['pretrained_url'] = model_urls['resnet50'] if pretrained else None
    model = _resnet(StochDepthBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet110(
    pretrained: bool = False,
    **kwargs: Any
) -> ThreeLayerResNet:
    """ResNet-110 model specially for Cifar10 and Cifar100 dataset. Trained from scratch.
    """
    model = _resnet(StochDepthBasicBlock, [18, 18, 18], **kwargs)
    return model
