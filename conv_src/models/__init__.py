from .resnet import ResNet, ThreeLayerResNet
from .stoch_depth import StochDepthBasicBlock, StochDepthBottleneck

import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from typing import Type, Any, Union, List
import logging


__all__ = ["resnet50", "resnet152", "resnet110", "wide_resnet101_2"]


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
