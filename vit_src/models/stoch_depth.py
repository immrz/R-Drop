import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from typing import Type, Any, Callable, Union, List, Tuple, Optional
import logging


__all__ = ['wide_resnet101_2', 'resnet152', 'resnet50']

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


logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class StoDepth_BasicBlock(nn.Module):
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
        prob: float = 0.5,
        mult_flag: bool = False,
    ) -> None:
        super(StoDepth_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.prob = prob
        self.coin = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.mult_flag = mult_flag

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        if self.training:
            if torch.equal(self.coin.sample(), torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                out += identity

            else:  # bypass this block
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                out = identity

        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.mult_flag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)
        return out


class StoDepth_Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        prob: float = 0.5,
        mult_flag: bool = False,
    ) -> None:
        super(StoDepth_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        self.prob = prob
        self.coin = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.mult_flag = mult_flag

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        if self.training:
            if torch.equal(self.coin.sample(), torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                self.conv3.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                out += identity

            else:  # bypass this block
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                self.conv3.weight.requires_grad = False

                out = identity

        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.mult_flag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)
        return out


class StoDepth_ResNet_lineardecay(nn.Module):

    def __init__(
        self,
        block: Type[Union[StoDepth_BasicBlock, StoDepth_Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        probs: Tuple[float, float] = (1, 0.5),
        mult_flag: bool = False,
        consistency: str = None,
        alpha: float = 1.0,
        stop_grad: bool = False,
    ) -> None:

        super(StoDepth_ResNet_lineardecay, self).__init__()
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.consistency = consistency
        self.alpha = alpha
        self.stop_grad = stop_grad
        if self.consistency is not None:
            logger.info(f"Consistency loss is imposed on {consistency} with alpha={alpha}, stop_grad={stop_grad}")

        self.mult_flag = mult_flag
        self.prob_now = probs[0]
        self.prob_delta = probs[0] - probs[1]
        self.prob_step = self.prob_delta / (sum(layers) - 1)
        logger.info(f"prob_keep ranges from {probs[0]} to {probs[1]} with delta={self.prob_step}")

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoDepth_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, StoDepth_BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[StoDepth_BasicBlock, StoDepth_Bottleneck]], planes: int, blocks: int,
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
                            prob=self.prob_now, mult_flag=self.mult_flag))
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, prob=self.prob_now, mult_flag=self.mult_flag))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
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

        return x, z

    def compute_bi_kld(self, logits1, logits2):
        """Always assume logits1 has smaller loss than logits2.
        """
        print(self.ce(logits1, self.labels).item(), ', ', self.ce(logits2, self.labels).item())

        logp1, p1 = F.log_softmax(logits1, dim=1), F.softmax(logits1, dim=1)
        logp2, p2 = F.log_softmax(logits2, dim=1), F.softmax(logits2, dim=1)

        # use the good submodel to teach the bad submodel
        if self.stop_grad:
            logp1, p1 = logp1.detach(), p1.detach()

        kld = F.kl_div(logp1, p2, reduction='batchmean')
        kld_reverse = F.kl_div(logp2, p1, reduction='batchmean')
        return kld + kld_reverse

    def forward(self, x: Tensor, labels: Tensor = None) -> Tensor:
        logits, hidden = self._forward_impl(x)
        if labels is not None:
            loss = self.ce(logits, labels)

            self.labels = labels

            # forward twice
            if self.consistency is not None:
                logits2, hidden2 = self._forward_impl(x)
                loss2 = self.ce(logits2, labels)
                loss1, loss = loss, 0.5 * (loss + loss2)

                if self.consistency == 'logit':
                    loss -= self.alpha * F.cosine_similarity(logits, logits2, dim=1).mean()
                elif self.consistency == 'hidden':
                    loss -= self.alpha * F.cosine_similarity(hidden, hidden2, dim=1).mean()
                else:  # prob
                    if loss1 < loss2:
                        cons_loss = self.compute_bi_kld(logits, logits2)
                    else:
                        cons_loss = self.compute_bi_kld(logits2, logits)
                    loss += self.alpha * cons_loss

            return loss
        else:
            return logits, None


def _resnet(
    block: Type[Union[StoDepth_BasicBlock, StoDepth_Bottleneck]],
    layers: List[int],
    pretrained_url: str = None,
    num_classes: int = 1000,
    replace_fc: bool = False,
    **kwargs,
) -> StoDepth_ResNet_lineardecay:

    model = StoDepth_ResNet_lineardecay(block, layers, **kwargs)
    if pretrained_url is not None:
        state_dict = model_zoo.load_url(pretrained_url)
        model.load_state_dict(state_dict)

    if num_classes != 1000 or replace_fc:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def wide_resnet101_2(
    pretrained: bool = False,
    **kwargs: Any,
) -> StoDepth_ResNet_lineardecay:
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
    model = _resnet(StoDepth_Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(
    pretrained: bool = True,
    **kwargs: Any,
) -> StoDepth_ResNet_lineardecay:

    kwargs['pretrained_url'] = model_urls['resnet152'] if pretrained else None
    model = _resnet(StoDepth_Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet50(
    pretrained: bool = False,
    **kwargs: Any,
) -> StoDepth_ResNet_lineardecay:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    kwargs['pretrained_url'] = model_urls['resnet50'] if pretrained else None
    model = _resnet(StoDepth_Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
