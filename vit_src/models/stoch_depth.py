import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
import logging


__all__ = ["StochDepthBasicBlock", "StochDepthBottleneck", "StochDepthConsistencyBase", "conv1x1"]


logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def drop_path(x, keep_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if keep_prob >= 1. or not training:
        return x
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    """
    def __init__(self, keep_prob=1.):
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, x):
        return drop_path(x, self.keep_prob, self.training)

    def __repr__(self):
        return f"DropPath(keep_prob={self.keep_prob})"


class StochDepthBasicBlock(nn.Module):
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
        keep_prob: float = 0.5,
    ) -> None:
        super(StochDepthBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.keep_prob = keep_prob
        self.drop_path = DropPath(keep_prob=keep_prob) if keep_prob < 1. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + self.drop_path(out)
        out = self.relu(out)

        return out


class StochDepthBottleneck(nn.Module):
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
        keep_prob: float = 0.5,
    ) -> None:
        super(StochDepthBottleneck, self).__init__()
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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.keep_prob = keep_prob
        self.drop_path = DropPath(keep_prob=keep_prob) if keep_prob < 1. else nn.Identity()

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

        out = identity + self.drop_path(out)
        out = self.relu(out)

        return out


class StochDepthConsistencyBase(nn.Module):
    """The base class for models with stochastic depth and R-Drop consistency loss.
    Child classes should implement the method '_forward_impl(self, x)'
    """
    def __init__(
        self,
        prob_start: float = 1.,
        prob_step: float = 0.,
        consistency: str = None,
        consist_func: str = None,
        alpha: float = 1.0,
        stop_grad: bool = False,
    ) -> None:

        super().__init__()
        self.prob_now = prob_start
        self.prob_step = prob_step
        self.consistency = consistency
        self.consist_func = consist_func
        self.alpha = alpha
        self.stop_grad = stop_grad
        self.ce = nn.CrossEntropyLoss()

        logger.info(f"prob_keep starts from {prob_start} with delta={prob_step}")

        if self.consistency is not None:
            assert self.consist_func is not None
            logger.info(f"Consistency loss is imposed on {consistency} with alpha={alpha}, "
                        f"stop_grad={stop_grad}, function={consist_func}.")
        else:
            logger.info("No consistency loss.")

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward input x and return both output and hidden emb.
        """
        raise NotImplementedError

    def compute_consistency_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute consistency loss between two outputs.
        Always assume z1 has smaller loss than z2.
        """

        # whether to use the good submodel to teach the bad submodel
        if self.stop_grad:
            z1 = z1.detach()

        if self.consistency in ["logit", "hidden"]:
            if self.consist_func == "cosine":
                loss = 1. - F.cosine_similarity(z1, z2, dim=1).mean()
            elif self.consist_func == "l2":
                loss = 0.5 * torch.sum((z1 - z2) ** 2) / z1.size(0)
            else:
                raise NotImplementedError

        elif self.consistency == "prob":
            # NOTE: KL(p||q) = kl_div(q.log(), p)
            logp1, p1 = F.log_softmax(z1, dim=1), F.softmax(z1, dim=1)
            logp2, p2 = F.log_softmax(z2, dim=1), F.softmax(z2, dim=1)

            if self.consist_func == "kl":
                kld = F.kl_div(logp1, p2, reduction="batchmean")
                kld_reverse = F.kl_div(logp2, p1, reduction="batchmean")
                loss = kld + kld_reverse
            elif self.consist_func == "js":
                mean = torch.clamp(0.5 * (p1 + p2), min=1e-7, max=1.)
                log_mean = mean.log()
                loss = F.kl_div(log_mean, p1, reduction="batchmean") + F.kl_div(log_mean, p2, reduction="batchmean")
            elif self.consist_func == "ce":
                ce = -(p1 * logp2).sum()
                ce_reverse = -(p2 * logp1).sum()
                loss = (ce + ce_reverse) / z1.size(0)
            else:
                raise NotImplementedError

            loss = loss * 0.5

        else:
            raise NotImplementedError

        return loss

    def forward(self, x: Tensor, labels: Tensor = None):
        logits, hidden = self._forward_impl(x)

        if labels is not None:
            loss = self.ce(logits, labels)

            if self.consistency is None:
                return loss

            else:
                # forward twice to compute consistency loss
                logits2, hidden2 = self._forward_impl(x)
                loss2 = self.ce(logits2, labels)

                if self.consistency == "hidden":
                    z1, z2 = hidden, hidden2
                else:
                    z1, z2 = logits, logits2

                if loss < loss2:
                    consist_loss = self.compute_consistency_loss(z1, z2)
                else:
                    consist_loss = self.compute_consistency_loss(z2, z1)

                cls_loss = 0.5 * (loss + loss2)
                agg_loss = cls_loss + self.alpha * consist_loss

                return {"cls": cls_loss.item(), "csst": consist_loss.item(), "agg": agg_loss}

        else:
            return logits, None
