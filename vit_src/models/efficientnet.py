import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torch import Tensor
import torch.nn as nn
from typing import Union, Tuple


class EffNet(nn.Module):
    def __init__(
        self,
        variant: str,
        pretrained: bool = False,
        num_classes: int = 1000,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.2,
    ) -> None:

        super().__init__()

        self.net = timm.create_model(variant,
                                     pretrained=pretrained,
                                     num_classes=num_classes,
                                     drop_rate=drop_rate,
                                     drop_path_rate=drop_path_rate)
        self.ce = nn.CrossEntropyLoss()

        self.embedding = None
        self.net.global_pool.register_forward_hook(self.get_emb_hook())

    def get_custom_transform(self):
        transforms = []
        for is_training in [True, False]:
            data_config = resolve_data_config({}, model=self.net, use_test_size=not is_training)
            transform = create_transform(is_training=is_training, **data_config)
            transforms.append(transform)
        return transforms

    def get_emb_hook(self):
        def hook(module, input, output):
            self.embedding = output
        return hook

    def forward(self, x: Tensor, labels: Tensor = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        logits = self.net(x)
        if labels is not None:
            loss = self.ce(logits, labels)
            return loss
        else:
            return logits, self.embedding


def efficientnetv2_m(**kwargs):
    return EffNet("tf_efficientnetv2_m", **kwargs)


def efficientnet_b0(**kwargs):
    return EffNet("efficientnet_b0", **kwargs)
