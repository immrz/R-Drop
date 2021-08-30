import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torch import Tensor
from models.stoch_depth import StochDepthConsistencyBase

from typing import Any, Tuple


class EffNetV2(StochDepthConsistencyBase):
    def __init__(
        self,
        variant: str,
        pretrained: bool = False,
        num_classes: int = 1000,
        dropout_prob: float = 0.1,
        survival_prob: float = 0.8,
        **kwargs: Any,
    ) -> None:

        super().__init__(prob_start=1, prob_end=survival_prob, **kwargs)

        self.dropout_prob = dropout_prob
        self.survival_prob = survival_prob
        self.net = timm.create_model(variant,
                                     pretrained=pretrained,
                                     num_classes=num_classes,
                                     drop_rate=self.dropout_prob,
                                     drop_path_rate=1 - self.survival_prob)

        self.embedding = None
        self.net.global_pool.register_forward_hook(self.get_emb_hook())

    def get_custom_transform(self, is_training=True):
        data_config = resolve_data_config({}, model=self.net, use_test_size=not is_training)
        transform = create_transform(is_training=is_training, **data_config)
        return transform

    def get_emb_hook(self):
        def hook(module, input, output):
            self.embedding = output
        return hook

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.net(x)
        return logits, self.embedding


def efficientnetv2_m(**kwargs):
    return EffNetV2("tf_efficientnetv2_m", **kwargs)
