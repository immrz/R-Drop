from .rdrop import ConsistencyWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union


def bidirectional_kl_divergence(z1, z2):
    """Compute bidirectional kl divergence between two logits `z1` and `z2`.
    """
    logp1, p1 = F.log_softmax(z1, dim=1), F.softmax(z1, dim=1)
    logp2, p2 = F.log_softmax(z2, dim=1), F.softmax(z2, dim=1)
    kld = F.kl_div(logp1, p2, reduction="batchmean")
    kld_reverse = F.kl_div(logp2, p1, reduction="batchmean")
    loss = kld + kld_reverse
    return 0.5 * loss


class TwoAugWrapper(ConsistencyWrapper):
    def __init__(
        self,
        model: nn.Module,
        consistency: bool = False,
        alpha: float = 1.0,
    ):
        super().__init__(model=model, alpha=alpha)
        self.consistency = consistency
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], labels: Tensor = None):
        """If training, `labels` will be given, and `x` will be a list of two tensors.
        Return classification loss, and consistency loss if required.
        If testing, return the outputs of `model` instead.
        """
        if self.training:
            bs = x[0].shape[0]  # original batch size
            x, labels = torch.cat(x, dim=0), torch.cat([labels, labels.clone()], dim=0)
            logits, _ = self.model(x)
            cls_loss = self.ce(logits, labels)

            # if consistency loss is used
            if self.consistency:
                logits1, logits2 = torch.split(logits, bs, dim=0)
                csst_loss = bidirectional_kl_divergence(logits1, logits2)
                loss = cls_loss + self.alpha * csst_loss
                return {"cls": cls_loss.item(), "csst": csst_loss.item(), "agg": loss}
            else:
                return cls_loss
        else:
            return self.model(x, labels=labels)

    def __str__(self):
        return f"{self.__class__.__name__}(consistency={self.consistency}, alpha={self.alpha})"
