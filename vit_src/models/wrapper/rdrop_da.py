from .rdrop import ConsistencyWrapper
from .two_aug import bidirectional_kl_divergence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple


class RDropDAWrapper(ConsistencyWrapper):
    def __init__(
        self,
        model: nn.Module,
        consistency: str,
        alpha: float = 1.0,
    ):
        super().__init__(model=model, alpha=alpha)
        self.consistency = consistency
        self.ce = nn.CrossEntropyLoss()
        assert self.consistency in ["prob", "logit"]

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], labels: Tensor = None):
        """If training, `labels` will be given, and `x` will be a list of two tensors.
        Each x_i will be forwarded twice. Return losses.
        If testing, return the outputs of `model` instead.
        """
        if self.training:
            # double each input and forward, compute classification loss
            bs = x[0].shape[0]  # original batch size
            x1, x2 = x
            x = torch.cat([x1, x1.clone(), x2, x2.clone()], dim=0)
            labels = torch.cat([labels, labels.clone(), labels.clone(), labels.clone()], dim=0)
            logits, _ = self.model(x)
            cls_loss = self.ce(logits, labels)

            # compute consistency loss among the four outputs
            logits1, logits2, logits3, logits4 = torch.split(logits, bs, dim=0)
            in_sample_csst_loss = (bidirectional_kl_divergence(logits1, logits2)
                                   + bidirectional_kl_divergence(logits3, logits4))
            if self.consistency == "logit":
                # mixup the logits
                cross_sample_csst_loss = bidirectional_kl_divergence(logits1 + logits2, logits3 + logits4)
            else:
                # average the probabilities
                p1, p2, p3, p4 = (F.softmax(_logits, dim=1) for _logits in [logits1, logits2, logits3, logits4])
                p1, p2 = 0.5 * (p1 + p2), 0.5 * (p3 + p4)
                logp1, logp2 = p1.log(), p2.log()
                cross_sample_csst_loss = 0.5 * (F.kl_div(logp1, p2, reduction="batchmean")
                                                + F.kl_div(logp2, p1, reduction="batchmean"))
            csst_loss = 0.25 * in_sample_csst_loss + 0.5 * cross_sample_csst_loss

            loss = cls_loss + self.alpha * csst_loss
            return {"cls": cls_loss.item(),
                    "in_sample_csst": in_sample_csst_loss.item(),
                    "cross_sample_csst": cross_sample_csst_loss.item(),
                    "csst": csst_loss.item(),
                    "agg": loss}

        else:
            return self.model(x, labels=labels)

    def __str__(self):
        return f"{self.__class__.__name__}(consistency={self.consistency}, alpha={self.alpha})"
