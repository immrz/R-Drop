from .rdrop import ConsistencyWrapper
from .two_aug import bidirectional_kl_divergence
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from typing import Tuple, Union


class SemiSupvWrapper(ConsistencyWrapper):
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.0,
        beta: float = 0.5,
        rdrop: bool = False,
    ):
        super().__init__(model=model, alpha=alpha)
        self.beta = beta
        self.rdrop = rdrop
        self.ce = nn.CrossEntropyLoss()

    def forward_with_rdrop(self, x: Tuple[Tensor, Tuple[Tensor, Tensor]], labels: Tensor = None):
        supv_x, (unsupv_x1, unsupv_x2) = x
        supv_bs, unsupv_bs = supv_x.shape[0], unsupv_x1.shape[0]

        # stack all the inputs
        x = torch.cat([supv_x, supv_x.clone(),
                       unsupv_x1, unsupv_x1.clone(),
                       unsupv_x2, unsupv_x2.clone()], dim=0)
        logits, _ = self.model(x)

        # split the logits
        l = torch.split(logits, (supv_bs,) * 2 + (unsupv_bs,) * 4, dim=0)
        cls_loss = 0.5 * (self.ce(l[0], labels) + self.ce(l[1], labels))

        # in sample consistency loss (rdrop loss)
        in_sample_csst_loss = (bidirectional_kl_divergence(l[0], l[1])
                               + bidirectional_kl_divergence(l[2], l[3])
                               + bidirectional_kl_divergence(l[4], l[5]))

        # cross sample consistency loss (uda loss)
        p1, p2, p3, p4 = (F.softmax(_logits, dim=1) for _logits in l[2:])
        p1, p2 = 0.5 * (p1 + p2), 0.5 * (p3 + p4)
        logp1, logp2 = p1.log(), p2.log()
        cross_sample_csst_loss = 0.5 * (F.kl_div(logp1, p2, reduction="batchmean")
                                        + F.kl_div(logp2, p1, reduction="batchmean"))

        csst_loss = 0.5 * (self.beta * in_sample_csst_loss + cross_sample_csst_loss)
        loss = cls_loss + self.alpha * csst_loss
        return {"cls": cls_loss.item(),
                "in_sample_csst": in_sample_csst_loss.item(),
                "cross_sample_csst": cross_sample_csst_loss.item(),
                "csst": csst_loss.item(),
                "agg": loss}

    def forward_without_rdrop(self, x: Tuple[Tensor, Tuple[Tensor, Tensor]], labels: Tensor = None):
        supv_x, (unsupv_x1, unsupv_x2) = x
        supv_bs, unsupv_bs = supv_x.shape[0], unsupv_x1.shape[0]

        # stack all the inputs
        x = torch.cat([supv_x,
                       unsupv_x1,
                       unsupv_x2], dim=0)
        logits, _ = self.model(x)

        # split the logits
        l = torch.split(logits, (supv_bs, unsupv_bs, unsupv_bs), dim=0)
        cls_loss = self.ce(l[0], labels)

        # uda loss
        csst_loss = bidirectional_kl_divergence(l[1], l[2])

        loss = cls_loss + self.alpha * csst_loss
        return {"cls": cls_loss.item(),
                "csst": csst_loss.item(),
                "agg": loss}

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]], labels: Tensor = None):
        if self.training:
            if self.rdrop:
                return self.forward_with_rdrop(x, labels=labels)
            else:
                return self.forward_without_rdrop(x, labels=labels)
        else:
            return self.model(x, labels=labels)

    def __str__(self):
        return f"{self.__class__.__name__}(use_rdrop={self.rdrop}, alpha={self.alpha}, beta={self.beta})"
