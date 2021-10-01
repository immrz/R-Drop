from .rdrop import ConsistencyWrapper
from .two_aug import bidirectional_kl_divergence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple
import itertools


def compute_DM_consistency(z1, z2, is_logit=True, func="kl"):
    """Compute data or model consistency loss. `z1` and `z2` can be either logits and probabilities,
    indicated by `is_logit`. If `func` is kl, compute bi-kl loss; else, compute l2 loss.
    """
    if func == "kl":
        if is_logit:
            return bidirectional_kl_divergence(z1, z2)
        else:
            p1, p2 = z1, z2
            logp1, logp2 = p1.log(), p2.log()
            return 0.5 * (F.kl_div(logp1, p2, reduction="batchmean")
                          + F.kl_div(logp2, p1, reduction="batchmean"))
    elif func == "l2":
        assert is_logit
        return torch.sum((z1 - z2) ** 2) / z1.size(0)
    else:
        raise NotImplementedError


class RDropDAWrapper(ConsistencyWrapper):
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.0,
        beta: float = 0.5,
        model_cfunc: str = "kl",
        data_cfunc: str = "kl",
    ):
        super().__init__(model=model, alpha=alpha)
        self.beta = beta

        assert model_cfunc in ["kl", "l2"] and data_cfunc in ["kl", "l2"]
        self.model_cfunc = model_cfunc
        self.data_cfunc = data_cfunc

        self.ce = nn.CrossEntropyLoss()

    def _cat_and_split(self, x: Tuple[Tensor, Tensor], labels: Tensor):
        bs = x[0].shape[0]
        x1, x2 = x
        x = torch.cat([x1, x1.clone(), x2, x2.clone()], dim=0)
        labels = torch.cat([labels, labels.clone(), labels.clone(), labels.clone()], dim=0)
        logits, _ = self.model(x)
        cls_loss = self.ce(logits, labels)
        logits1, logits2, logits3, logits4 = torch.split(logits, bs, dim=0)
        return cls_loss, (logits1, logits2, logits3, logits4)

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], labels: Tensor = None):
        """If training, `labels` will be given, and `x` will be a list of two tensors.
        Each x_i will be forwarded twice. Return losses.
        If testing, return the outputs of `model` instead.
        """
        if self.training:
            # concat x, compute classification loss, and then split into logits
            cls_loss, (logits1, logits2, logits3, logits4) = self._cat_and_split(x, labels)

            # the rdrop loss -> model randomness
            in_sample_csst_loss = (compute_DM_consistency(logits1, logits2, is_logit=True, func=self.model_cfunc)
                                   + compute_DM_consistency(logits3, logits4, is_logit=True, func=self.model_cfunc))

            # the data consistency loss -> data randomness
            if self.data_cfunc == "kl":
                p1, p2, p3, p4 = (F.softmax(_logits, dim=1) for _logits in [logits1, logits2, logits3, logits4])
                p1, p2 = 0.5 * (p1 + p2), 0.5 * (p3 + p4)
                cross_sample_csst_loss = compute_DM_consistency(p1, p2, is_logit=False, func=self.data_cfunc)
            else:
                cross_sample_csst_loss = compute_DM_consistency(logits1 + logits2, logits3 + logits4,
                                                                is_logit=True, func=self.data_cfunc)

            csst_loss = self.alpha * in_sample_csst_loss + self.beta * cross_sample_csst_loss
            loss = cls_loss + csst_loss
            return {"cls": cls_loss.item(),
                    "in_sample_csst": in_sample_csst_loss.item(),
                    "cross_sample_csst": cross_sample_csst_loss.item(),
                    "csst": csst_loss.item(),
                    "agg": loss}

        else:
            return self.model(x, labels=labels)

    def __str__(self):
        return (f"{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta}, "
                f"model_cfunc={self.model_cfunc}, data_cfunc={self.data_cfunc})")


class RDropDAMutualWrapper(ConsistencyWrapper):
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.0,
    ):
        super().__init__(model=model, alpha=alpha)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], labels: Tensor = None):
        """If training, `labels` will be given, and `x` will be a list of two tensors.
        Each x_i will be forwarded twice. Consistency loss will be computed between each pair. Return losses.
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
            split_log_prob = torch.split(F.log_softmax(logits, dim=1), bs, dim=0)
            csst_loss = None
            for _logp1, _logp2 in itertools.product(split_log_prob, repeat=2):
                if _logp1 is _logp2:
                    continue
                kld = F.kl_div(_logp1, _logp2, reduction="batchmean", log_target=True)
                if csst_loss is None:
                    csst_loss = kld
                else:
                    csst_loss += kld

            loss = cls_loss + self.alpha * csst_loss / 12
            return {"cls": cls_loss.item(),
                    "csst": csst_loss.item(),
                    "agg": loss}

        else:
            return self.model(x, labels=labels)
