from .rdrop import ConsistencyWrapper
from .two_aug import bidirectional_kl_divergence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple
import itertools


class RDropDAWrapper(ConsistencyWrapper):
    def __init__(
        self,
        model: nn.Module,
        consistency: str,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = -1,
    ):
        super().__init__(model=model, alpha=alpha)
        self.consistency = consistency
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        assert self.consistency in ["prob", "logit"]

    def _cat_and_split(self, x: Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]], labels: Tensor):
        bs = x[0].shape[0]
        if isinstance(x[1], Tensor):
            x1, x2 = x
            x = torch.cat([x1, x1.clone(), x2, x2.clone()], dim=0)
            labels = torch.cat([labels, labels.clone(), labels.clone(), labels.clone()], dim=0)
            logits, _ = self.model(x)
            cls_loss = self.ce(logits, labels)
            logits1, logits2, logits3, logits4 = torch.split(logits, bs, dim=0)
        else:
            assert isinstance(x[1], (tuple, list))
            orig_x, (x1, x2) = x
            x = torch.cat([orig_x, x1, x1.clone(), x2, x2.clone()], dim=0)
            labels = torch.cat([labels, labels.clone(), labels.clone(), labels.clone(), labels.clone()], dim=0)
            logits, _ = self.model(x)
            cls_loss = self.ce(logits, labels)
            _, logits1, logits2, logits3, logits4 = torch.split(logits, bs, dim=0)
        return cls_loss, (logits1, logits2, logits3, logits4)

    def forward(self, x: Union[Tensor, Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]], labels: Tensor = None):
        """If training, `labels` will be given, and `x` will be a list of two tensors.
        Each x_i will be forwarded twice. Return losses.
        If testing, return the outputs of `model` instead.
        """
        if self.training:
            # concat x, compute classification loss, and then split into logits
            cls_loss, (logits1, logits2, logits3, logits4) = self._cat_and_split(x, labels)

            # the rdrop loss -> model randomness
            in_sample_csst_loss = (bidirectional_kl_divergence(logits1, logits2)
                                   + bidirectional_kl_divergence(logits3, logits4))

            # the additional classification loss, if any
            additional_cls_loss = 0.0

            # the data consistency loss -> data randomness
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
                if self.gamma > 0:
                    additional_cls_loss = 0.5 * (self.nll(logp1, labels) + self.nll(logp2, labels))

            # csst_loss = 0.5 * (self.beta * in_sample_csst_loss + cross_sample_csst_loss)
            # loss = cls_loss + self.alpha * csst_loss
            csst_loss = self.alpha * in_sample_csst_loss + self.beta * cross_sample_csst_loss
            loss = cls_loss + csst_loss + self.gamma * additional_cls_loss
            return {"cls": cls_loss.item(),
                    "cls_additional": additional_cls_loss.item() if isinstance(additional_cls_loss, Tensor) else 0.,
                    "in_sample_csst": in_sample_csst_loss.item(),
                    "cross_sample_csst": cross_sample_csst_loss.item(),
                    "csst": csst_loss.item(),
                    "agg": loss}

        else:
            return self.model(x, labels=labels)

    def __str__(self):
        return (f"{self.__class__.__name__}(consistency={self.consistency}, alpha={self.alpha}, "
                f"beta={self.beta}, gamma={self.gamma})")


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
