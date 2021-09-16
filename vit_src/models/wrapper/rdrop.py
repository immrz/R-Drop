import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class RDropWrapper(nn.Module):
    """Wrapper of model. Will train the wrapped model with RDrop. That is,
    forward the input twice, and force the consistency between the two outputs.
    """
    def __init__(
        self,
        model: nn.Module,
        consistency: str,
        consist_func: str,
        alpha: float = 1.0,
        stop_grad: bool = False,
    ) -> None:

        super().__init__()
        self.model = model
        self.consistency = consistency
        self.consist_func = consist_func
        self.alpha = alpha
        self.stop_grad = stop_grad

        assert self.consistency is not None and self.consist_func is not None
        logger.info(f"Consistency loss is imposed on {consistency} with alpha={alpha}, "
                    f"stop_grad={stop_grad}, function={consist_func}.")

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
        """If training, `labels` will be given, and `x` will be forwarded twice. Return the losses.
        If testing, `labels` is unavailable, return the logits and hidden embedding instead.
        """
        if labels is not None:
            # double the inputs and forward
            bs = x.shape[0]  # original batch size
            x, labels = torch.cat([x, x.clone()], dim=0), torch.cat([labels, labels.clone()], dim=0)
            logits, hidden = self.model(x)
            cls_loss = F.cross_entropy(logits, labels, reduction="none")

            # split the output
            if self.consistency == "hidden":
                z1, z2 = torch.split(hidden, bs, dim=0)
            else:
                z1, z2 = torch.split(logits, bs, dim=0)

            # find the output with smaller classification loss
            loss1, loss2 = cls_loss[:bs].sum(), cls_loss[bs:].sum()
            if loss1 < loss2:
                consist_loss = self.compute_consistency_loss(z1, z2)
            else:
                consist_loss = self.compute_consistency_loss(z2, z1)

            cls_loss = cls_loss.mean()
            agg_loss = cls_loss + self.alpha * consist_loss

            return {"cls": cls_loss.item(), "csst": consist_loss.item(), "agg": agg_loss}

        else:
            return self.model(x)
