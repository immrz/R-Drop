from .base_trainer import BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers.trainer import TrainingArguments
from argparse import Namespace
from typing import Callable, Dict


class RDropTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        extra_args: Namespace,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        compute_metrics: Callable,
        alpha: float,
    ):
        super().__init__(model=model, args=args, extra_args=extra_args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, compute_metrics=compute_metrics)
        self.alpha = alpha
        self.cls_loss, self.csst_loss, self.denominator = 0., 0., 0

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Override this to log losses besides tr_loss.
        """
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            if self.denominator > 0:
                logs["cls_loss"] = self.cls_loss / self.denominator
                logs["csst_loss"] = self.csst_loss / self.denominator
                self.cls_loss, self.csst_loss, self.denominator = 0., 0., 0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def compute_consistency_loss(self, logits):
        """Compute consistency loss among positive examples.
        """
        # split logits
        n = logits.shape[0] // 2
        logits1, logits2 = torch.split(logits, n, dim=0)

        # compute RDrop loss
        logp1, p1 = F.log_softmax(logits1, dim=1), F.softmax(logits1, dim=1)
        logp2, p2 = F.log_softmax(logits2, dim=1), F.softmax(logits2, dim=1)
        kld = F.kl_div(logp1, p2, reduction="batchmean")
        kld_reverse = F.kl_div(logp2, p1, reduction="batchmean")
        return 0.5 * (kld + kld_reverse)

    def compute_loss(self, model, inputs, return_outputs=False):
        if model.training:
            # double the inputs
            x, y = inputs["x"], inputs["labels"]
            x = torch.cat([x, x.clone()], dim=0)
            y = torch.cat([y, y.clone()], dim=0)

            # forward twice
            outputs = model(x=x, labels=y)
            cls_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # get consistency loss
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            csst_loss = self.compute_consistency_loss(logits)

            # weight the losses
            loss = cls_loss + self.alpha * csst_loss

            # accumulate classification loss and consistency loss for logging
            self.cls_loss += cls_loss.item()
            self.csst_loss += csst_loss.item()
            self.denominator += 1

            return (loss, outputs) if return_outputs else loss

        else:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"
