from .base_trainer import BaseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms
from utils.utils import TwoCropsTransform

from transformers.trainer import TrainingArguments
from argparse import Namespace
from typing import Callable, Dict


def bidirectional_kl_div(logits1, logits2):
    """Compute bidirectional KL Divergence from two logits.
    """
    logp1, p1 = F.log_softmax(logits1, dim=1), F.softmax(logits1, dim=1)
    logp2, p2 = F.log_softmax(logits2, dim=1), F.softmax(logits2, dim=1)
    kld = F.kl_div(logp1, p2, reduction="batchmean")
    kld_reverse = F.kl_div(logp2, p1, reduction="batchmean")
    return 0.5 * (kld + kld_reverse)


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

    def compute_loss(self, model, inputs, return_outputs=False):
        if model.training:
            # double the inputs
            x, y = inputs["x"], inputs["labels"]
            x = torch.cat([x, x.clone()], dim=0)
            y = torch.cat([y, y.clone()], dim=0)

            # forward twice
            outputs = model(x=x, labels=y)
            cls_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # split the logits
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            logits1, logits2 = torch.split(logits, x.size(0) // 2, dim=0)

            # get consistency loss
            csst_loss = bidirectional_kl_div(logits1, logits2)

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


class RDropDataAugTrainer(RDropTrainer):
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
                         eval_dataset=eval_dataset, compute_metrics=compute_metrics, alpha=alpha)

    def set_image_transform(self):
        """Use data augmentation to return two different inputs for training.
        Note that now 'x' is a list of length 2.
        """
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.train_dataset.transform = TwoCropsTransform(base_transform=transform_train)
        self.eval_dataset.transform = transform_test

    def hierarchical_consistency_loss(self, logits):
        """Compute hierarchical consistency loss.
        """
        n = logits.size(0) // 4
        logits1, logits2, logits3, logits4 = torch.split(logits, n, dim=0)
        csst1 = bidirectional_kl_div(logits1, logits2)
        csst2 = bidirectional_kl_div(logits3, logits4)
        csst_btw_aug = bidirectional_kl_div(logits1 + logits2, logits3 + logits4)
        return 0.25 * csst1 + 0.25 * csst2 + 0.5 * csst_btw_aug

    def compute_loss(self, model, inputs, return_outputs=False):
        if model.training:
            # double the inputs
            (x1, x2), y = inputs["x"], inputs["labels"]
            x = torch.cat([x1, x1.clone(), x2, x2.clone()], dim=0)
            y = torch.cat([y, y.clone(), y.clone(), y.clone()], dim=0)

            # forward four times
            outputs = model(x=x, labels=y)
            cls_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # get consistency loss
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            csst_loss = self.hierarchical_consistency_loss(logits)

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
