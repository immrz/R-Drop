import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from transformers.trainer import Trainer, TrainingArguments
from timm.optim import create_optimizer_v2

from argparse import Namespace
from typing import Callable


class BaseTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        extra_args: Namespace,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        compute_metrics: Callable,
    ):
        super().__init__(model=model, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, compute_metrics=compute_metrics)
        self.extra_args = extra_args

        # set transform
        self.set_image_transform()

        # get optimizer and lr_scheduler
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(self.args.max_steps, optimizer=self.optimizer)

    def set_image_transform(self):
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
        self.train_dataset.transform = transform_train
        self.eval_dataset.transform = transform_test

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_optimizer_v2(self.model,
                                                 optimizer_name=self.extra_args.opt,
                                                 learning_rate=self.args.learning_rate,
                                                 weight_decay=self.args.weight_decay,
                                                 momentum=self.extra_args.momentum)
        return self.optimizer

    def __repr__(self):
        return self.__class__.__name__
