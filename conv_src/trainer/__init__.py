from .base_trainer import BaseTrainer
from .consistency_trainer import RDropTrainer, RDropDataAugTrainer
from transformers.trainer import EvalPrediction
import logging

logger = logging.getLogger(__name__)


def compute_accuracy(output: EvalPrediction) -> dict:
    logits, labels = output.predictions, output.label_ids
    pred = logits.argmax(1)
    return {"acc": (pred == labels).mean().item()}


def get_trainer(model, args, extra_args, train_dataset, eval_dataset, compute_metrics=compute_accuracy):
    kwargs = dict(model=model,
                  args=args,
                  extra_args=extra_args,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  compute_metrics=compute_metrics)
    if extra_args.trainer == "base":
        return BaseTrainer(**kwargs)
    elif extra_args.trainer == "rdrop":
        logger.info(f"Using RDrop with alpha={extra_args.alpha}.")
        return RDropTrainer(alpha=extra_args.alpha, **kwargs)
    elif extra_args.trainer == "rdrop_da":
        logger.info(f"Using RDropDataAug with alpha={extra_args.alpha}.")
        return RDropDataAugTrainer(alpha=extra_args.alpha, **kwargs)
    else:
        raise NotImplementedError
