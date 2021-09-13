from .base_trainer import BaseTrainer
from transformers.trainer import EvalPrediction


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
    else:
        raise NotImplementedError
