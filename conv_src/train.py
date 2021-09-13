from utils.parser_utils import parse_args
from trainer import get_trainer
import models

from torchvision import datasets


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root=None, train=False, download=False):
        super().__init__(root=root, train=train, download=download)

    def __getitem__(self, item):
        """Return dict with a 'label' key to be compatible with huggingface Trainer.
        """
        x, y = super().__getitem__(item)
        return {"x": x, "label": y}


def main():
    # get args
    args, extra_args = parse_args()
    for k, v in vars(extra_args).items():
        print(f"{k:>40s}: {str(v):}")
    print(args)

    # get model
    model = getattr(models, extra_args.model)(
        pretrained=extra_args.pretrained,
        survival_prob=extra_args.survival_prob,
        num_classes=100,
    )

    # get datasets
    train_dataset = CIFAR100(root="data", train=True, download=True)
    eval_dataset = CIFAR100(root="data", train=False, download=True)

    # setup trainer
    trainer = get_trainer(
        model=model,
        args=args,
        extra_args=extra_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    if extra_args.dry_run:
        print(model)
        print(trainer)
        return

    # train
    trainer.train()


if __name__ == "__main__":
    main()
