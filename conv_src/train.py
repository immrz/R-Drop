from utils.parser_utils import parse_args
from utils.utils import set_seed
from trainer import get_trainer
import models

from torchvision import datasets


class CIFAR100(datasets.CIFAR100):
    def __getitem__(self, item):
        """Return dict with a 'label' key to be compatible with huggingface Trainer.
        """
        x, y = super().__getitem__(item)
        return {"x": x, "labels": y}


def main():
    # get args
    args, extra_args = parse_args()

    # set seed
    set_seed(args.seed)

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

    if args.local_rank in [-1, 0]:
        print(args)
        for k, v in vars(extra_args).items():
            print(f"{k:>40s}: {str(v):}")
        print(model)
        print(trainer)

    if extra_args.dry_run:
        return

    # train
    trainer.train()


if __name__ == "__main__":
    main()
