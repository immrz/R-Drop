from utils.parser_utils import parse_args
from trainer import BaseTrainer
import models

from torchvision import datasets


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root=None, train=False, download=False):
        super().__init__(root=root, train=train, download=download)

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        return {"x": x, "label": y}


def main():
    args, extra_args = parse_args()
    model = getattr(models, extra_args.model)(
        pretrained=extra_args.pretrained,
        survival_prob=extra_args.survival_prob,
        num_classes=100,
    )

    train_dataset = CIFAR100(root="data", train=True, download=True)
    eval_dataset = CIFAR100(root="data", train=False, download=True)

    trainer = BaseTrainer(model, args, extra_args, train_dataset, eval_dataset)

    trainer.train()


if __name__ == "__main__":
    main()
