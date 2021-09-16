import logging
import os

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args, transform=None):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if transform is not None:
        transform_train, transform_test = transform
    else:
        transform_train, transform_test = get_transform(args.aug_type, args.img_size, two_aug=args.two_aug)

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "imagenet":
        trainset = datasets.ImageFolder(root=os.path.join(args.data_dir, "trainset"), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(args.data_dir, "val"), transform=transform_test) \
            if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root=args.data_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir,
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)

    # use more workers if on ITP server
    num_workers = 32 if "AMLT_OUTPUT_DIR" in os.environ else 4
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=num_workers,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


def get_transform(aug_type: str, img_size, two_aug=False):
    if aug_type is None:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    elif aug_type == "cifar":
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

    else:
        raise NotImplementedError

    if two_aug:
        transform_train = TwoCropsTransform(transform_train)

    return transform_train, transform_test


class TwoCropsTransform:
    """Take two random crops of one image as the query and key.
    From https://github.com/facebookresearch/simsiam/blob/main/simsiam/loader.py.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
