import logging
import os
from typing import Tuple, Any
from PIL import Image

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform


logger = logging.getLogger(__name__)


class KeepOrigImgCIFAR100(datasets.CIFAR100):
    def __init__(self, resize_transform, **kwargs):
        super().__init__(**kwargs)
        assert resize_transform is not None
        self.resize_transform = resize_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # `resize_transform` only resizes the image, so this is the original image
        orig_img = self.resize_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (orig_img, img), target


def get_loader(args, transform=None):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if transform is not None:
        transform_train, transform_test = transform
    else:
        transform_train, transform_test = get_transform(args.aug_type, args.img_size, rand_aug=args.rand_aug)
    # if use two augmentations in a batch
    if args.two_aug:
        transform_train = TwoCropsTransform(transform_train)

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
        if not args.keep_original_image:
            trainset = datasets.CIFAR100(root=args.data_dir,
                                         train=True,
                                         download=True,
                                         transform=transform_train)
        else:
            trainset = KeepOrigImgCIFAR100(root=args.data_dir,
                                           train=True,
                                           download=True,
                                           transform=transform_train,
                                           resize_transform=transform_test)
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


def get_transform(aug_type: str, img_size, rand_aug=None):
    if aug_type is None:
        # determine the augmentation to use
        if rand_aug is None:
            # default augmentation
            aug = [transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0))]
        elif rand_aug.startswith("augmix"):
            # AugMix
            aug = [transforms.Resize((img_size, img_size)),
                   augment_and_mix_transform(config_str=rand_aug, hparams={})]
        else:
            # RandAugment
            aug = [transforms.Resize((img_size, img_size)),
                   rand_augment_transform(config_str=rand_aug, hparams={})]
        transform_train = transforms.Compose([
            *aug,
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

    def __repr__(self):
        return "TwoCropsTransform(" + repr(self.base_transform) + ")"
