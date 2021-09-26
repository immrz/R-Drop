from torchvision import datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import torch
import os

from .data_utils import get_transform


class CIFAR100(datasets.CIFAR100):
    def __init__(self, size=-1, **kwargs):
        super().__init__(**kwargs)
        if 0 < size < len(self.data):
            self.data = self.data[:size]
            self.targets = self.targets[:size]


class SemiSupvDataLoader:
    def __init__(self, supv_loader, unsupv_loader):
        self.supv_loader = supv_loader
        self.unsupv_loader = unsupv_loader
        self.dataset = self.supv_loader.dataset
        self.supv_iter = None
        self.unsupv_iter = iter(self.unsupv_loader)

    def __len__(self):
        return len(self.supv_loader)

    def __iter__(self):
        self.supv_iter = iter(self.supv_loader)
        return self

    def __next__(self):
        supv_data = next(self.supv_iter)
        try:
            unsupv_data = next(self.unsupv_iter)
        except StopIteration:
            self.unsupv_iter = iter(self.unsupv_loader)
            unsupv_data = next(self.unsupv_iter)
        supv_x, supv_y = supv_data
        unsupv_x, unsupv_y = unsupv_data
        return (supv_x, unsupv_x), supv_y


def get_uda_loader(args, unsupv_ratio=7, supv_size=4000):
    assert args.dataset == "cifar100" and args.aug_type == "cifar", "Other datasets not supported currently"
    assert args.train_batch_size % (unsupv_ratio + 1) == 0

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    supv_transform_train, transform_test = get_transform(args.aug_type, args.img_size,
                                                         two_aug=False, rand_aug=args.rand_aug)
    unsupv_transform_train, _ = get_transform(args.aug_type, args.img_size,
                                              two_aug=True, rand_aug=args.rand_aug)

    supv_dataset = CIFAR100(size=supv_size, root=args.data_dir, train=True,
                            download=True, transform=supv_transform_train)
    unsupv_dataset = CIFAR100(size=-1, root=args.data_dir, train=True,
                              download=True, transform=unsupv_transform_train)
    testset = CIFAR100(size=-1, root=args.data_dir, train=False, download=True,
                       transform=transform_test) if args.local_rank in [-1, 0] else None

    if args.local_rank == 0:
        torch.distributed.barrier()

    supv_train_sampler = RandomSampler(supv_dataset) if args.local_rank == -1 \
        else DistributedSampler(supv_dataset)
    unsupv_train_sampler = RandomSampler(unsupv_dataset) if args.local_rank == -1 \
        else DistributedSampler(unsupv_dataset)
    test_sampler = SequentialSampler(testset)

    # use more workers if on ITP server
    num_workers = 32 if "AMLT_OUTPUT_DIR" in os.environ else 4

    supv_loader = DataLoader(supv_dataset,
                             sampler=supv_train_sampler,
                             batch_size=args.train_batch_size // (unsupv_ratio + 1),
                             num_workers=num_workers,
                             pin_memory=True)
    unsupv_loader = DataLoader(unsupv_dataset,
                               sampler=unsupv_train_sampler,
                               batch_size=args.train_batch_size // (unsupv_ratio + 1) * unsupv_ratio,
                               num_workers=num_workers,
                               pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=num_workers,
                             pin_memory=True) if testset is not None else None

    train_loader = SemiSupvDataLoader(supv_loader, unsupv_loader)
    return train_loader, test_loader
