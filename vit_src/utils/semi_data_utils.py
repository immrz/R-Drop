from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import torch
import os
from timm.data.auto_augment import rand_augment_transform
from PIL import Image


class SemiSupvCIFAR100(datasets.CIFAR100):
    def __init__(self, size=-1, **kwargs):
        super().__init__(**kwargs)
        self.normal_aug_train = [transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.id_transform = [transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        if not self.train:
            # test set
            self.transform = transforms.Compose(self.id_transform)
        else:
            # train set
            if 0 < size < len(self.data):
                # supervised train set
                self.data = self.data[:size]
                self.targets = self.targets[:size]
                self.transform = transforms.Compose(self.normal_aug_train)
            else:
                # unsupervised train set
                ra = rand_augment_transform(config_str="rand-m10-n2-mstd200", hparams={})
                self.transform = [transforms.Compose(self.normal_aug_train),
                                  transforms.Compose([ra] + self.normal_aug_train)]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # for supervised trainset or test, img is a Tensor;
        # for unsupervised trianset, img is a tuple of two Tensors, where the first is *original* image,
        # and the second is *augmented* image.
        if self.transform is not None:
            if isinstance(self.transform, (tuple, list)):
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


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

    supv_dataset = SemiSupvCIFAR100(size=supv_size, root=args.data_dir, train=True, download=True)
    unsupv_dataset = SemiSupvCIFAR100(size=-1, root=args.data_dir, train=True, download=True)
    testset = SemiSupvCIFAR100(size=-1, root=args.data_dir, train=False, download=True) \
        if args.local_rank in [-1, 0] else None

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
