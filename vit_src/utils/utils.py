import os
import random
import numpy as np
import torch
import argparse
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        if self.count == 0:
            return -1.
        else:
            return self.sum["cls"] / self.count

    def get_avg(self, key):
        if self.count == 0:
            return -1.
        else:
            return self.sum[key] / self.count

    def reset(self):
        self.sum = defaultdict(lambda: 0.)
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, dict):
            for k, v in val.items():
                self.sum[k] += v * n
        else:
            self.sum["cls"] += val * n
        self.count += n


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False


def display_all_param(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))


def display_resnet_layers(net):
    def _draw_border(length=80):
        print("=" * length)

    print(net.conv1)
    num_layer = 4 if hasattr(net, "layer4") else 3
    for i in range(num_layer):
        layer = getattr(net, f"layer{i+1}")
        _draw_border()
        print(f"Layer {i+1}")
        for block in layer:
            print(block)
    _draw_border()
    print(net.fc)
