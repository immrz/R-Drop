# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import numpy as np

from datetime import timedelta

import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp

from models.modeling import VisionTransformer, CONFIGS
import models

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.utils import AverageMeter, bool_flag, simple_accuracy, \
    save_model, count_parameters, set_seed, move_to_device

logger = logging.getLogger(__name__)


def setup_ViT(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100
    if args.dataset == "imagenet":
        num_classes=1000

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, alpha=args.alpha)
    model.load_from(np.load(args.pretrained_dir))

    wrapped = models.get_wrapper(
        wrapper=args.wrapper,
        model=model,
        consistency=args.consistency,
        consist_func=args.consist_func,
        alpha=args.alpha,
        stop_grad=args.stop_grad,
    )

    wrapped.to(args.device)
    logger.info("{}".format(config))
    return args, wrapped


def setup_effnet(args):
    num_classes = 10 if args.dataset == "cifar10" else 100
    if args.dataset == "imagenet":
        num_classes = 1000

    model = getattr(models, args.model_type)(
        pretrained=args.pretrained,
        num_classes=num_classes,
        dropout_prob=args.dropout_prob,
        survival_prob=args.prob_end,
        consistency=args.consistency,
        consist_func=args.consist_func,
        alpha=args.alpha,
        stop_grad=args.stop_grad,
    )

    model.to(args.device)
    return args, model


def setup_ResNet(args):
    num_classes = 10 if args.dataset == "cifar10" else 100
    if args.dataset == "imagenet":
        num_classes=1000

    model = getattr(models, args.model_type)(
        pretrained=args.pretrained,
        probs=(args.prob_start, args.prob_end) if args.stoch_depth else (1, 1),
        num_classes=num_classes,
    )
    wrapped = models.get_wrapper(
        wrapper=args.wrapper,
        model=model,
        consistency=args.consistency,
        consist_func=args.consist_func,
        alpha=args.alpha,
        stop_grad=args.stop_grad,
    )

    wrapped.to(args.device)
    return args, wrapped


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.disable_tqdm or args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = move_to_device(batch, args.device)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item(), n=y.size(0))

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.avg)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("valid/loss", scalar_value=eval_losses.avg, global_step=global_step)
    writer.add_scalar("valid/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    if not hasattr(model, "get_custom_transform"):
        transform = None
    else:
        transform = (model.get_custom_transform(is_training=True), model.get_custom_transform(is_training=False))
    train_loader, test_loader = get_loader(args, transform=transform)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # scaler for mixed precision training
    scaler = amp.GradScaler(enabled=args.fp16)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model,
                    device_ids=[args.local_rank],
                    find_unused_parameters=False)  # no need to be True for current stoch_depth implementation

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.disable_tqdm or args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = move_to_device(batch, args.device)
            x, y = batch

            with amp.autocast(enabled=args.fp16):
                loss = model(x, y)
                if isinstance(loss, dict):
                    to_backward = loss.pop("agg") / args.gradient_accumulation_steps
                else:
                    to_backward = loss / args.gradient_accumulation_steps
                    loss = {"cls": loss.item()}

            # scale loss if necessary
            scaler.scale(to_backward).backward()

            # update loss moving average
            losses.update(loss, n=y.size(0))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # unscale gradients for gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # perform updates
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.avg)
                )
                if args.local_rank in [-1, 0]:
                    for loss_key in loss:
                        writer.add_scalar(f"train/{loss_key}_loss",
                                          scalar_value=losses.get_avg(loss_key),
                                          global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    print("Accuracy:", accuracy)
                    if best_acc < accuracy:
                        save_model(args, model)
                        logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    print("Best Accuracy:", best_acc)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "wide_resnet101_2",
                                                 "resnet152", "resnet50", "resnet110",
                                                 "efficientnetv2_m"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="The root of the datasets.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--pretrained", type=bool_flag, default=True, const=True, nargs="?",
                        help="Whether load pretrained model from url.")

    parser.add_argument("--aug_type", type=str, choices=["cifar"], default=None,
                        help="Type of data augmentation to use.")
    parser.add_argument("--two_aug", type=bool_flag, nargs="?", default=False, const=True,
                        help="Create two augmentations in a batch. This will be set by the program automatically.")
    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    # stochastic depth & dropout args
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--stoch_depth", type=bool_flag, default=True, const=True, nargs="?",
                        help="Whether to use stochastic depth.")
    parser.add_argument("--prob_start", type=float, default=1., help="Survival probability of the first layer.")
    parser.add_argument("--prob_end", type=float, default=0.5, help="Survial probability of the last layer.")

    # wrapper args
    parser.add_argument("--wrapper", type=str, default=None, choices=["rdrop", "twoaug", "rdropDA"],
                        help="How to train the model. Default is None, i.e., train as usual.")
    parser.add_argument("--alpha", default=0.3, type=float,
                        help="alpha for kl loss")
    parser.add_argument("--consistency", default=None, type=str, nargs="?", const="prob",
                        choices=["prob", "logit", "hidden"], help="Whether and where to put consistency loss.")
    parser.add_argument("--consist_func", default=None, type=str, choices=["kl", "js", "ce", "cosine", "l2"],
                        help="Type of divergence function if consistency is adopted.")
    parser.add_argument("--stop_grad", action="store_true", help="Whether stop grad for the good submodel.")

    # optimizer args
    parser.add_argument("--learning_rate", default=1e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=200000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("-gas", "--gradient_accumulation_steps", type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--dry_run", action="store_true", help="Display model and exit.")

    args = parser.parse_args()

    # disable tqdm if on itp server
    args.disable_tqdm = "AMLT_OUTPUT_DIR" in os.environ

    # must use two augmentations if wrapper is twoaug or rdropDA
    args.two_aug = args.wrapper in ["twoaug", "rdropDA"]

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    if args.model_type.startswith("ViT"):
        args, model = setup_ViT(args)
    elif args.model_type.startswith("efficientnet"):
        args, model = setup_effnet(args)
    else:
        args, model = setup_ResNet(args)

    # log args and model
    if args.local_rank in [-1, 0]:
        for k, v in vars(args).items():
            print(f"{k:>40s}: {str(v)}")
        print(repr(model))
        print(f"Number of parameters of the model: {count_parameters(model):.1f}M")
    if args.dry_run:
        return

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
