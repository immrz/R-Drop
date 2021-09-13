from transformers import HfArgumentParser, TrainingArguments
from argparse import ArgumentTypeError, Namespace
from typing import Tuple


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def parse_args() -> Tuple[TrainingArguments, Namespace]:
    parser = HfArgumentParser(TrainingArguments)

    # add absent arguments
    parser.add_argument("--model", choices=["resnet152", "resnet50", "resnet110", "efficientnetv2_m"],
                        required=True, type=str, help="Model to use.")
    parser.add_argument("--dataset", choices=["cifar100"], required=True, type=str, help="Dataset to use.")

    parser.add_argument("--pretrained", action="store_true", help="Whether load pretrained model.")
    parser.add_argument("--trainer", default="base", type=str, help="Training strategy.")
    parser.add_argument("--dry_run", action="store_true", help="Print args and exit.")

    parser.add_argument("--opt", default="sgd", type=str, metavar="OPTIMIZER",
                        help="Optimizer (default: sgd)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="Optimizer momentum (default: 0.9)")

    parser.add_argument("--survival_prob", default=0.8, type=float,
                        help="Survival prob of the last layer if stochastic depth is used.")
    parser.add_argument("--consistency", default=None, type=str, choices=["prob", "logit", "hidden"],
                        help="Whether and where to put consistency loss.")
    parser.add_argument("--stop_grad", action="store_true", help="Whether stop grad for the good submodel.")
    parser.add_argument("--alpha", default=1.0, type=float, help="Weight for consistency loss.")

    # add consistency loss type
    args, _ = parser.parse_known_args()
    if args.consistency == "prob":
        parser.add_argument("--consist_func", default="kl", type=str, choices=["kl", "js", "ce"],
                            help="Type of divergence function if consistency is set to prob.")
    elif args.consistency in ["logit", "hidden"]:
        parser.add_argument("--consist_func", default="cosine", type=str, choices=["cosine", "l2"],
                            help="Type of divergence function if consistency is set to hidden or logit.")
    else:
        parser.set_defaults(consist_func=None)

    # TrainingArguments and extra customized args
    args, extra_args = parser.parse_args_into_dataclasses(return_remaining_strings=False, look_for_args_file=False)

    return args, extra_args


if __name__ == "__main__":
    args, extra_args = parse_args()
    for k, v in vars(extra_args).items():
        print(f"{k:>40s}: {str(v):}")
    print(args)
