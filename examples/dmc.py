import argparse
import numpy as np
import random

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_bwt
from avalanche.evaluation.metrics.forgetting_bwt import bwt_metrics
from avalanche.evaluation.metrics.forward_transfer import (
    forward_transfer_metrics,
)

from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from avalanche.training.plugins import EarlyStoppingPlugin
from torchvision import models
from avalanche_addons.utils import resnet32
from avalanche_addons.strategy import DMC

import sys

sys.path.insert(0, "../.")
# from avalanche_addons.plugins import DMC  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)


def main(args):
    cifar_mean = [0.5071, 0.4866, 0.4409]
    cifar_std = [0.2009, 0.1984, 0.2023]

    train_transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std),
        ]
    )
    train_cifar = CIFAR100(
        "../data", train=True, transform=train_transform, download=True
    )
    val_cifar = CIFAR100(
        "../data", train=False, transform=val_transform, download=True
    )
    trn_loader = torch.utils.data.DataLoader(
        train_cifar, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_cifar, batch_size=32, shuffle=True
    )

    # Config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # model
    model = resnet32(False)

    # CL Benchmark Creation
    benchmark = SplitCIFAR100(
        n_experiences=5,
        return_task_id=True,
        fixed_class_order=list(range(100)),
        class_ids_from_zero_in_each_exp=True,
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-5
    )
    criterion = CrossEntropyLoss()

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open("dmc.txt", "a"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=True, experience=True, stream=True
        ),
        bwt_metrics(experience=True, stream=True),
        forward_transfer_metrics(experience=True, stream=True),
        loggers=[interactive_logger, text_logger],
    )

    strategy = DMC(
        model=model,
        lr=0.1,
        classes_per_task=160,
        aux_trn_loader=trn_loader,
        aux_val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=32,
        train_epochs=30,
        eval_mb_size=32,
        device=device,
        evaluator=eval_plugin,
        plugins=[
            EarlyStoppingPlugin(patience=5, val_stream_name="train"),
        ],
    )

    # train and test loop
    for i, train_task in enumerate(train_stream):
        strategy.train(train_task)
        strategy.eval(test_stream[: i + 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
