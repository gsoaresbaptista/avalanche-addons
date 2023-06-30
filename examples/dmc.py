import argparse
import numpy as np
import random

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from avalanche.training.plugins import EarlyStoppingPlugin
from torchvision import models
from avalanche_addons.utils import resnet32

import sys

sys.path.insert(0, "../.")
from avalanche_addons.plugins import DMC  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)


def get_animals10():
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
    from avalanche.benchmarks.generators import nc_benchmark

    T = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = ImageFolder(
        "/home/gabriel/Documents/github/data/archive/raw-img", transform=T
    )
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
    )
    scenario = nc_benchmark(
        train_set,
        val_set,
        n_experiences=5,
        shuffle=True,
        seed=1234,
        class_ids_from_zero_in_each_exp=True,
        task_labels=False,
    )
    return scenario


def main(args):
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std),
        ]
    )
    cifar = CIFAR100(
        "../data", train=False, transform=transform, download=True
    )
    cifar = torch.utils.data.DataLoader(cifar, batch_size=256, shuffle=True)

    # Config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # model
    # model = models.resnet50()
    model = resnet32(False)

    # CL Benchmark Creation
    benchmark = SplitCIFAR100(
        n_experiences=5,
        return_task_id=True,
        class_ids_from_zero_in_each_exp=True,
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    criterion = CrossEntropyLoss()

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open("dmc.txt", "a"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=True, experience=True, stream=False
        ),
        loggers=[interactive_logger, text_logger],
    )

    # Choose a CL strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=256,
        train_epochs=300,
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin,
        plugins=[
            DMC(
                stage_2_epochs=300,
                lr=0.001,
                classes_per_task=20,
                dataloader=cifar,
            ),
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
