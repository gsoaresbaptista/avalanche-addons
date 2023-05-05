import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.models import MTSimpleCNN
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

import sys
sys.path.insert(0, '../.')
from avalanche_addons.plugins import EEIL  # noqa: E402


def main(args):

    # Config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # model
    model = MTSimpleCNN()

    # CL Benchmark Creation
    benchmark = SplitCIFAR10(n_experiences=5, return_task_id=True)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.1,
                    momentum=0.9, weight_decay=1e-4)
    criterion = CrossEntropyLoss()

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open('eeil.txt', 'a'))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=False, experience=True, stream=False
        ),
        loggers=[interactive_logger, text_logger],
    )

    # Choose a CL strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=3,
        eval_mb_size=128,
        device=device,
        evaluator=eval_plugin,
        plugins=[EEIL(mem_size=2000)],
    )

    # train and test loop
    for i, train_task in enumerate(train_stream):
        strategy.train(train_task)
        strategy.eval(test_stream[:i + 1])


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
