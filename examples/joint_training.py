import json
import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from torchvision.models import resnet34, ResNet34_Weights

from avalanche_addons.dataset.animals10 import get_animals10
from avalanche.training.supervised import JointTraining
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.evaluation.metrics import accuracy_metrics


def main(args):
    # Config
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    # model
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # CL Benchmark Creation
    perm_mnist = get_animals10(
        "/run/media/gabriel/Arquivos/data_lake/datasets/animals10/raw-img",
        random_state=42,
        n_exp=1,
    )
    train_stream = perm_mnist.train_stream
    test_stream = perm_mnist.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 5, min_lr=1e-4)

    # loggers
    interactive_logger = InteractiveLogger()
    text_logger = TextLogger(open(f"joint_training_{args.seed}.txt", "a"))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=True, experience=True, stream=False
        ),
        loggers=[interactive_logger, text_logger],
    )

    # Joint training strategy
    joint_train = JointTraining(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=60,  # 60
        eval_mb_size=32,
        device=device,
        evaluator=eval_plugin,
        plugins=[LRSchedulerPlugin(scheduler, metric="train_loss")],
    )

    # train and test loop
    results = []
    print("Starting training.")
    # Differently from other avalanche strategies, you NEED to call train
    # on the entire stream.
    joint_train.train(train_stream)
    results.append(joint_train.eval(test_stream))
    print(results)

    with open(f"joint_results_{args.seed}.json", "w") as jsonf:
        json.dump(results[0], jsonf, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    main(args)
