from copy import deepcopy
from collections import defaultdict
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.templates import SupervisedTemplate


class DMC(SupervisedPlugin):
    def __init__(
        self, stage_2_epochs, dataloader, lr, classes_per_task, clipgrad=10000
    ):
        """A simple replay plugin with reservoir sampling."""
        super().__init__()
        self.model_new = None
        self.model_old = None
        self.clipgrad = clipgrad
        self.stage_2_epochs = stage_2_epochs
        self.dataloader = dataloader
        self.lr = lr
        self.classes_per_task = classes_per_task
        self.acc_clases = 0

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):
        self.acc_clases += self.classes_per_task
        self.re_initialize_model(strategy)
        self.linear_in = strategy.model.fc.in_features
        strategy.model.fc = torch.nn.Linear(
            self.linear_in, self.classes_per_task, True
        ).to("cuda")
        self.model_new = deepcopy(strategy.model)

    def after_backward(self, strategy, **kwargs):
        if self.clipgrad:
            torch.nn.utils.clip_grad_norm_(
                strategy.model.parameters(), self.clipgrad
            )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        task_id = strategy.clock.train_exp_counter

        #
        if task_id != 0:
            loss_acc, iterations = 0, 0
            stage2_str = "| E {:3d} | Train: loss={:.3f} |"

            self.model_new = deepcopy(strategy.model)
            self.re_initialize_model(strategy)
            strategy.model.fc = torch.nn.Linear(
                self.linear_in, self.acc_clases, True
            ).to("cuda")

            self.tuning_optim = SGD(
                strategy.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            self.tuning_optim.zero_grad()

            for e in range(self.stage_2_epochs):
                for images, targets in self.dataloader:
                    images, targets = images.to(strategy.device), targets.to(
                        strategy.device
                    )

                    # Forward old and new model
                    targets_old = self.model_old(images)
                    targets_new = self.model_new(images)

                    # Forward current model
                    outputs = strategy.model(images)
                    loss = self.double_loss_distillation(
                        task_id, outputs, targets_old, targets_new
                    )
                    loss_acc += loss
                    iterations += 1
                    # Backward
                    self.tuning_optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        strategy.model.parameters(), self.clipgrad
                    )
                    self.tuning_optim.step()

                print(stage2_str.format(e + 1, loss_acc / iterations))

        self.model_old = deepcopy(strategy.model)

    def double_loss_distillation(
        self, t, outputs, targets_old, targets_new=None
    ):
        with torch.no_grad():
            targets = torch.cat([targets_old, targets_new], dim=1)
            targets -= targets.mean(0)
        return torch.nn.functional.mse_loss(
            outputs, targets.detach(), reduction="mean"
        )

    def re_initialize_model(self, strategy):
        for m in strategy.model.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                m.reset_parameters()
        # Get new model
        self.model_new = deepcopy(strategy.model)
        with torch.no_grad():
            self.model_new.fc.weight.zero_()
            self.model_new.fc.bias.zero_()
            for p in self.model_new.fc.parameters():
                p.requires_grad = False
