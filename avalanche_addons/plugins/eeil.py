from copy import deepcopy
from collections import defaultdict
from typing import Dict, List

import torch
from torch.nn import functional as F
from torch.optim import SGD
from avalanche.benchmarks.utils.data_loader import (
    ReplayDataLoader,
    TaskBalancedDataLoader,
)
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.templates import SupervisedTemplate


class EEIL(SupervisedPlugin):
    def __init__(
        self, mem_size, stage_2_epochs=12, noise_grad=True, clipgrad=10000
    ):
        """A simple replay plugin with reservoir sampling."""
        super().__init__()
        self.buffer = ReservoirSamplingBuffer(max_size=mem_size)
        self.T = 2
        self.model_old = None
        self.class_to_tasks: Dict[int, int] = {}
        self.lamb = 1.0
        self.tuning_optim = None
        self.stage_2_epochs = 12
        self.noise_grad = noise_grad
        self.clipgrad = clipgrad

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):
        """Use a custom dataloader to combine samples from the current data
        and memory buffer."""
        if len(self.buffer.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return
        # TODO: remove this code block
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.buffer.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            task_balanced_dataloader=True,
            shuffle=shuffle,
        )

    def before_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate", **kwargs
    ):
        task_id = strategy.clock.train_exp_counter

        cl_idxs: Dict[int, List[int]] = defaultdict(list)

        for c in cl_idxs.keys():
            self.class_to_tasks[c] = task_id

    def before_backward(self, strategy, **kwargs):
        task_id = strategy.clock.train_exp_counter

        if task_id != 0:
            outputs, outputs_old = [], []
            for i in range(task_id + 1):
                outputs.append(
                    strategy.model(strategy.mb_x.to(strategy.device), i)
                )
                if i != task_id:
                    outputs_old.append(
                        self.model_old(strategy.mb_x.to(strategy.device), i)
                    )

            return self.cross_distilled_loss(
                outputs, strategy.mb_y, outputs_old
            )
        else:
            return strategy.loss

    def after_backward(self, strategy, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            strategy.model.parameters(), self.clipgrad
        )
        if self.noise_grad:
            self._noise_grad(
                strategy.model.parameters(), strategy.clock.train_iterations
            )

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(
                torch.randn(p.grad.shape, device=p.grad.device) * variance
            )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        task_id = strategy.clock.train_exp_counter

        #
        self.buffer.update(strategy, **kwargs)
        self.model_old = deepcopy(strategy.model)

        #
        if task_id != 0:
            stage2_str = "| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |"
            loader = TaskBalancedDataLoader(self.buffer.buffer)

            self.tuning_optim = SGD(
                strategy.model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=1e-4,
            )
            self.tuning_optim.zero_grad()

            for e in range(self.stage_2_epochs):
                total, t_acc, t_loss = 0, 0, 0
                for inputs in loader:
                    x = inputs[0].to(strategy.device)
                    y_real = inputs[1].to(strategy.device)

                    outputs, outputs_old = [], []
                    for i in range(task_id):
                        outputs.append(strategy.model(x, i))
                        if i != task_id:
                            outputs_old.append(self.model_old(x, i))

                    loss = self.cross_distilled_loss(
                        outputs, y_real, outputs_old
                    )

                    _, preds = torch.max(torch.cat(outputs, dim=1), 1)
                    t_acc += torch.sum(preds == y_real.data)
                    t_loss += loss.item() * x.size(0)
                    total += x.size(0)

                    self.tuning_optim.zero_grad()
                    loss.backward()
                    self.tuning_optim.step()

                if (e + 1) % (int(self.stage_2_epochs / 4)) == 0:
                    print(
                        stage2_str.format(
                            e + 1, t_loss / total, 100 * t_acc / total
                        )
                    )

    def cross_distilled_loss(self, outputs, targets, outputs_old=None):
        """Calculates cross-distilled function"""
        loss = F.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distilation loss
        if outputs_old:
            for output, output_old in zip(outputs, outputs_old):
                loss += self.lamb * F.binary_cross_entropy(
                    F.softmax(output / self.T, dim=1),
                    F.softmax(output_old / self.T, dim=1),
                )
        return loss
