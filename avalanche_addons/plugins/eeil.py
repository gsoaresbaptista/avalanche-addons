from copy import deepcopy
from collections import defaultdict
from typing import Dict, List

import torch
from torch.optim import SGD
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader, TaskBalancedDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.templates import SupervisedTemplate


# TODO: add fine-tuning
# TODO: change buffer method
# TODO: add noise

class EEIL(SupervisedPlugin):

    def __init__(self, mem_size):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.buffer = ReservoirSamplingBuffer(max_size=mem_size)
        self.T = 2
        self.model_old = None
        self.class_to_tasks: Dict[int, int] = {}
        self.lamb = 1.0
        self.tuning_optim = None
        self.stage_2_epochs = 12

    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Use a custom dataloader to combine samples from the current data
        and memory buffer. """
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
            shuffle=shuffle)

    def before_train_dataset_adaptation(
        self,
        strategy: "SupervisedTemplate",
        **kwargs
    ):
        task_id = strategy.clock.train_exp_counter

        cl_idxs: Dict[int, List[int]] = defaultdict(list)

        for c in cl_idxs.keys():
            self.class_to_tasks[c] = task_id

    def before_backward(self, strategy, **kwargs):
        # Distill
        task_id = strategy.clock.train_exp_counter

        if task_id != 0:
            out_new = strategy.model(strategy.mb_x.to(strategy.device))
            out_old = self.model_old(strategy.mb_x.to(strategy.device))
            #
            old_clss = []
            for c in self.class_to_tasks.keys():
                if self.class_to_tasks[c] < task_id:
                    old_clss.append(c)

            #
            loss_dist = self.cross_entropy(
                out_new[:, old_clss],
                out_old[:, old_clss])

            return strategy.loss + self.lamb * loss_dist
        else:
            return strategy.loss

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        task_id = strategy.clock.train_exp_counter

        #
        self.buffer.update(strategy, **kwargs)
        self.model_old = deepcopy(strategy.model)

        #

        if task_id != 0:
            stage2_str = '| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |'
            loader = TaskBalancedDataLoader(self.buffer.buffer)

            self.tuning_optim = SGD(strategy.model.parameters(), lr=0.1,
                                    momentum=0.9, weight_decay=1e-4)
            self.tuning_optim.zero_grad()

            for e in range(self.stage_2_epochs):
                total, t_acc, t_loss = 0, 0, 0
                for inputs in loader:
                    x = inputs[0].to(strategy.device)
                    y_real = inputs[1].to(strategy.device)

                    out_new = strategy.model(x)
                    out_old = self.model_old(x)
                    #
                    old_clss = []
                    for c in self.class_to_tasks.keys():
                        if self.class_to_tasks[c] < task_id:
                            old_clss.append(c)

                    #
                    loss_dist = self.cross_entropy(
                        out_new[:, old_clss],
                        out_old[:, old_clss])

                    loss = torch.nn.CrossEntropyLoss()(out_new, y_real)

                    _, preds = torch.max(out_new, 1)
                    t_acc += torch.sum(preds == y_real.data)
                    t_loss += loss.item() * x.size(0)
                    total += x.size(0)

                    loss += self.lamb * loss_dist

                    self.tuning_optim.zero_grad()
                    loss.backward()
                    self.tuning_optim.step()

                if (e + 1) % (int(self.stage_2_epochs / 4)) == 0:
                    print(stage2_str.format(e + 1, t_loss / total,
                                            100 * t_acc / total))

    def cross_entropy(self, outputs, targets):
        """Calculates cross-entropy with temperature scaling"""
        logp = torch.nn.functional.log_softmax(outputs/self.T, dim=1)
        pre_p = torch.nn.functional.softmax(targets/self.T, dim=1)
        return -torch.mean(torch.sum(pre_p * logp, dim=1)) * self.T * self.T
