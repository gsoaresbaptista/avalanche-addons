from copy import deepcopy
from collections import defaultdict
from typing import Dict, List

import torch
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
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

    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Use a custom dataloader to combine samples from the current data
        and memory buffer. """
        if len(self.buffer.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.buffer.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
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
        """ Update the buffer. """
        self.buffer.update(strategy, **kwargs)
        self.model_old = deepcopy(strategy.model)

    def cross_entropy(self, outputs, targets):
        """Calculates cross-entropy with temperature scaling"""
        logp = torch.nn.functional.log_softmax(outputs/self.T, dim=1)
        pre_p = torch.nn.functional.softmax(targets/self.T, dim=1)
        return -torch.mean(torch.sum(pre_p * logp, dim=1)) * self.T * self.T
