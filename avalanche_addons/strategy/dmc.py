from copy import deepcopy
from typing import Optional, Sequence, Union
from avalanche.benchmarks import CLExperience
from avalanche.training.templates.base import ExpSequence

import torch
from avalanche.training.templates import SupervisedTemplate
from avalanche_addons.utils.dmc_train import StudentTrainer

from avalanche.training.templates import BaseSGDTemplate


def reset_child_params(module):
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        reset_child_params(layer)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.children():
        if hasattr(layer, "requires_grad"):
            layer.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    for layer in model.children():
        if hasattr(layer, "requires_grad"):
            layer.requires_grad = True


class DMC(SupervisedTemplate):
    def __init__(
        self,
        model,
        device,
        aux_trn_loader,
        aux_val_loader,
        lr,
        optimizer,
        classes_per_task,
        plugins=None,
        nepochs=3,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=10000,
        momentum=0.9,
        wd=5e-5,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=False,
        eval_on_train=False,
        *args,
        **kwargs
    ):
        super().__init__(model, optimizer)  # type: ignore
        BaseSGDTemplate.__init__(
            self=self,
            model=model,
            optimizer=optimizer,
            device=device,
            *args,
            **kwargs
        )
        self.model_new = None
        self.model_old = None
        self.lr = lr
        self.acc_clases = 0
        self.classes_per_task = classes_per_task
        # second step params
        self.lr_patience = 4
        self.nepochs = nepochs
        self.aux_trn_loader = aux_trn_loader
        self.aux_val_loader = aux_val_loader
        #
        self.clipgrad = clipgrad
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.wu_nepochs = wu_nepochs
        self.wu_lr_factor = wu_lr_factor
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor

    def train(
        self,
        experiences: CLExperience | ExpSequence,
        eval_streams: Sequence[CLExperience | ExpSequence] | None = None,
        **kwargs
    ):
        t = experiences.current_experience

        # initialize
        self.acc_clases += self.classes_per_task

        # add new classification layer
        reset_child_params(self.model)
        self.linear_in = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(
            self.linear_in, self.classes_per_task, True
        ).to(self.device)

        # train
        super().train(experiences, eval_streams, **kwargs)

        if t > 0:
            # freeze new model
            self.model_new = deepcopy(self.model)
            freeze_model(self.model_new)
            self.model_new.eval()

            # create new model
            self.model = deepcopy(self.model)
            reset_child_params(self.model)

            # update classification layer
            self.model.fc = torch.nn.Linear(
                self.linear_in, self.acc_clases, True
            ).to(self.device)
            self.model.train()
            unfreeze_model(self.model)

            # train student with new loss
            dmc_args = dict(
                nepochs=self.nepochs,
                lr=self.lr,
                lr_min=self.lr_min,
                lr_factor=self.lr_factor,
                lr_patience=self.lr_patience,
                clipgrad=self.clipgrad,
                momentum=self.momentum,
                wd=self.wd,
                multi_softmax=self.multi_softmax,
                wu_nepochs=self.warmup_epochs,
                wu_lr_factor=self.warmup_lr,
                fix_bn=self.fix_bn,
            )
            student_trainer = StudentTrainer(
                self.model,
                self.model_new,
                self.model_old,
                self.device,
                **dmc_args
            )
            student_trainer.train_loop(
                t, self.aux_trn_loader, self.aux_val_loader
            )

        self.model_old = deepcopy(self.model)

        return self.evaluator.get_last_metrics()
