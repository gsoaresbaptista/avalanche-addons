import time
import torch
import numpy as np
from copy import deepcopy


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


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(
        self,
        model,
        device,
        nepochs=100,
        lr=0.05,
        lr_min=1e-4,
        lr_factor=10,
        lr_patience=5,
        clipgrad=10000,
        momentum=0,
        wd=0,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=False,
        eval_on_train=False,
        exemplars_dataset=None,
    ):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None

    def _get_optimizer(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            momentum=self.momentum,
        )

    def train(self, t, trn_loader, val_loader):
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(
                self.model.fc.parameters(), lr=self.warmup_lr
            )
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.fc.train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(
                        outputs[t],
                        targets.to(self.device) - self.model.task_offset[t],
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.fc.parameters(), self.clipgrad
                    )
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(
                            outputs[t],
                            targets.to(self.device)
                            - self.model.task_offset[t],
                        )
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (
                                self.model.task_cls.cumsum(0) <= targets[m]
                            ).sum()
                            pred[m] = (
                                outputs[this_task][m].argmax()
                                + self.model.task_offset[this_task]
                            )
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = (
                    total_loss / total_num,
                    total_acc_taw / total_num,
                )
                warmupclock2 = time.time()
                print(
                    (
                        "| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | "
                        "Train: loss={:.3f}, TAw acc={:5.1f}% |"
                    ).format(
                        e + 1,
                        warmupclock1 - warmupclock0,
                        warmupclock2 - warmupclock1,
                        trn_loss,
                        100 * trn_acc,
                    )
                )

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = deepcopy(self.model)

        self.optimizer = self._get_optimizer()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print(
                    (
                        "| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: "
                        "loss={:.3f}, TAw acc={:5.1f}% |"
                    ).format(
                        e + 1,
                        clock1 - clock0,
                        clock2 - clock1,
                        train_loss,
                        100 * train_acc,
                    ),
                    end="",
                )
            else:
                print(
                    "| Epoch {:3d}, time={:5.1f}s | Train: skip eval |".format(
                        e + 1, clock1 - clock0
                    ),
                    end="",
                )

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(
                " Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |".format(
                    clock4 - clock3, valid_loss, 100 * valid_acc
                ),
                end="",
            )

            # Adapt learning rate - patience scheme
            # early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and
                # end line with a star ( * )
                best_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                patience = self.lr_patience
                print(" *", end="")
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(" lr={:.1e}".format(lr), end="")
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop
                        # the training session
                        print()
                        break
                    # reset patience and recover best model so far
                    # to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]["lr"] = lr
                    self.model.load_state_dict(best_model)
            print()
        self.model.load_state_dict(best_model)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        unfreeze_model(self.model)
        self.model.train()

        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clipgrad
            )
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return (
            total_loss / total_num,
            total_acc_taw / total_num,
            total_acc_tag / total_num,
        )

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = (
                outputs[this_task][m].argmax()
                + self.model.task_offset[this_task]
            )
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [
                torch.nn.functional.log_softmax(output, dim=1)
                for output in outputs
            ]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(
            outputs[t], targets - self.model.task_offset[t]
        )


class StudentTrainer(Inc_Learning_Appr):
    def __init__(
        self,
        model,
        model_new,
        model_old,
        device,
        nepochs=160,
        lr=0.1,
        lr_min=1e-4,
        lr_factor=10,
        lr_patience=8,
        clipgrad=10000,
        momentum=0.9,
        wd=5e-4,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=False,
        eval_on_train=False,
        logger=None,
    ):
        super(StudentTrainer, self).__init__(
            model,
            device,
            nepochs,
            lr,
            lr_min,
            lr_factor,
            lr_patience,
            clipgrad,
            momentum,
            wd,
            multi_softmax,
            wu_nepochs,
            wu_lr_factor,
            fix_bn,
            eval_on_train,
            logger,
        )

        self.model_old = model_old
        self.model_new = model_new
        freeze_model(self.model_old)
        freeze_model(self.model_new)

    # Runs a single epoch of student's training
    def train_epoch(self, t, trn_loader):
        self.model.train()
        unfreeze_model(self.model)
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.cuda(), targets.cuda()
            # Forward old and new model
            targets_old = self.model_old(images)
            targets_new = self.model_new(images)
            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets_old, targets_new)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clipgrad
            )
            self.optimizer.step()

    # Contains the evaluation code for evaluating the student
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_num = 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images = images.cuda()
                # Forward old and new model
                targets_old = self.model_old(images)
                targets_new = self.model_new(images)
                outputs = self.model(images)
                # Forward current model
                loss = self.criterion(t, outputs, targets_old, targets_new)
                # Log
                total_loss += loss.item() * len(targets)
                total_num += len(targets)
        return total_loss / total_num, -1, -1

    # Returns the loss value for the student
    def criterion(self, t, outputs, targets_old, targets_new=None):
        # Eq. 2: Model Consolidation
        with torch.no_grad():
            # Eq. 4: "The regression target of the consolidated model is
            # the concatenation of normalized logits of
            # the two specialist models."
            targets = torch.cat([targets_old, targets_new], dim=1)
            targets -= targets.mean(0)
        # Eq. 3: Double Distillation Loss
        return torch.nn.functional.mse_loss(
            outputs, targets.detach(), reduction="mean"
        )


class NewTaskTrainer(Inc_Learning_Appr):
    def __init__(
        self,
        model,
        device,
        nepochs=160,
        lr=0.1,
        lr_min=1e-4,
        lr_factor=10,
        lr_patience=8,
        clipgrad=10000,
        momentum=0.9,
        wd=5e-4,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=False,
        eval_on_train=False,
        logger=None,
    ):
        super(NewTaskTrainer, self).__init__(
            model,
            device,
            nepochs,
            lr,
            lr_min,
            lr_factor,
            lr_patience,
            clipgrad,
            momentum,
            wd,
            multi_softmax,
            wu_nepochs,
            wu_lr_factor,
            fix_bn,
            eval_on_train,
            logger,
        )
