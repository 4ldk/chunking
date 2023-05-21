import numpy as np
from timm.scheduler import CosineLRScheduler
from torch import optim


def recall_score(target, pred, average=None):
    if type(target) not in [list, tuple]:
        target = target.tolist()
        pred = pred.tolist()
    target_type = list(set(target))
    count_dict = {t_t: [0, 0] for t_t in target_type}
    for tar, pre in zip(target, pred):
        if tar == pre:
            count_dict[tar][0] += 1
        else:
            count_dict[tar][1] += 1
    count_dict = [c_d[0] / sum(c_d) for c_d in count_dict.values()]
    if average is not None:
        count_dict = sum(count_dict) / len(count_dict)
    return np.array(count_dict)


def precision_score(target, pred, average=None):
    if type(target) not in [list, tuple]:
        target = target.tolist()
        pred = pred.tolist()
    pred_type = list(set(pred))
    count_dict = {p_t: [0, 0] for p_t in pred_type}
    for tar, pre in zip(target, pred):
        if tar == pre:
            count_dict[pre][0] += 1
        else:
            count_dict[pre][1] += 1
    count_dict = [c_d[0] / sum(c_d) for c_d in count_dict.values()]
    if average is not None:
        count_dict = sum(count_dict) / len(count_dict)
    return np.array(count_dict)


def f1_score(target, pred):
    recall = recall_score(target, pred, average="micro")
    prec = precision_score(target, pred, average="micro")

    return 2 / (1 / recall + 1 / prec)


class CosineScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, **kwargs):
        self.init_lr = optimizer.param_groups[0]["lr"]
        self.timmsteplr = CosineLRScheduler(optimizer, **kwargs)
        super().__init__(optimizer, self)

    def __call__(self, epoch):
        desired_lr = self.timmsteplr.get_epoch_values(epoch)[0]
        mult = desired_lr / self.init_lr
        return mult
