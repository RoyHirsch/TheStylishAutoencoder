import numpy as np
import torch
from torch.nn import functional as F

from params import *

def preict_labels(preds):
    return preds.detach().argmax(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_tokens(batch, pad_ind=0):
    batch = batch.detach().cpu().numpy()
    return len(np.where(batch != pad_ind)[0])


class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = np.Inf

        self.is_current_ist_best = False

    def is_new_best_score(self):
        return not (self.counter)

    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score

        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0