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