import numpy as np
import logging
import os
import time
from datetime import timedelta

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

def pprint_params(paramsObj):
    logging.info('Params for experiment:')
    for attr in dir(paramsObj):
        if attr.startswith('_'):
            pass
        else:
            logging.info("%s = %r" % (attr, getattr(paramsObj, attr)))

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

class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)

def create_logger(params):
    # Create output folder
    exp_name = 'exp_' + str(params.EXP_NUM)
    root = params.DATA_PATH

    exp_folder = os.path.join(root, exp_name)
    filepath = os.path.join(root, exp_name, 'main_exp_{}.log'.format(params.EXP_NUM))

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    # Create logger
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger, exp_folder
