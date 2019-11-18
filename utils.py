import numpy as np
import logging
import os
import time
from datetime import timedelta
from torch.nn import CosineSimilarity

import torch


class Loss(object):

    def __init__(self):
        ''' Running loss metric '''
        self.num_steps = 0.0
        self.total_loss = 0.0

    def update(self, loss):
        ''' Inputs are torch tensors '''

        self.total_loss += loss.item()
        self.num_steps += 1.0

    def __call__(self):
        return self.total_loss / self.num_steps if self.num_steps else self.num_steps

    def reset(self):
        self.num_steps = 0.0
        self.total_loss = 0.0


class AccuracyRec(object):

    def __init__(self, pad_ind=1):
        ''' Running accuracy metric '''

        self.correct = 0.0
        self.total = 0.0
        self.pad_ind = pad_ind

    def update(self, outputs, targets):
        ''' Inputs are torch tensors '''
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        relevant_ids = np.where(targets != self.pad_ind)
        predicted = outputs[relevant_ids].argmax(-1)
        targets = targets[relevant_ids]

        self.total += len(targets)
        self.correct += (predicted == targets).sum().item()

    def __call__(self):
        return self.correct / self.total * 100.0 if self.total else self.total

    def reset(self):
        self.correct = 0.0
        self.total = 0.0


class AccuracyCls(object):

    def __init__(self):
        ''' Running accuracy for classification '''

        self.correct = 0.0
        self.total = 0.0

    def update(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()

    def __call__(self):
        return self.correct / self.total * 100.0 if self.total else self.total

    def reset(self):
        self.correct = 0.0
        self.total = 0.0


def preict_labels(preds):
    return preds.detach().argmax(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_tokens(batch, pad_ind=1):
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
    exp_name = params.EXP_NAME
    root = params.DATA_PATH

    exp_folder = os.path.join(root, exp_name)
    filepath = os.path.join(root, exp_name, '{}.log'.format(params.EXP_NAME))

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


def preds_embedding_cosine_similarity(preds, embedding):
    vocab_size = embedding.lut.num_embeddings
    preds.unsqueeze_(-1)
    preds = preds.expand(-1, -1, -1, vocab_size)
    embeddings = embedding.lut.weight.transpose(0,1).unsqueeze(0).unsqueeze(0)
    embeddings = embeddings.expand(preds.shape[0], preds.shape[1], -1, -1)
    cosine_sim = CosineSimilarity(dim=2)
    return cosine_sim(preds, embeddings)