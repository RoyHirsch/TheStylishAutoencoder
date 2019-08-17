import os
import sys
import numpy as np
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchnlp.datasets import imdb_dataset

from data import *
from train import *
from evaluate import *
from utils import *

class Params(object):

    # Loggin
    # Free text to describe the experiment
    COMMENT = ''
    VERBOSE = True
    EXP_NUM = 0
    # DATA_PATH = "/content/drive/My Drive/NLP/final/"
    # MODELS_PATH = "/content/drive/My Drive/NLP/final/"
    PRINT_INTERVAL = 100

    # TODO: for local use
    DATA_PATH = MODELS_PATH = os.path.abspath(__file__+'/../')

    # Data
    DATASET_NAME = 'IMDB'
    # Maximal number of batches for test model
    TEST_MAX_BATCH_SIZE = 500
    # Min freq for word in dataset to include in vocab
    VOCAB_MIN_FREQ = 5
    # Whether to use Glove embadding - if TRUE set H_DIM to 300
    VOCAB_USE_GLOVE = False
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    # maximum length of allowed sentence - can be also None
    MAX_LEN = 128

    # Transformer model
    N_LAYERS = 6
    H_DIM = 512
    N_ATTN_HEAD = 8
    FC_DIM = 2048
    DO_RATE = 0.1

    # Classification model
    N_STYLES = 2
    DO_RATE_CLS = 0.5

    # Train
    N_EPOCHS = 100
    PATIENCE = 3
    LR = 0.1
    OPT_WARMUP_FACTOR = 4000
    REC_LAMBDA = 0.5
    CLS_STEPS = 500
    TRANS_STEPS = 500

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = Params()
logger = create_logger(params)
pprint_params(params)

TEXT, word_embeddings, train_iter, test_iter = load_dataset(params=params, device=params.device)
logging.info('Train dataset len: {} Test dataset len: {}'.format(len(train_iter.dataset), len(test_iter.dataset)))

# Clear CUDA memory if needed
# TODO: local use
# torch.cuda.empty_cache()

### Init models ###

vocab_size = len(TEXT.vocab)
rm = ResourcesManager(vocab_size, "basic_exp_{}".format(params.EXP_NUM), params, load_path=None)
models = rm.get_models()
model_enc = models["enc"]
model_dec = models["dec"]
model_cls = models["cls"]

# TODO: if using this option set H_DIM = 300
# model_enc = load_pretrained_embedding_to_encoder(model_enc, word_embeddings)

### Init losses ###
cls_criteria = nn.CrossEntropyLoss()
# seq2seq_criteria = LabelSmoothing(size=vocab_size, padding_idx=1)
seq2seq_criteria = nn.CrossEntropyLoss(reduction='mean', ignore_index=1)
ent_criteria = EntropyLoss()

### Init optimizers ###
cls_opt = get_std_opt(model_cls, params)
enc_opt = get_std_opt(model_enc, params)
dec_opt = get_std_opt(model_dec, params)

model_enc.train()
model_dec.train()
model_cls.train()
early_stop = EarlyStopping(params.PATIENCE)

for epoch in range(Params.N_EPOCHS):
  logging.info('Epoch {}:'.format(epoch))

  run_epoch(epoch, train_iter, model_enc, enc_opt, model_dec, dec_opt,
              model_cls, cls_opt, cls_criteria, seq2seq_criteria,
              ent_criteria, params)

  rm.save_models_on_epoch_end(epoch)
  test_acc = evaluate(epoch, test_iter, model_enc, model_dec,
             model_cls, cls_criteria, seq2seq_criteria,
             ent_criteria, params)

  test_random_samples(test_iter, TEXT, model_enc, model_dec, model_cls, params.device,
                      decode_func=greedy_decode_sent, num_samples=2, transfer_style=True)

  # TODO - Roy - currently not in use, what metric to follow ?
  # early_stop(test_acc)
  # if early_stop.early_stop:
  #     break

