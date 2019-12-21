import os
import sys
import numpy as np
import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

import torchtext
from torchtext.data import Field, LabelField, TabularDataset
from spacy.lang.en import English
from transformers import BertTokenizer

from data import *
from train import *
from evaluate import *
from utils import *

class Params(object):

    # Loggin
    # Free text to describe the experiment
    COMMENT = ''
    VERBOSE = True
    DUMP_LOG = True
    EXP_NAME = "roy_test"
    DATA_PATH = os.path.join(os.path.abspath(''), 'data', 'StyleTransformerData', 'yelp')
    MODELS_PATH = os.path.join(os.path.abspath(''), 'outputs', EXP_NAME)
    PRINT_INTERVAL = 10

    # TODO: for local use
    # DATA_PATH = MODELS_PATH = os.path.abspath(__file__+'/../')
    MODELS_LOAD_PATH = ""

    # Data
    DATASET_NAME = 'YELP'
    # Maximal number of batches for test model
    TEST_MAX_BATCH_SIZE = 300
    # Min freq for word in dataset to include in vocab
    VOCAB_MIN_FREQ = 1
    VOCAB_MAX_SIZE = 30000
    # Whether to use Glove embadding - if TRUE set H_DIM to 300
    VOCAB_USE_GLOVE = False
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    # maximum length of allowed sentence - can be also None
    MAX_LEN = 25

    # Transformer model
    N_LAYERS = 8
    N_LAYERS_CLS = 4
    H_DIM = 300
    N_ATTN_HEAD = 5
    FC_DIM = 2048
    DO_RATE = 0.1
    TRANS_GEN = True

    # Classification model
    N_STYLES = 2
    DO_RATE_CLS = 0.1
    TRANS_CLS = True

    # Train
    N_EPOCHS = 20
    GEN_LR = 3e-4
    CLS_LR = 3e-4
    PERIOD_STEPS = 100
    WARMUP_STEPS = 4000
    GEN_WARMUP_RATIO = 0.2
    CLS_WARMUP_RATIO = 0.2
    BT_LAMBDA = 0.5
    STYLE_LAMBDA = 0.5
    CLS_FACTOR = 0.7
    GEN_FACTOR = 1.0
    WEIGHT_MLM_LOSS = 100.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = Params()
if not(os.path.exists(params.MODELS_PATH)):
    os.makedirs(params.MODELS_PATH)
logger = create_logger(params, params.DUMP_LOG)
pprint_params(params)

### Create data ###
en = English()

def tokenize(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

TEXT = Field(sequential=True, tokenize=tokenize, lower=True, eos_token='<eos>', batch_first=True, fix_length=params.MAX_LEN)
LABEL = LabelField()

fields_list = [('text', TEXT),
                ('label', LABEL)]

logging.info('Creating train dataset')
train_dataset = TabularDataset(
                            path=os.path.join(params.DATA_PATH, "train.csv"), # the root directory where the data lies
                            format='csv',
                            skip_header=True,
                            fields=fields_list)

logging.info('Creating test dataset')
test_dataset = TabularDataset(
                            path=os.path.join(params.DATA_PATH, "test.csv"), # the root directory where the data lies
                            format='csv',
                            skip_header=True,
                            fields=fields_list)

if params.VOCAB_USE_GLOVE:
    logging.info('Start loading pretrained Glove vectors')
    TEXT.build_vocab(train_dataset, test_dataset, min_freq=params.VOCAB_MIN_FREQ, max_size=params.VOCAB_MAX_SIZE ,specials=['[MASK]'], vectors=torchtext.vocab.GloVe(name='6B', dim=params.H_DIM))
    logging.info("Loaded Glove embedding, Vector size of Text Vocabulary: " + str(TEXT.vocab.vectors.size()))

else:
    TEXT.build_vocab(train_dataset, test_dataset, min_freq=params.VOCAB_MIN_FREQ, specials=['[MASK]'])
LABEL.build_vocab(train_dataset)

word_embeddings = TEXT.vocab.vectors
logging.info("Length of Text Vocabulary: " + str(len(TEXT.vocab)))

train_iter = data.BucketIterator(train_dataset,
                                 batch_size=params.TRAIN_BATCH_SIZE,
                                 sort_key=lambda x: len(x.text), repeat=False, shuffle=True,
                                 device=params.device)

test_iter = data.BucketIterator(test_dataset,
                                 batch_size=params.TEST_BATCH_SIZE,
                                 sort_key=lambda x: len(x.text), repeat=False, shuffle=False,
                                 device=params.device)

train_dataset_len = len(train_iter.dataset)
print('Train dataset len: {} Test dataset len: {}'.format(len(train_iter.dataset), len(test_iter.dataset)))

### Init models ###

vocab_size = len(TEXT.vocab)
model_gen, model_cls = init_models(vocab_size, params, word_embeddings)

### Init optimizers ###
opt_gen, opt_cls = init_optimizers(model_gen, model_cls, len(train_iter.dataset), params)

### Init losses ###
cls_criteria = nn.CrossEntropyLoss()
mlm_criteria = nn.CrossEntropyLoss(ignore_index=-1)
cls_criteria = cls_criteria.to(params.device)

seq2seq_criteria = nn.CrossEntropyLoss(reduction='mean', ignore_index=1)
seq2seq_criteria = seq2seq_criteria.to(params.device)

### train reconstruction loss ###
# train_gen_on_rec_loss(train_iter, model_gen, opt_gen, seq2seq_criteria, 10, params)

### pretrain classifier
train_cls(train_iter, model_cls, opt_cls, cls_criteria, mlm_criteria, params, TEXT, epochs=10)

model_cls_path = os.path.join(params.MODELS_PATH, "model_cls_{}_yelp_freq{}_len{}_dim_{}.pth".format(params.N_LAYERS_CLS, params.VOCAB_MIN_FREQ, params.MAX_LEN, params.H_DIM))
model_gen_path = os.path.join(params.MODELS_PATH, "model_gen_{}_yelp_freq{}_len{}_dim_{}.pth".format(params.N_LAYERS, params.VOCAB_MIN_FREQ, params.MAX_LEN, params.H_DIM))

torch.save(model_cls.state_dict(), model_cls_path)
torch.save(model_gen.state_dict(), model_gen_path)

### train ###
for epoch in range(params.N_EPOCHS):
    run_train_epoch(epoch=epoch, data_iter=train_iter,
                model_gen=model_gen, opt_gen=opt_gen,
                model_cls=model_cls, cls_criteria=cls_criteria,
                seq2seq_criteria=seq2seq_criteria, params=params)

    test_random_samples(train_iter, TEXT, model_gen, model_cls, params.device,
                        decode_func=greedy_decode_sent, num_samples=10, transfer_style=True,
                        trans_cls=params.TRANS_CLS)

    model_cls_path = os.path.join(params.MODELS_PATH, "model_cls_{}_e_.pth".format(epoch))
    model_gen_path = os.path.join(params.MODELS_PATH, "model_gen_{}_e_.pth".format(epoch))

    torch.save(model_cls.state_dict(), model_cls_path)
    torch.save(model_gen.state_dict(), model_gen_path)