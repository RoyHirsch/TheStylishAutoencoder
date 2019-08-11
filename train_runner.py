import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchnlp.datasets import imdb_dataset

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from pytorch_transformers import BertTokenizer

from data import *
from train import *
from utils import *
from params import *

# sys.path.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

output_dataset_file = DATA_PATH + "data.pkl"
try:
    train_dataset, test_dataset = torch.load(output_dataset_file)

except IOError:
    imdb_train = imdb_dataset(train=True)
    imdb_test = imdb_dataset(test=True)

    train_dataset = IMDBDataset(imdb_train, tokenizer)
    test_dataset = IMDBDataset(imdb_test, tokenizer)

    torch.save((train_dataset, test_dataset), output_dataset_file)

print('Train dataset len: {} Test dataset len: {}'.format(len(train_dataset), len(test_dataset)))

train_sampler = RandomSampler(train_dataset, replacement=False)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=TEST_BATCH_SIZE)

# Clear CUDA memory if needed
# TODO: local use
# torch.cuda.empty_cache()

### Init models ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device is {}'.format(device))

vocab_size = len(tokenizer.vocab)
rm = ResourcesManager(vocab_size, MODELS_PATH, "basic", device, load_path=MODELS_PATH)
models = rm.get_models()
model_enc = models["enc"]
model_dec = models["dec"]
model_cls = models["cls"]

### Init losses ###
cls_criteria = nn.CrossEntropyLoss()
seq2seq_criteria = LabelSmoothing(size=vocab_size, padding_idx=0)
ent_criteria = EntropyLoss()

### Init optimizers ###
cls_opt = NoamOpt(H_DIM, 2, OPT_WARMUP_FACTOR,
            torch.optim.Adam(model_cls.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9))
enc_opt = get_std_opt(model_enc)
dec_opt = get_std_opt(model_dec)

model_enc.train()
model_dec.train()
model_cls.train()

for epoch in range(N_EPOCHS):
  run_epoch(epoch, rm, train_dataloader, model_enc, enc_opt, model_dec, dec_opt,
              model_cls, cls_opt, cls_criteria, seq2seq_criteria,
              ent_criteria, device, trans_steps=TRANS_STEPS, cls_steps=CLS_STEPS,
              rec_lambda=REC_LAMBDA, print_interval=PRINT_INTERVAL, verbose=VERBOSE)
