from bs4 import BeautifulSoup
import torch
import numpy as np
import logging
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from spacy.lang.en import English


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_masks(src, tgt, device, pad=1):
    ''' Pad id in TEXT.vocab.stoi['<pad>'] = 1'''
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    tgt_mask = tgt_mask.to(device)

    # Add vector of one's for style embadding
    bs, max_len, h_size = tgt_mask.size()
    tgt_mask = torch.cat((torch.ones(bs, max_len, 1).byte().to(device), tgt_mask), 2)
    tgt_mask = tgt_mask[:, :, :-1]

    src_mask = (src != pad).unsqueeze(-2)
    return src_mask, tgt_mask

def load_dataset(params, device):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """
    # define tokenizer
    en = English()
    def tokenize_spacy_with_html_parsing(sentence):
        sentence = BeautifulSoup(sentence, 'html.parser').get_text()
        return [tok.text for tok in en.tokenizer(sentence)]

    # eos_token - end of sentence token, batch_first - first dimension is batch, fix_length - can be also None
    TEXT = data.Field(sequential=True, tokenize=tokenize_spacy_with_html_parsing,
                      lower=True, eos_token='<eos>', batch_first=True, fix_length=params.MAX_LEN)
    LABEL = data.LabelField()

    data_source = params.DATASET_NAME
    logging.info('Start loading dataset {}:'.format(data_source))

    if data_source == 'IMDB':
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    elif data_source == 'SST':
        train_data, test_data = datasets.SST.splits(TEXT, LABEL)

    if params.VOCAB_USE_GLOVE:
        TEXT.build_vocab(train_data, test_data, min_freq=params.VOCAB_MIN_FREQ, vectors=GloVe(name='6B', dim=300))
        logging.info("Loaded Glove embedding, Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())

    else:
        TEXT.build_vocab(train_data, test_data, min_freq=params.VOCAB_MIN_FREQ)
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    logging.info("Length of Text Vocabulary: " + str(len(TEXT.vocab)))

    train_iter, test_iter = data.BucketIterator.splits((train_data, test_data),
                                                       batch_sizes=(params.TRAIN_BATCH_SIZE, params.TRAIN_BATCH_SIZE),
                                                       sort_key=lambda x: len(x.text), repeat=False, shuffle=True,
                                                       device=device)
    # Disable shuffle
    test_iter.shuffle = False
    return TEXT, word_embeddings, train_iter, test_iter

def get_data_loaders(params, TEXT, LABEL):
    data_source = params.DATASET_NAME
    output_dataset_file = params.DATA_PATH + "data_{}.pkl".format(data_source)

    if data_source == 'IMDB':
        try:
            train_dataset, test_dataset = torch.load(output_dataset_file)

        except IOError:
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            torch.save((train_data, test_data), output_dataset_file)

    elif data_source == 'SST':
        try:
            train_dataset, test_dataset = torch.load(output_dataset_file)

        except IOError:
            train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
            torch.save((train_data, test_data), output_dataset_file)
    else:
        raise ValueError('Invalid data source ' + data_source)
    return train_dataset, test_dataset

