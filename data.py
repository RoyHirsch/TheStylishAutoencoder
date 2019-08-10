import numpy as np
import torch
from torch.utils.data import DataLoader

from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup

from params import *

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_masks(src, tgt, device, pad=0):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    tgt_mask = tgt_mask.to(device)

    # Add vector of one's for style embadding
    bs, max_len, h_size = tgt_mask.size()
    tgt_mask = torch.cat((torch.ones(bs, max_len, 1).byte().to(device), tgt_mask), 2)
    tgt_mask = tgt_mask[:, :, :-1]

    src_mask = (src != pad).unsqueeze(-2)
    return src_mask, tgt_mask


class IMDBDataset(torch.utils.data.Dataset):
    ''' Generate sentences with gt target for encoding '''

    def __init__(self, imdb_dataset, tokenizer, max_len=MAX_LEN):
        self.data = self._process_data(imdb_dataset)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _process_data(self, imdb_dataset):
        ''' From raw data to processed sentences '''
        data_list = []
        for row in imdb_dataset.rows:
            text = BeautifulSoup(row["text"], 'html.parser').get_text()
            target = True if row['sentiment'] == 'pos' else False

            sents = sent_tokenize(text)
            for sent in sents:
                data_list.append({'text': sent, 'label': target})
        return data_list

    def __getitem__(self, ind):
        item = self.data[ind]

        data = item['text']
        target = item['label']

        indexed_tokens = self.tokenizer.encode(data)
        curr_len = len(indexed_tokens)

        if curr_len < self.max_len:
            indexed_tokens = indexed_tokens + np.zeros((self.max_len - curr_len)).tolist()

        elif curr_len > self.max_len:
            indexed_tokens = indexed_tokens[:self.max_len]

        else:
            pass

        src = torch.tensor(indexed_tokens).long()
        label = torch.tensor(int(target))
        return src, label

    def __len__(self):
        return len(self.data)