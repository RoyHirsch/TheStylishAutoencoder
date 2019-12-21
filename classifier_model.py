import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_model import MultiHeadedAttention, PositionwiseFeedForward, BasicEncoder, EncoderLayer, Embeddings, \
    ArgMaxEmbed, PositionalEncoding
import copy


class MaskedMean(nn.Module):
    " Calculate masked mean of input 3D tensor "

    def __init__(self, normalize=True):
        self.normalize = normalize
        super().__init__()

    def forward(self, x, mask):
        batch_size, _, embed_size = x.size()
        mask_expanded = mask.transpose(-1, -2).expand(-1, -1, embed_size).float()
        masked_input = x * mask_expanded
        sum_ = torch.sum(masked_input, 1)
        div = mask.sum(-1).float()
        embed = torch.div(sum_, div.view(batch_size, 1))
        if self.normalize:
            return F.normalize(embed, p=2, dim=1)
        else:
            return embed


class TransformerClassifier(nn.Module):
    """
    Transformer for style classification
    """

    def __init__(self, output_size, input_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=128):
        super().__init__()
        # self.src_embed = nn.Linear(input_size, d_model)
        self.src_embed = Embeddings(d_model, input_size)
        self.argmax = ArgMaxEmbed.apply
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = BasicEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.generator = nn.Linear(d_model, output_size)
        self.masked_mean = MaskedMean()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, argmax=False):
        if argmax:
            src = self.argmax(src, self.src_embed)
        else:
            src = self.src_embed(src)
        src = self.position(src)
        out = self.encoder(src, src_mask)
        out = self.generator(out)
        out = self.masked_mean(out, src_mask)
        return out


class TransformerClassifierAndLM(nn.Module):
    """
    Transformer for style classification
    """

    def __init__(self, output_size, input_size, vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=128):
        super().__init__()
        # self.src_embed = nn.Linear(input_size, d_model)
        self.src_embed = Embeddings(d_model, input_size)
        self.argmax = ArgMaxEmbed.apply
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = BasicEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        # todo maybe add layer norm ?
        self.generator = nn.Linear(d_model, output_size)
        self.masked_mean = MaskedMean()

        self.vocab_generator = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask, argmax=False):
        if argmax:
            src = self.argmax(src, self.src_embed)
        else:
            src = self.src_embed(src)
        src = self.position(src)
        out = self.encoder(src, src_mask)
        out = self.generator(out)
        out = self.masked_mean(out, src_mask)
        return out

    def forward_cls_and_lm(self, src, src_mask, argmax=False):
        if argmax:
            src = self.argmax(src, self.src_embed)
        else:
            src = self.src_embed(src)
        src = self.position(src)
        out = self.encoder(src, src_mask)

        out_1 = self.generator(out)
        out_1 = self.masked_mean(out_1, src_mask)

        out_2 = self.vocab_generator(out) + self.bias

        return out_1, out_2

