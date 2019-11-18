import logging
import torch
import torch.nn as nn

from data import make_masks
from transformer_model import StyleTransformer
from classifier_model import TransformerClassifier
from utils import AccuracyCls, AccuracyRec, Loss

"""
Optimizers
"""


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model, h_dim, lr, warmup, eps=1e-9, factor=2, betas=(0.9, 0.98)):
    return NoamOpt(h_dim, factor, warmup,
                   torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps))


"""
Losses
"""


class MaskedCosineEmbeddingLoss(nn.Module):
    """
    calculates mean of cosine embedding loss between masked Tensors
    """

    def __init__(self, device, pad=1):
        super().__init__()
        self._loss = nn.CosineEmbeddingLoss()
        self._pad = pad
        self._device = device

    def calc_sample_loss(self, src_embeds, preds, src):
        pad_idx = (src == self._pad).nonzero()
        if pad_idx.shape[0]:
            pad_idx = pad_idx[0]
            src_embeds = src_embeds[:pad_idx, :]
            preds = preds[:pad_idx, :]
        target = torch.ones(preds.shape[0]).to(self._device)
        return self._loss(preds, src_embeds, target)

    def forward(self, src_embeds, preds, src):
        total_loss = 0.0
        n_samples = src.shape[0]
        for i in range(n_samples):
            total_loss += self.calc_sample_loss(src_embeds[i, ...],
                                                preds[i, ...],
                                                src[i, ...])
        return total_loss / n_samples


"""
Init Functions
"""


def load_pretrained_embedding_to_encoder(src_embed, embedding):
    ''' Helper function to modify encoder model embedding with pre-trained
        embedding like Glove. '''
    src_embed.lut.weight.data.copy_(embedding)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_warmup_steps_from_params(train_set_size, train_batch_size, n_epochs,
                                 dec_ratio, cls_ratio):
    steps_per_epoch = train_set_size // train_batch_size
    n_total_steps = n_epochs * steps_per_epoch
    warmup_gen_steps = n_total_steps * dec_ratio
    warmup_cls_steps = n_total_steps * cls_ratio

    logging.info("total_steps {}, gen_warmup {}, cls_warmup {}".format(n_total_steps, warmup_gen_steps,
                                                                       warmup_cls_steps))
    return warmup_gen_steps, warmup_cls_steps


def init_models(vocab_size, params, word_embeddings=None):
    model_gen = StyleTransformer(src_vocab=vocab_size, tgt_vocab=vocab_size,
                                 N=params.N_LAYERS, d_model=params.H_DIM, d_ff=params.FC_DIM,
                                 h=params.N_ATTN_HEAD, n_styles=params.N_STYLES, dropout=params.DO_RATE,
                                 max_len=params.MAX_LEN)
    model_cls = TransformerClassifier(output_size=params.N_STYLES, N=params.N_LAYERS_CLS, d_model=params.H_DIM,
                                      d_ff=params.FC_DIM, h=params.N_ATTN_HEAD, dropout=params.DO_RATE_CLS,
                                      input_size=vocab_size, max_len=params.MAX_LEN)

    if word_embeddings is not None:
        load_pretrained_embedding_to_encoder(model_gen.src_embed, word_embeddings)
        load_pretrained_embedding_to_encoder(model_cls.src_embed, word_embeddings)

    model_gen = model_gen.to(params.device)
    model_cls = model_cls.to(params.device)

    logging.info(f'model_cls has {count_parameters(model_cls):,} trainable parameters')
    logging.info(f'model_gen has {count_parameters(model_gen):,} trainable parameters')
    return model_gen, model_cls


def init_optimizers(model_gen, model_cls, train_iter_size, params):
    if params.WARMUP_STEPS > 0:
        gen_warmup = cls_warmup = params.WARMUP_STEPS
    else:
        gen_warmup, cls_warmup = get_warmup_steps_from_params(train_iter_size,
                                                              params.TRAIN_BATCH_SIZE,
                                                              params.N_EPOCHS,
                                                              params.GEN_WARMUP_RATIO,
                                                              params.CLS_WARMUP_RATIO)

    opt_gen = get_std_opt(model_gen, h_dim=params.H_DIM, lr=params.GEN_LR, warmup=gen_warmup, factor=params.GEN_FACTOR)
    opt_cls = get_std_opt(model_cls, h_dim=params.H_DIM, lr=params.CLS_LR, warmup=cls_warmup, factor=params.CLS_FACTOR)

    return opt_gen, opt_cls


"""
Training Functions
"""


def train_cls_step(model_cls, cls_criteria,
                   opt_cls, src, src_mask, labels, cls_running_loss,
                   cls_acc, trans_cls=False):
    # classifier loss
    if trans_cls:
        cls_preds = model_cls(src, src_mask, argmax=False)
    else:
        cls_preds = model_cls(src)

    opt_cls.zero_grad()
    cls_loss = cls_criteria(cls_preds, labels)
    cls_acc.update(cls_preds, labels)
    cls_running_loss.update(cls_loss)
    cls_loss.backward()
    opt_cls.step()


def train_gen_step(model_gen, seq2seq_criteria, model_cls, cls_criteria,
                   opt_gen, src, src_mask, labels, bt_running_loss, bt_acc, style_running_loss,
                   style_acc, trans_cls=False, bt_lambda=1.0, style_lambda=1.0):
    model_gen.train()

    # Negate labels for style transfer
    target_labels = (~labels.bool()).long()
    target_preds = model_gen(src, src_mask, target_labels, argmax=False)
    if trans_cls:
        style_preds = model_cls(target_preds, src_mask, argmax=True)
    else:
        style_preds = model_cls(target_preds)

    bt_preds = model_gen(target_preds, src_mask, labels, argmax=True)
    opt_gen.zero_grad()

    # classifier
    style_loss = cls_criteria(style_preds, target_labels)
    style_acc.update(style_preds, target_labels)
    style_running_loss.update(style_loss)

    # bt loss
    bt_preds = bt_preds.contiguous().view(-1, bt_preds.size(-1))
    src = src.contiguous().view(-1)
    bt_loss = seq2seq_criteria(bt_preds, src)
    bt_running_loss.update(bt_loss)
    bt_acc.update(bt_preds, src)

    # optimize
    loss = bt_lambda * bt_loss + style_lambda * style_loss
    loss.backward()
    opt_gen.step()


"""
Training loops
"""


def run_train_epoch(epoch, data_iter, model_gen, opt_gen,
                    model_cls, cls_criteria, seq2seq_criteria,
                    params):
    verbose = params.VERBOSE
    device = params.device
    total_steps = len(data_iter.dataset) // params.TRAIN_BATCH_SIZE
    period_steps = params.PERIOD_STEPS
    logging.info('total epoch steps {}, period size {}'.format(total_steps,
                                                               period_steps))

    style_running_loss = Loss()
    bt_running_loss = Loss()

    style_acc = AccuracyCls()
    bt_acc = AccuracyRec()

    model_cls.train()
    model_gen.train()
    curr_step = 0
    for step, batch in enumerate(data_iter):
        # prepare batch
        src, labels = batch.text, batch.label

        src_mask, _ = make_masks(src, src, device)

        src = src.to(device)
        src_mask = src_mask.to(device)
        labels = labels.to(device)

        train_gen_step(model_gen=model_gen, seq2seq_criteria=seq2seq_criteria, model_cls=model_cls,
                       cls_criteria=cls_criteria,
                       opt_gen=opt_gen, src=src, src_mask=src_mask, labels=labels, bt_running_loss=bt_running_loss,
                       bt_acc=bt_acc, style_running_loss=style_running_loss,
                       style_acc=style_acc, trans_cls=params.TRANS_CLS, bt_lambda=params.BT_LAMBDA,
                       style_lambda=params.STYLE_LAMBDA)

        curr_step += 1
        if curr_step == period_steps:
            if verbose:
                logging.info(
                    "e-{},s-{}: Training transformer on back-translation loss: style_loss {:.3f}, style_acc {:.3f},"
                    " bt_loss {:.3f}, bt_acc {:.3f}".format(
                        epoch,
                        step,
                        style_running_loss(),
                        style_acc(),
                        bt_running_loss(),
                        bt_acc()))
            curr_step = 0
            style_acc.reset()
            bt_running_loss.reset()
            style_running_loss.reset()
            bt_acc.reset()


def train_gen_on_rec_loss(train_iter, model_gen, opt_gen, seq2seq_criteria, steps, params):
    verbose = params.VERBOSE
    device = params.device

    rec_running_loss = Loss()
    rec_acc = AccuracyRec()

    model_gen.train()
    for step, batch in enumerate(train_iter):
        # prepare batch
        src, labels = batch.text, batch.label
        src_mask, _ = make_masks(src, src, device)

        src = src.to(device)
        src_mask = src_mask.to(device)
        labels = labels.to(device)
        preds = model_gen(src, src_mask, labels, argmax=False)
        preds = preds.contiguous().view(-1, preds.size(-1))
        src = src.contiguous().view(-1)
        rec_loss = seq2seq_criteria(preds, src)
        rec_running_loss.update(rec_loss)
        rec_acc.update(preds, src)

        # optimize decoder
        loss = rec_loss
        opt_gen.zero_grad()
        loss.backward()
        opt_gen.step()

        if verbose and step % steps == steps - 1:
            logging.info(
                "s-{}: Training transformer on rec loss, rec_loss {}, rec_acc {}".format(step,
                                                                                              rec_running_loss(),
                                                                                              rec_acc()))
            break
            rec_running_loss.reset()
            rec_acc.reset()


def train_cls(train_iter, model_cls, opt_cls, cls_criteria, params, epochs=1):
    for epoch in range(epochs):
        verbose = params.VERBOSE
        device = params.device

        cls_running_loss = Loss()
        cls_acc = AccuracyCls()

        model_cls.train()
        for step, batch in enumerate(train_iter):
            # prepare batch
            src, labels = batch.text, batch.label
            src_mask, _ = make_masks(src, src, device)

            src = src.to(device)
            src_mask = src_mask.to(device)
            labels = labels.to(device)
            train_cls_step(model_cls=model_cls, cls_criteria=cls_criteria, opt_cls=opt_cls, src=src, src_mask=src_mask,
                           labels=labels, cls_running_loss=cls_running_loss, cls_acc=cls_acc,
                           trans_cls=params.TRANS_CLS)

            if verbose and step % 100 == 99:
                logging.info(
                    "e-{},s-{}: Training cls loss {} acc {}".format(epoch, step, cls_running_loss(),
                                                                    cls_acc()))
                cls_running_loss.reset()
                cls_acc.reset()
