import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
import logging
import os

from data import make_masks
from transformer_model import *
from classifier_model import *
from utils import AccuracyCls, AccuracyRec, Loss


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


class ResourcesManager:
    def __init__(self, model_enc, model_dec, model_cls, params,
                 enc_name="enc", dec_name="dec", cls_name="cls"):
        self.models = {
            "enc": model_enc.to(params.device),
            "dec": model_dec.to(params.device),
            "cls": model_cls.to(params.device)
        }

        self.exp_name = params.EXP_NAME
        self.suffix = '.pth'
        self.path_ends = {
            "enc": enc_name,
            "dec": dec_name,
            "cls": cls_name
        }

        exp_folder = os.path.join(params.MODELS_PATH, params.EXP_NAME)

        self.save_path = exp_folder
        self.save_paths = self.get_paths_dict(exp_folder)
        self.load_path = params.MODELS_LOAD_PATH
        if self.load_path:
            self.load_models(self.load_path)

    def get_paths_dict(self, basic_path):
        return {
            key: os.path.join(basic_path, val + self.suffix) for (key, val) in self.path_ends.items()
        }

    def load_models(self, load_path=None):
        if load_path is None:
            load_path = self.load_path
            if load_path is None:
                raise ValueError("load_models: load_path is null")
        load_paths = self.get_paths_dict(load_path)
        for (key, path) in load_paths.items():
            try:
                self.models[key].load_state_dict(torch.load(path))
            except IOError:
                logging.info("{}'s state not found in {}".format(key, path))

    def save_model(self, key, save_path=None, verbose=False):
        try:
            model = self.models[key]
        except KeyError:
            raise ValueError("key must be enc, dec or cls")
        if save_path is None:
            save_path = self.save_paths[key]
        try:
            torch.save(model.state_dict(), save_path)
            if verbose:
                logging.info("Saved {} model to {}".format(key, save_path))
        except IOError:
            raise ValueError("couldn't save {} model's state, {} doesn't exist".format(key, save_path))

    def get_models(self):
        return self.models

    def save_models_on_epoch_end(self, epoch):
        epoch_str = '_e_{}'.format(epoch)
        for key, val in self.path_ends.items():
            self.save_model(key, save_path=os.path.join(self.save_path, self.exp_name + epoch_str + val + self.suffix),
                            verbose=False)

    def save_models(self, epoch):
        for key, val in self.path_ends.items():
            self.save_model(key, save_path=os.path.join(self.save_path, self.exp_name + "_" + val + self.suffix),
                            verbose=False)


"""
Optimizers
"""


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        b = x * x.log()
        b = b.sum(dim=1)
        b = b.mean()
        return b


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx=1, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Seq2SeqSimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        return loss.item()


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
Training Functions
"""


def init_models(vocab_size, params):
    model_trans = StyleTransformer(src_vocab=vocab_size, tgt_vocab=params.H_DIM,
                                   N=params.N_LAYERS, d_model=params.H_DIM, d_ff=params.FC_DIM,
                                   h=params.N_ATTN_HEAD, n_styles=params.N_STYLES, dropout=params.DO_RATE)
    if params.TRANS_CLS:
        model_cls = TransformerClassifier(output_size=params.N_STYLES, N=params.N_LAYERS, d_model=params.H_DIM,
                                          d_ff=params.FC_DIM, h=params.N_ATTN_HEAD, dropout=params.DO_RATE_CLS,
                                          input_size=params.H_DIM)
    else:
        model_cls = Descriminator(input_size=vocab_size, output_size=params.N_STYLES, hidden_size=params.H_DIM,
                                  embedding_size=params.H_DIM, drop_rate=params.DO_RATE_CLS, num_layers=params.N_LAYERS,
                                  num_layers_for_output=4)
    return model_trans, model_cls


def train_cls_step(model_enc, model_cls, cls_opt, cls_criteria,
                   src, src_mask, labels, cls_running_loss, cls_acc):
    with torch.no_grad():
        encode_out = model_enc(src, src_mask)
    cls_preds = model_cls(encode_out)

    cls_opt.zero_grad()
    loss = cls_criteria(cls_preds, labels)
    cls_acc.update(cls_preds, labels)
    cls_running_loss.update(loss)
    loss.backward()
    cls_opt.step()
    cls_opt.zero_grad()


def train_transformer_step(model_cls, model_enc, model_dec, seq2seq_criteria,
                           ent_criteria, opt_enc, opt_dec, rec_lambda, ent_lambda,
                           src, src_mask, labels, trg_mask, rec_running_loss,
                           ent_running_loss, rec_acc):
    encode_out = model_enc(src, src_mask)
    preds = model_dec(encode_out, labels, src_mask, src, trg_mask)

    # Ignore the style embedding locations preds.size() = [batch_size, max_len, vocab_size]
    preds = preds[:, 1:, :]
    preds = preds.contiguous().view(-1, preds.size(-1))
    # Ignore the last token src.size() = [batch_size, max_len]
    src = src[:, :-1]
    src = src.contiguous().view(-1)
    rec_loss = seq2seq_criteria(preds, src)
    rec_running_loss.update(rec_loss)
    rec_acc.update(preds, src)

    # optimize decoder
    opt_dec.zero_grad()
    rec_loss.backward(retain_graph=True)
    opt_dec.step()
    opt_dec.zero_grad()

    opt_enc.zero_grad()
    cls_preds = model_cls(encode_out)
    ent_loss = ent_criteria(cls_preds)
    ent_running_loss.update(ent_loss)
    enc_loss = (rec_lambda * rec_loss) + (ent_lambda * ent_loss)

    # optimizer encoder
    enc_loss.backward()
    opt_enc.step()
    opt_enc.zero_grad()


def train_entropy_step(model_cls, model_enc, ent_criteria, opt_enc, opt_cls,
                       src, src_mask, ent_running_loss, ent_lambda=1.0):
    encode_out = model_enc(src, src_mask)
    cls_preds = model_cls(encode_out)
    ent_loss = ent_criteria(cls_preds)
    ent_loss *= ent_lambda
    ent_running_loss.update(ent_loss)
    opt_enc.zero_grad()
    opt_cls.zero_grad()

    # optimizer encoder
    if ent_loss:
        ent_loss.backward()
        opt_enc.step()
        opt_enc.zero_grad()
        opt_cls.zero_grad()


def train_rec_step(model_enc, model_dec, seq2seq_criteria,
                   opt_enc, opt_dec,
                   src, src_mask, labels, trg_mask, rec_running_loss,
                   rec_acc, rec_lambda=0.0):
    encode_out = model_enc(src, src_mask)
    preds = model_dec(encode_out, labels, src_mask, src, trg_mask)

    # Ignore the style embedding locations preds.size() = [batch_size, max_len, vocab_size]
    preds = preds[:, 1:, :]
    preds = preds.contiguous().view(-1, preds.size(-1))
    # Ignore the last token src.size() = [batch_size, max_len]
    src = src[:, :-1]
    src = src.contiguous().view(-1)
    rec_loss = seq2seq_criteria(preds, src)
    rec_running_loss.update(rec_loss)
    rec_acc.update(preds, src)

    # optimize decoder
    opt_dec.zero_grad()
    rec_loss.backward(retain_graph=True)
    opt_dec.step()
    opt_dec.zero_grad()
    opt_enc.zero_grad()

    enc_loss = rec_lambda * rec_loss
    # optimizer encoder
    if enc_loss:
        enc_loss.backward()
        opt_enc.step()
        opt_enc.zero_grad()


def get_train_steps_from_params(train_set_size, train_batch_size,
                                period_to_epoch_ratio, trans_steps_ratio):
    steps_per_epoch = train_set_size // train_batch_size
    period_size = int(steps_per_epoch * period_to_epoch_ratio)
    trans_period_steps = int(period_size * trans_steps_ratio)
    cls_period_steps = period_size - trans_period_steps

    logging.info("steps_per_epoch {}, period is {} steps: {} trans, {} cls".format(steps_per_epoch,
                                                                                   period_size,
                                                                                   trans_period_steps,
                                                                                   cls_period_steps))
    return trans_period_steps, cls_period_steps


def get_warmup_steps_from_params(train_set_size, train_batch_size, n_epochs,
                                 enc_ratio, dec_ratio, cls_ratio):
    steps_per_epoch = train_set_size // train_batch_size
    n_total_steps = n_epochs * steps_per_epoch
    warmup_dec_steps = n_total_steps * dec_ratio
    warmup_cls_steps = n_total_steps * cls_ratio

    logging.info("total_steps {}, dec_warmup {}, cls_warmup {}".format(n_total_steps, warmup_dec_steps,
                                                                       warmup_cls_steps))
    return warmup_dec_steps, warmup_cls_steps


def run_epoch(epoch, data_iter, model_enc, opt_enc, model_dec, opt_dec,
              model_cls, opt_cls, cls_criteria, seq2seq_criteria,
              ent_criteria, params):
    trans_steps, cls_steps = get_train_steps_from_params(len(data_iter.dataset),
                                                         params.TRAIN_BATCH_SIZE,
                                                         params.PERIOD_EPOCH_RATIO,
                                                         params.TRANS_STEPS_RATIO)
    rec_lambda = params.REC_LAMBDA
    ent_lambda = params.ENT_LAMBDA
    verbose = params.VERBOSE
    device = params.device

    cls_running_loss = Loss()
    rec_running_loss = Loss()
    ent_running_loss = Loss()

    rec_acc = AccuracyRec()
    cls_acc = AccuracyCls()

    model_cls.train()
    model_dec.train()

    for step, batch in enumerate(data_iter):
        # prepare batch
        i = step % (trans_steps + cls_steps)
        src, labels = batch.text, batch.label
        src_mask, trg_mask = make_masks(src, src, device)

        src = src.to(device)
        src_mask = src_mask.to(device)
        trg_mask = trg_mask.to(device)
        labels = labels.to(device)

        if i < trans_steps:  # training the transformer
            if i == 0:  # switch from cls_train setting to trans_train setting
                logging.debug("Training TRANS")
                model_enc.train()

            train_transformer_step(model_cls=model_cls, model_enc=model_enc,
                                   model_dec=model_dec, seq2seq_criteria=seq2seq_criteria,
                                   ent_criteria=ent_criteria, opt_enc=opt_enc,
                                   opt_dec=opt_dec, rec_lambda=rec_lambda, ent_lambda=ent_lambda,
                                   src=src, src_mask=src_mask, labels=labels, trg_mask=trg_mask,
                                   ent_running_loss=ent_running_loss,
                                   rec_running_loss=rec_running_loss, rec_acc=rec_acc)
            if i == trans_steps - 1:
                loss = rec_lambda * rec_running_loss() + (1 - rec_lambda) * ent_running_loss()
                if verbose:
                    logging.info(
                        "e-{},s-{}: Training transformer ent_loss {}, rec_loss {}, loss {}, rec_acc {}".format(epoch,
                                                                                                               step,
                                                                                                               ent_running_loss(),
                                                                                                               rec_running_loss(),
                                                                                                               loss,
                                                                                                               rec_acc()))
                ent_running_loss.reset()
                rec_running_loss.reset()
                rec_acc.reset()

        else:  # training the classifier
            if i == trans_steps:  # switch from trans_train setting to cls_train setting
                logging.debug("Training CLS")
                model_enc.eval()

            train_cls_step(model_enc=model_enc, model_cls=model_cls, cls_opt=opt_cls,
                           cls_criteria=cls_criteria, src=src, src_mask=src_mask,
                           labels=labels, cls_acc=cls_acc, cls_running_loss=cls_running_loss)

            if i == trans_steps + cls_steps - 1:
                if verbose:
                    logging.info(
                        "e-{},s-{}: Training cls loss {} acc {}".format(epoch, step, cls_running_loss(),
                                                                        cls_acc()))
                cls_running_loss.reset()
                cls_acc.reset()


def run_epoch_rec_ent(epoch, data_iter, model_enc, opt_enc, model_dec, opt_dec,
                      model_cls, opt_cls, cls_criteria, seq2seq_criteria,
                      ent_criteria, params):
    trans_steps, cls_steps = get_train_steps_from_params(len(data_iter.dataset),
                                                         params.TRAIN_BATCH_SIZE,
                                                         params.PERIOD_EPOCH_RATIO,
                                                         params.TRANS_STEPS_RATIO)
    rec_lambda = params.REC_LAMBDA
    verbose = params.VERBOSE
    device = params.device

    cls_running_loss = Loss()
    rec_running_loss = Loss()
    ent_running_loss = Loss()

    rec_acc = AccuracyRec()
    cls_acc = AccuracyCls()

    rec_steps = int(trans_steps * params.REC_STEPS_RATIO)

    model_cls.train()
    model_dec.train()

    for step, batch in enumerate(data_iter):
        # prepare batch
        i = step % (trans_steps + cls_steps)
        src, labels = batch.text, batch.label
        src_mask, trg_mask = make_masks(src, src, device)

        src = src.to(device)
        src_mask = src_mask.to(device)
        trg_mask = trg_mask.to(device)
        labels = labels.to(device)

        if i < rec_steps:  # training the rec
            if i == 0:  # switch from cls_train setting to trans_train setting
                model_enc.train()
            train_rec_step(model_enc, model_dec, seq2seq_criteria, opt_enc, opt_dec, src,
                           src_mask, labels, trg_mask, rec_running_loss, rec_acc, rec_lambda)
            if i == rec_steps - 1:
                enc_loss = rec_lambda * rec_running_loss()
                if verbose:
                    logging.info(
                        "e-{},s-{}: Training transformer on rec, rec_loss {}, enc_loss {}, rec_acc {}".format(epoch,
                                                                                                              step,
                                                                                                              rec_running_loss(),
                                                                                                              enc_loss,
                                                                                                              rec_acc()))
                rec_running_loss.reset()
                rec_acc.reset()
        elif rec_steps <= i < trans_steps:
            train_entropy_step(model_cls, model_enc, ent_criteria, opt_enc, opt_cls, src,
                               src_mask, ent_running_loss, ent_lambda=params.ENT_LAMBDA)
            if i == trans_steps - 1:
                if verbose:
                    logging.info(
                        "e-{},s-{}: Training encoder ent_loss {}".format(epoch, step,
                                                                         ent_running_loss()))
                    ent_running_loss.reset()

        else:  # training the classifier
            if i == trans_steps:  # switch from trans_train setting to cls_train setting
                model_enc.eval()

            train_cls_step(model_enc=model_enc, model_cls=model_cls, cls_opt=opt_cls,
                           cls_criteria=cls_criteria, src=src, src_mask=src_mask,
                           labels=labels, cls_acc=cls_acc, cls_running_loss=cls_running_loss)

            if i == trans_steps + cls_steps - 1:
                if verbose:
                    logging.info(
                        "e-{},s-{}: Training cls loss {} acc {}".format(epoch, step, cls_running_loss(),
                                                                        cls_acc()))
                    cls_running_loss.reset()
                    cls_acc.reset()


def train_true_label_step(model_dec, seq2seq_criteria, model_cls, opt_dec, opt_cls,
                          cls_criteria,
                          src, src_mask, labels, cls_running_loss, cls_acc,
                          rec_running_loss, trans_cls=False,
                          rec_lambda=1.0, cls_lambda=0.0, train_cls=False):
    model_dec.train()
    # style classifier loss
    preds = model_dec(src, src_mask, labels)
    if cls_lambda or train_cls:
        if trans_cls:
            cls_preds = model_cls(preds, src_mask)
        else:
            cls_preds = model_cls(preds)
        cls_loss = cls_criteria(cls_preds, labels)
        cls_acc.update(cls_preds, labels)
        cls_running_loss.update(cls_loss)

    # optimize cls
    if train_cls:
        opt_cls.zero_grad()
        cls_loss.backward(retain_graph=True)
        opt_cls.step()

    src_embeds = model_dec.src_embed(src)
    rec_loss = seq2seq_criteria(src_embeds, preds, src)
    rec_running_loss.update(rec_loss)
    # rec_acc.update(preds_cont, src_cont)

    # optimize decoder
    loss = rec_lambda * rec_loss
    if cls_lambda:
        loss += cls_lambda * cls_loss
    opt_dec.zero_grad()
    loss.backward()
    opt_dec.step()


def train_neg_label_step(model_dec, seq2seq_criteria, model_cls, cls_criteria,
                         opt_dec, src, src_mask, labels, rec_running_loss, cls_running_loss,
                         cls_acc, trans_cls=False, rec_lambda=0.0, cls_lambda=1.0):
    model_dec.train()
    # Negate labels
    labels = ~labels.byte()
    labels = labels.long()

    preds = model_dec(src, src_mask, labels)
    if trans_cls:
        cls_preds = model_cls(preds, src_mask)
    else:
        cls_preds = model_cls(preds)

    opt_dec.zero_grad()
    cls_loss = cls_criteria(cls_preds, labels)
    cls_acc.update(cls_preds, labels)
    cls_running_loss.update(cls_loss)
    src_embeds = model_dec.src_embed(src)
    rec_loss = seq2seq_criteria(src_embeds, preds, src)
    rec_running_loss.update(rec_loss)
    loss = cls_lambda * cls_loss + rec_lambda * rec_loss
    loss.backward()
    opt_dec.step()


def train_true_neg_cls_step(model_dec, model_cls, cls_criteria,
                            opt_cls, src, src_mask, labels, cls_running_loss,
                            cls_acc, trans_cls=False):
    model_dec.eval()
    # style classifier loss
    with torch.no_grad():
        preds = model_dec(src, src_mask, labels)
    if trans_cls:
        cls_preds = model_cls(preds, src_mask)
    else:
        cls_preds = model_cls(preds)

    opt_cls.zero_grad()
    cls_loss = cls_criteria(cls_preds, labels)
    cls_acc.update(cls_preds, labels)
    cls_running_loss.update(cls_loss)
    cls_loss.backward()
    opt_cls.step()


def run_epoch_true_neg(epoch, data_iter, model_dec, opt_dec,
                       model_cls, opt_cls, cls_criteria, seq2seq_criteria,
                       params):
    verbose = params.VERBOSE
    device = params.device
    total_steps = len(data_iter.dataset) // params.TRAIN_BATCH_SIZE
    period_steps = int(params.PERIOD_EPOCH_RATIO * total_steps)
    logging.info('total epoch steps {}, period size {}'.format(total_steps,
                                                               period_steps))

    cls_running_loss = Loss()
    rec_running_loss = Loss()

    cls_acc = AccuracyCls()

    model_cls.train()
    model_dec.train()
    curr_step = 0
    curr_phase = 0
    for step, batch in enumerate(data_iter):
        # prepare batch
        src, labels = batch.text, batch.label
        src_mask, _ = make_masks(src, src, device)

        src = src.to(device)
        src_mask = src_mask.to(device)
        labels = labels.to(device)

        if curr_phase == 0:  # training the classifier
            train_true_neg_cls_step(model_dec, model_cls, cls_criteria,
                                    opt_cls, src, src_mask, labels, cls_running_loss,
                                    cls_acc, trans_cls=params.TRANS_CLS)
            curr_step += 1
            if curr_step == period_steps:
                if verbose:
                    logging.info(
                        "e-{},s-{}: Trained cls loss {} acc {}".format(epoch, step, cls_running_loss(),
                                                                       cls_acc()))
                cls_running_loss.reset()
                curr_cls_acc = cls_acc()
                cls_acc.reset()
                curr_step = 0
                if curr_cls_acc >= params.CLS_ACC_BAR:
                    curr_phase = curr_phase + 1

        elif curr_phase == 1:
            train_true_label_step(model_dec, seq2seq_criteria, model_cls, opt_dec, opt_cls,
                                  cls_criteria, src, src_mask, labels, cls_running_loss, cls_acc,
                                  rec_running_loss, rec_lambda=params.TRUE_REC_LAMBDA,
                                  cls_lambda=params.TRUE_CLS_LAMBDA,
                                  trans_cls=params.TRANS_CLS)
            curr_step += 1
            if curr_step == period_steps:
                if verbose:
                    logging.info(
                        "e-{},s-{}: Trained transformer on true label, rec_loss {}".format(epoch,
                                                                                           step,
                                                                                           rec_running_loss()))
                curr_rec_loss = rec_running_loss()
                rec_running_loss.reset()
                curr_step = 0
                if curr_rec_loss < params.REC_LOSS_BAR:
                    curr_phase = curr_phase + 1

        else:
            train_neg_label_step(model_dec, seq2seq_criteria, model_cls, cls_criteria,
                                 opt_dec, src, src_mask, labels, rec_running_loss, cls_running_loss,
                                 cls_acc, trans_cls=params.TRANS_CLS,
                                 rec_lambda=params.NEG_REC_LAMBDA, cls_lambda=params.NEG_CLS_LAMBDA)
            curr_step += 1
            if curr_step == period_steps:
                if verbose:
                    logging.info(
                        "e-{},s-{}: Trained transformer on negated label, cls_loss {}, cls_acc {}, rec_loss {}".format(
                            epoch,
                            step,
                            cls_running_loss(),
                            cls_acc(),
                            rec_running_loss()))
                cls_running_loss.reset()
                cls_acc.reset()
                curr_step = 0
                curr_phase = 0
