import numpy as np
import copy
import logging

import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn

from data import make_masks

class AccuracyRec(object):

    def __init__(self, pad_ind=0):
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
        return self.correct / self.total * 100.0


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
        return self.correct / self.total * 100.0


def evaluate(epoch, data_iter, model_enc, model_dec,
             model_cls, cls_criteria, seq2seq_criteria,
             ent_criteria, device):
    ''' Evaluate performances over test/validation dataloader '''

    model_cls.eval()
    model_enc.eval()
    model_dec.eval()

    cls_running_loss = 0.0
    rec_running_loss = 0.0
    ent_running_loss = 0.0

    rec_acc = AccuracyRec()
    cls_acc = AccuracyCls()
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            # Prepare batch
            src, labels = batch.text, batch.label
            src_mask, trg_mask = make_masks(src, src, device)

            src = src.to(device)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)
            labels = labels.to(device)

            # Classifier loss
            encode_out = model_enc(src, src_mask)
            cls_preds = model_cls(encode_out)
            cls_loss = cls_criteria(cls_preds, labels)
            cls_running_loss += cls_loss.item()

            # Rec loss
            preds = model_dec(encode_out, labels, src_mask, src, trg_mask)
            rec_loss = seq2seq_criteria(preds.contiguous().view(-1, preds.size(-1)),
                                        src.contiguous().view(-1))
            rec_running_loss += rec_loss.item()

            # Entropy loss
            cls_preds = model_cls(encode_out)
            ent_loss = ent_criteria(cls_preds)
            ent_running_loss += ent_loss.item()

            # Accuracy
            rec_acc.update(preds, src)
            cls_acc.update(cls_preds, labels)

    logging.info("eval-e-{}: loss cls: {:.3f}, loss rec: {:.3f}, loss ent: {:.3f}".format(epoch, cls_running_loss / i,
                                                                                   rec_running_loss / i,
                                                                                   ent_running_loss / i))
    logging.info("eval-e-{}: acc cls: {:.3f}, acc rec: {:.3f}".format(epoch, rec_acc(), cls_acc()))
    # TODO - Roy - what metric to report ?
    return rec_acc

def greedy_decode_sent(preds, id2word, eos_id):
    ''' Nauve greedy decoding - just argmax over the vocabulary distribution '''
    preds = preds.squeeze(0).detach().cpu().numpy()
    preds = np.argmax(preds, -1)
    decoded_sent = sent2str(preds, id2word, eos_id)
    return decoded_sent

def sent2str(sent_as_np, id2word, eos_id=None):
    ''' Gets sentence as a list of ids and transfers to string
        Input is np array of ids '''
    if not(isinstance(sent_as_np, np.ndarray)):
        raise ValueError('Invalid input type, expected np array')
    if eos_id:
        end_id = np.where(sent_as_np == eos_id)[0]
        if end_id.size != 0:
            sent_as_np = sent_as_np[:int(end_id)]
    return " ".join([id2word[i] for i in sent_as_np])

def test_random_samples(data_iter, TEXT, model_enc, model_dec, model_cls, device, decode_func=None, num_samples=2,
                        transfer_style=True):

    ''' Print some sample text to validate the model.
        transfer_style - bool, if True apply style transfer '''

    word2id = TEXT.vocab.stoi
    eos_id = int(word2id['<eos>'])
    id2word = {v: k for k, v in word2id.items()}
    model_enc.eval()
    model_dec.eval()

    with torch.no_grad():

        for step, batch in enumerate(data_iter):
            if num_samples == 0: break

            # Prepare batch
            sample_num = int(np.random.randint(0, len(batch), 1))
            src, labels = batch.text[sample_num], batch.label[sample_num]
            src = src.unsqueeze(0)
            src_mask, trg_mask = make_masks(src, src, device)

            src = src.to(device)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)
            labels = labels.to(device)
            true_labels = copy.deepcopy(labels)

            # Logical not on labels if transfer_style is set
            if transfer_style:
                labels = ~labels.byte()
                labels = labels.long()

            encode_out = model_enc(src, src_mask)
            preds = model_dec(encode_out, labels, src_mask, src, trg_mask)
            cls_preds = model_cls(encode_out)

            sent_as_list = src.squeeze(0).detach().cpu().numpy()
            src_sent = sent2str(sent_as_list, id2word, eos_id)
            src_label = 'pos' if true_labels.detach().item() == 1 else 'neg'
            logging.info('Original: text: {}'.format(src_sent))
            logging.info('Original: class: {}'.format(src_label))

            pred_label = 'pos' if torch.argmax(cls_preds) == 1 else 'neg'
            if decode_func:
                dec_sent = decode_func(preds, id2word, eos_id)
                if transfer_style:
                    logging.info('Style transfer output:')
                logging.info('Predicted: text: {}'.format(dec_sent))
                logging.info('Predicted: class: {}'.format(pred_label))

            else:
                logging.info('Predicted: class: {}'.format(pred_label))
            logging.info('\n')

            num_samples -= 1
