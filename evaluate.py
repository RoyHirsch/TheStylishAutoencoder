import numpy as np
import copy

import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn

from params import *
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
            # prepare batch
            src, labels = batch
            src_mask, trg_mask = make_masks(src, src, device)

            src = src.to(device)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)
            labels = labels.to(device)

            # classifier loss
            encode_out = model_enc(src, src_mask)
            cls_preds = model_cls(encode_out)
            cls_loss = cls_criteria(cls_preds[0], labels)
            cls_running_loss += cls_loss.item()

            # rec loss
            preds = model_dec(encode_out, labels, src_mask, src, trg_mask)
            rec_loss = seq2seq_criteria(preds.contiguous().view(-1, preds.size(-1)),
                                        src.contiguous().view(-1))
            rec_running_loss += rec_loss.item()

            # entrophy loss
            cls_preds = model_cls(encode_out)
            ent_loss = ent_criteria(cls_preds[0])
            ent_running_loss += rec_loss.item()
            rec_acc.update(preds, src)
            cls_acc.update(cls_preds[0], labels)

    print("eval-e-{}: loss cls: {:.3f}, loss rec: {:.3f}, loss ent: {:.3f}".format(epoch, cls_running_loss / i,
                                                                                   rec_running_loss / i,
                                                                                   ent_running_loss / i))
    print("eval-e-{}: cls acc: {:.3f}, rec acc: {:.3f}".format(epoch, rec_acc(), cls_acc()))


def greedy_decode(preds, stop_ind=1012):
    '''stop_ind = 1012, stop when hit a period'''
    _, predicted = torch.max(preds.data, 1)
    return crop_sentence(predicted, stop_ind)


def crop_sentence(ids_tensor, stop_ind):
    ''' TODO: roy - not done !
        Gets  an input tensor of word ids with padding,
        drop out the padding and return a list
        0 - padding of 0
        1012 - period '''
    ids_tensor = ids_tensor.detach().cpu().numpy()
    stop_indx = np.min(np.where(ids_tensor == stop_ind)[0])
    return ids_tensor[:stop_indx]


def test_random_samples(data_iter, model_enc, model_dec, model_cls, device, decode_func=None, num_samples=1,
                        transfer_style=True):
    ''' Print some sample text to validate the model.
        transfer_style - bool, if True apply style transfer '''
    conv_id_tokens = data_iter.dataset.tokenizer.convert_ids_to_tokens
    model_cls.eval()
    model_enc.eval()
    model_dec.eval()

    with torch.no_grad():

        for i, batch in enumerate(data_iter):
            if num_samples <= 0: break
            sample_num = int(np.random.randint(0, len(batch[1]), 1))

            # prepare batch
            src, labels = batch
            src_mask, trg_mask = make_masks(src, src, device)

            src = src.to(device)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)
            labels = labels.to(device)
            true_labels = copy.deepcopy(labels)

            # logical not on labels if transfer_style is set
            if transfer_style:
                labels = ~labels.byte()
                labels = labels.long()

            encode_out = model_enc(src, src_mask)
            preds = model_dec(encode_out, labels, src_mask, src, trg_mask)
            cls_preds = model_cls(encode_out)

            # get specific sentence
            src_sent = crop_sentence(src[sample_num], stop_ind=0)
            src_label = 'pos' if true_labels[sample_num].detach().item() == 1 else 'neg'
            print('Original: text: {}'.format(" ".join(conv_id_tokens(src_sent))))
            print('Original: class: {}'.format(src_label))

            pred_label = 'pos' if torch.argmax(cls_preds[0][sample_num, :]) == 1 else 'neg'
            if decode_func:
                rec_sent = decode_func(preds[sample_num, :, :])

                print('Predicted: text: {}'.format(" ".join(conv_id_tokens(rec_sent))))
                print('Predicted: class: {}'.format(pred_label))

            else:
                print('Predicted: class: {}'.format(pred_label))
            print('\n')

            num_samples -= 1
