from data import *
from train import *
from evaluate import *
from utils import *


class Params(object):
    # Loggin
    # Free text to describe the experiment
    COMMENT = ''
    VERBOSE = True
    EXP_NAME = "gal_exp"
    DATA_PATH = "/content/drive/My Drive/NLP/"
    MODELS_PATH = "/content/drive/My Drive/NLP/final/"
    PRINT_INTERVAL = 10

    # TODO: for local use
    # DATA_PATH = MODELS_PATH = os.path.abspath(__file__+'/../')
    MODELS_LOAD_PATH = "/content/drive/My Drive/NLP/final/gal_exp"

    # Data
    DATASET_NAME = 'IMDB'
    # Maximal number of batches for test model
    TEST_MAX_BATCH_SIZE = 300
    # Min freq for word in dataset to include in vocab
    VOCAB_MIN_FREQ = 5
    # Whether to use Glove embadding - if TRUE set H_DIM to 300
    VOCAB_USE_GLOVE = True
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    # maximum length of allowed sentence - can be also None
    MAX_LEN = 128

    # Transformer model
    N_LAYERS = 12
    H_DIM = 300
    N_ATTN_HEAD = 6
    FC_DIM = 2048
    DO_RATE = 0.01

    # Classification model
    N_STYLES = 2
    DO_RATE_CLS = 0.6
    TRANS_CLS = True
    CLS_ACC_BAR = 80.0

    # Train
    N_EPOCHS = 20
    PATIENCE = 3
    ENC_LR = 0
    DEC_LR = 0
    CLS_LR = 5e-4
    TRANS_STEPS_RATIO = 0.1
    TRUE_STEPS_RATIO = 0.9
    PERIOD_EPOCH_RATIO = 1e-2
    ENC_WARMUP_RATIO = 0.2
    DEC_WARMUP_RATIO = 0.2
    CLS_WARMUP_RATIO = 0.2
    TRUE_REC_LAMBDA = 1e-2
    TRUE_CLS_LAMBDA = 0.0
    NEG_REC_LAMBDA = 1.0
    NEG_CLS_LAMBDA = 1.0
    ENT_LAMBDA = 0.0
    TRAIN_ON_CLS_LOSS = False
    REC_LOSS_BAR = 1e-1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


params = Params()
logger = create_logger(params)
pprint_params(params)

TEXT, word_embeddings, train_iter, test_iter = load_dataset_from_csv(params=params, device=params.device)
logging.info('Train dataset len: {} Test dataset len: {}'.format(len(train_iter.dataset), len(test_iter.dataset)))

# Clear CUDA memory if needed
# TODO: local use
# torch.cuda.empty_cache()

"""
Init models
"""


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


vocab_size = len(TEXT.vocab)
model_dec, model_cls = init_models(vocab_size, params)
if params.H_DIM == 300:
    model_dec = load_pretrained_embedding_to_encoder(model_dec, word_embeddings)

print(f'model_cls has {count_parameters(model_cls):,} trainable parameters')
print(f'model_dec has {count_parameters(model_dec):,} trainable parameters')

model_dec = model_dec.to(params.device)
model_cls = model_cls.to(params.device)

### Init losses ###
cls_criteria = nn.CrossEntropyLoss()
cls_criteria = cls_criteria.to(params.device)

# seq2seq_criteria = LabelSmoothing(size=vocab_size, padding_idx=1)
seq2seq_criteria = MaskedCosineEmbeddingLoss(params.device)
seq2seq_criteria = seq2seq_criteria.to(params.device)

ent_criteria = EntropyLoss()
ent_criteria = ent_criteria.to(params.device)

### Init optimizers ###
dec_warmup, cls_warmup = get_warmup_steps_from_params(len(train_iter.dataset),
                                                      params.TRAIN_BATCH_SIZE,
                                                      params.N_EPOCHS,
                                                      params.ENC_WARMUP_RATIO,
                                                      params.DEC_WARMUP_RATIO,
                                                      params.CLS_WARMUP_RATIO)

# cls_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model_cls.parameters()),
#                            lr=params.LR_CLS, weight_decay=params.WD_CLS)
# enc_opt = get_std_opt(model_enc, h_dim=params.H_DIM, lr=params.LR_ENC, warmup=enc_warmup)
# dec_opt = get_std_opt(model_dec, h_dim=params.H_DIM, lr=params.LR_DEC, warmup=dec_warmup)
opt_cls = get_std_opt(model_cls, h_dim=params.H_DIM, lr=params.CLS_LR, warmup=cls_warmup)

opt_dec = get_std_opt(model_dec, h_dim=params.H_DIM, lr=params.DEC_LR, warmup=dec_warmup)

# early_stop = EarlyStopping(params.PATIENCE)

# model_enc.train()
model_dec.train()
model_cls.train()
# early_stop = EarlyStopping(params.PATIENCE)

for epoch in range(params.N_EPOCHS):
    run_epoch_true_neg(epoch=epoch, data_iter=train_iter,
                       model_dec=model_dec, opt_dec=opt_dec,
                       model_cls=model_cls, opt_cls=opt_cls, cls_criteria=cls_criteria,
                       seq2seq_criteria=seq2seq_criteria,
                       params=params)

    # rm.save_models_on_epoch_end(epoch)
    test_acc = evaluate_true_neg(epoch, test_iter, model_dec,
                                 model_cls, cls_criteria, seq2seq_criteria,
                                 params)

    test_random_samples(test_iter, TEXT, model_dec, model_cls, params.device,
                        decode_func=greedy_decode_sent, num_samples=2, transfer_style=True,
                        trans_cls=params.TRANS_CLS, embed_preds=True)

    # TODO - Roy - currently not in use, what metric to follow ?
    # early_stop(test_acc)
    # if early_stop.early_stop:
    #     break
