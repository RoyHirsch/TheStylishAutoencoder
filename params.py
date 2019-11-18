import torch


class Params(object):
    # Loggin
    # Free text to describe the experiment
    COMMENT = ''
    VERBOSE = True
    EXP_NAME = "gal_exp"
    DATA_PATH = "/content/drive/My Drive/NLP/"
    MODELS_PATH = "/content/drive/My Drive/NLP/final/"
    PRINT_INTERVAL = 10

    MODELS_LOAD_PATH = "/content/drive/My Drive/NLP/final/gal_exp"

    # Data
    DATASET_NAME = 'YELP'
    # Maximal number of batches for test model
    TEST_MAX_BATCH_SIZE = 300
    # Min freq for word in dataset to include in vocab
    VOCAB_MIN_FREQ = 1
    VOCAB_MAX_SIZE = 30000
    # Whether to use Glove embadding - if TRUE set H_DIM to 300
    VOCAB_USE_GLOVE = True
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
