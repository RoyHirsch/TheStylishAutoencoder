import os

# Loggin
VERBOSE = True

# Data
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
TRAIN_SIZE = 20000 # For debug
MAX_LEN = 128
# DATA_PATH = "/content/drive/My Drive/NLP/final/"
# TODO: for local use
DATA_PATH = os.path.abspath(__file__+'/../')

# Transformer model
N_LAYERS = 6
H_DIM = 512
N_ATTN_HEAD = 8
FC_DIM = 2048
DO_RATE = 0.1

# Classification model
N_STYLES = 2
DO_RATE_CLS = 0.5

# Resource Managment
MODELS_PATH = "/content/drive/My Drive/NLP/final/"

# Train
N_EPOCHS = 100
LR = 0.1
OPT_WARMUP_FACTOR = 4000
REC_LAMBDA = 0.5
CLS_STEPS = 500
TRANS_STEPS = 500
PRINT_INTERVAL = 100