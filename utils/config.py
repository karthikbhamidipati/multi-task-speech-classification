import multiprocessing
from os.path import join

from models.lstm import simple_lstm, bi_lstm, lstm_attention, bi_lstm_attention
from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.simple_cnn import simple_cnn

# Config for reading csv files
RAW_DATA_DIR = "raw_data/"
PRE_PROCESSED_DATA_DIR = "pre_processed_data/"
ANNOTATED_FILE_NAME = join(PRE_PROCESSED_DATA_DIR, "common_voice_annotated.csv")
FILTERED_FILE_NAME = join(PRE_PROCESSED_DATA_DIR, "common_voice_annotated_filtered.csv")

# Config for multiprocessing
NUM_CORES = int(max((multiprocessing.cpu_count() / 2) - 1, 1))

# Audio Extraction configs
SAMPLING_RATE = 22050
MAX_TIME_LEN = 256

FEATURE_ARGS = {
    "sr": SAMPLING_RATE,
    "n_fft": 2048,
    "win_length": 1024,
    "hop_length": 512,
}
EXTRACTOR_ARGS = {
    "n_mels": 64,
    "n_mfcc": 64
}

FEATURES_PATH = join(PRE_PROCESSED_DATA_DIR, "common_voice_features.pkl")

# wandb args
PROJECT_NAME = "multitask-audio-classification"
HYPER_PARAMETERS = {
    'beta': 0.001,  # loss weighting parameter
    'num_epochs': 50,  # number of epochs
    'batch_size': 64,  # batch size
    'learning_rate': 0.001,  # learning rate
    'feature_type': 'mfcc',  # feature_type (mfcc or mel-spectrogram)
    'use_class_weights': False,  # use class weights for loss
    'fc_dropout': 0.1,  # dropout for fully connected layers
    'lstm_dropout': 0.2  # dropout for lstm layers
}

# Model args
MODELS = {
    "simple_cnn": ("CNN Baseline", simple_cnn),
    "resnet18": ("CNN ResNet18", resnet18),
    "resnet34": ("CNN ResNet34", resnet34),
    "resnet50": ("CNN ResNet50", resnet50),
    "resnet101": ("CNN ResNet101", resnet101),
    "simple_lstm": ("LSTM Baseline", simple_lstm),
    "bi_lstm": ("Bidirectional LSTM", bi_lstm),
    "lstm_attention": ("Bidirectional LSTM", lstm_attention),
    "bi_lstm_attention": ("Bidirectional LSTM with Attention", bi_lstm_attention)
}
