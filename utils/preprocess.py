from glob import glob
from os.path import join
from pickle import dump

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from librosa import get_duration, load
from librosa.feature import melspectrogram, mfcc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.config import RAW_DATA_DIR, NUM_CORES, PRE_PROCESSED_DATA_DIR, FILTERED_FILE_NAME, ANNOTATED_FILE_NAME, \
    MAX_TIME_LEN, FEATURE_ARGS, EXTRACTOR_ARGS, FEATURES_PATH


def _compute_audio_duration(parent_dir, filename):
    filepath = parent_dir + filename
    return get_duration(filename=filepath)


def _get_mappings(df, col_name):
    idx2col = dict(enumerate(df[col_name].cat.categories))
    col2idx = dict((v, k) for k, v in idx2col.items())
    return col2idx, idx2col


def _get_class_weights(df, col_name, lookup):
    weights = np.zeros(len(lookup))
    value_counts = df[col_name].value_counts(normalize=True).to_dict()
    for key, value in lookup.items():
        weights[value] = value_counts[key]
    return weights


def _clean_features(features):
    features = features.transpose(1, 0)
    if len(features) < MAX_TIME_LEN:
        features = np.pad(features, ((MAX_TIME_LEN - len(features), 0), (0, 0)))
    return features[:MAX_TIME_LEN]


def extract_audio_features(root_dir, row):
    raw_data_dir = join(root_dir, RAW_DATA_DIR)
    row_dict = row.to_dict()
    waveform, _ = load(raw_data_dir + row_dict['filename'], sr=FEATURE_ARGS['sr'])
    row_dict['melspec'] = _clean_features(melspectrogram(waveform, n_mels=EXTRACTOR_ARGS['n_mels'], **FEATURE_ARGS))
    row_dict['mfcc'] = _clean_features(mfcc(waveform, n_mfcc=EXTRACTOR_ARGS['n_mfcc'], **FEATURE_ARGS))
    return row_dict


def _get_features(root_dir, df):
    job = Parallel(n_jobs=NUM_CORES)
    return job(delayed(extract_audio_features)(root_dir, row)
               for index, row in tqdm(df.iterrows(), total=df.shape[0]))


def preprocess_csv(root_dir):
    # getting the full path of raw features dir
    raw_data_dir = join(root_dir, RAW_DATA_DIR)

    # reading the csv files
    common_voice_df = pd.concat(map(pd.read_csv, glob(raw_data_dir + "*.csv")))
    print('Total number of records: {}'.format(common_voice_df.shape[0]), flush=True)

    # filtering rows and columns that are not required
    common_voice_df = common_voice_df[common_voice_df.accent.notna() &
                                      common_voice_df.gender.notna() &
                                      common_voice_df.age.notna()].reset_index(drop=True)

    common_voice_df = common_voice_df[["filename", "age", "gender", "accent"]]
    print('After removing empty/null rows: {}'.format(common_voice_df.shape[0]), flush=True)

    # calculating and adding duration to the dataset
    job = Parallel(n_jobs=NUM_CORES)
    durations = job(delayed(_compute_audio_duration)(raw_data_dir, filename)
                    for filename in tqdm(common_voice_df.filename))

    common_voice_df['duration'] = durations

    # saving the dataframe
    common_voice_df.to_csv(join(root_dir, ANNOTATED_FILE_NAME), index=False)

    # filtering the dataframe based on audio duration
    common_voice_df = common_voice_df[(common_voice_df.duration >= 2.0) &
                                      (common_voice_df.duration <= 5.0)].reset_index(drop=True)
    print('After filtering using duration length: {}'.format(common_voice_df.shape[0]), flush=True)

    # saving the filtered dataframe
    common_voice_df.to_csv(join(root_dir, FILTERED_FILE_NAME), index=False)


def extract_features(root_dir):
    # reading the filtered dataframe
    common_voice_df = pd.read_csv(join(root_dir, FILTERED_FILE_NAME))
    common_voice_df = common_voice_df.astype(
        {'filename': 'string', 'age': 'category', 'gender': 'category', 'accent': 'category'})

    # Train test split
    common_voice_train_df, common_voice_test_df = train_test_split(common_voice_df, test_size=0.2, random_state=0,
                                                                   stratify=common_voice_df[['gender', 'accent']])

    # Train validation split
    common_voice_train_df, common_voice_val_df = train_test_split(common_voice_train_df, test_size=0.2, random_state=0,
                                                                  stratify=common_voice_train_df[['gender', 'accent']])

    print("Train Shape: {}, Validation Shape: {}, Test Shape: {}".format(common_voice_train_df.shape,
                                                                         common_voice_val_df.shape,
                                                                         common_voice_test_df.shape))

    age2idx, idx2age = _get_mappings(common_voice_df, 'age')
    gender2idx, idx2gender = _get_mappings(common_voice_df, 'gender')
    accent2idx, idx2accent = _get_mappings(common_voice_df, 'accent')

    age_weights = _get_class_weights(common_voice_train_df, 'age', age2idx)
    gender_weights = _get_class_weights(common_voice_train_df, 'gender', gender2idx)
    accent_weights = _get_class_weights(common_voice_train_df, 'accent', accent2idx)

    print("Extracting train features", flush=True)
    train_features = _get_features(root_dir, common_voice_train_df)

    print("Extracting val features", flush=True)
    val_features = _get_features(root_dir, common_voice_val_df)

    print("Extracting test features", flush=True)
    test_features = _get_features(root_dir, common_voice_test_df)

    features = {
        "mappings": {
            "accent": {
                "accent2idx": accent2idx,
                "idx2accent": idx2accent,
                "weights": accent_weights
            },
            "age": {
                "age2idx": age2idx,
                "idx2age": idx2age,
                "weights": age_weights
            },
            "gender": {
                "gender2idx": gender2idx,
                "idx2gender": idx2gender,
                "weights": gender_weights
            }
        },
        "processed_data": {
            "train_set": train_features,
            "val_set": val_features,
            "test_set": test_features
        }
    }

    print("Saving the features...", flush=True)
    dump(features, open(join(root_dir, FEATURES_PATH), 'wb'))


def preprocess(root_dir):
    preprocess_csv(root_dir)
    extract_features(root_dir)
