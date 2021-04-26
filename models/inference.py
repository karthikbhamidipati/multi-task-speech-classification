import os
from os.path import join

import torch

from models import run_device
from utils.config import GENDER_CLASSES, ACCENT_CLASSES, RAW_DATA_DIR
from utils.preprocess import extract_audio_features


class Row:
    def __init__(self, filename):
        self.filename = filename

    def to_dict(self):
        return {
            "filename": self.filename
        }


def inference(root_dir, model_path):
    audio_features = [extract_audio_features(root_dir, Row(filename))['mfcc'] for filename in
                      os.listdir(join(root_dir, RAW_DATA_DIR))]
    audio_features = torch.Tensor(audio_features).to(run_device)
    model = torch.load(model_path).to(run_device)

    model.eval()
    with torch.no_grad():
        predictions_gender, predictions_accent = model(audio_features)
        predictions_gender = predictions_gender.argmax(dim=1).cpu().numpy()
        predictions_accent = predictions_accent.argmax(dim=1).cpu().numpy()

    for filename, gender, accent in zip(os.listdir(join(root_dir, RAW_DATA_DIR)), predictions_gender,
                                        predictions_accent):
        print("Filename: {}, Gender: {}, Accent: {}".format(filename, GENDER_CLASSES[gender], ACCENT_CLASSES[accent]))
