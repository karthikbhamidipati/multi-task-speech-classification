from _pickle import load
from math import ceil
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from models import run_device
from models.dataset import get_data_loaders
from models.loss_func import get_loss_function
from models.tests.test_models import Config
from models.train import compute_metrics
from utils.config import FEATURES_PATH, MODELS, HYPER_PARAMETERS


def test(checkpoint_path, model_key, test_loader, criterion):
    test_loss, test_metrics = 0.0, np.zeros(2)

    dataset_size = len(test_loader.dataset)
    num_test_iters = ceil(dataset_size / test_loader.batch_size)
    ground_truth_accent, predictions_accent = [], []
    ground_truth_gender, predictions_gender = [], []

    model = torch.load(join(checkpoint_path, model_key + ".pt"))
    model.eval()

    for data, labels in tqdm(test_loader):
        # move data, labels to run_device
        data = data.to(run_device)

        ground_truth_gender.extend(labels[0].tolist())
        ground_truth_accent.extend(labels[1].tolist())

        labels = [label.to(run_device) for label in labels]

        # forward pass without grad to calculate the validation loss
        with torch.no_grad():
            outputs = model(data)

            predictions_gender.extend(outputs[0].argmax(dim=1).cpu().tolist())
            predictions_accent.extend(outputs[1].argmax(dim=1).cpu().tolist())

            loss = criterion(outputs, labels)

        # calculate validation loss
        test_loss += loss.item()
        test_metrics += compute_metrics(outputs, labels)

    print("Test Loss: {}, Test Metrics: {}"
          .format(test_loss / num_test_iters, test_metrics / num_test_iters))

    return np.array(ground_truth_gender), np.array(predictions_gender), np.array(ground_truth_accent), np.array(predictions_accent)


def test_model(root_dir, model_key, checkpoint_path):
    features = load(open(join("A:/Data/Audio/common-voice", "pre_processed_data", "common_voice_features.pkl"), 'rb'))
    _, model_func = MODELS[model_key]
    config = Config(HYPER_PARAMETERS)

    data_loaders = get_data_loaders(features, config)
    loss_func = get_loss_function(config, features)

    ground_truth_gender, predictions_gender, ground_truth_accent, predictions_accent = test(checkpoint_path, model_key,
                                                                                            data_loaders['test'],
                                                                                            loss_func)
    plot_confusion_matrix(ground_truth_gender, predictions_gender,
                          list(features['mappings']['gender']['idx2gender'].values()),
                          "gender")
    plot_confusion_matrix(ground_truth_accent, predictions_accent,
                          list(features['mappings']['accent']['idx2accent'].values()),
                          "accent")


def plot_confusion_matrix(ground_truth, predictions, display_labels, fig_name):
    cm = confusion_matrix(ground_truth, predictions)
    fig, ax = plt.subplots(figsize=(15, 15))

    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=display_labels)
    display.plot(ax=ax, xticks_rotation='vertical')
    plt.savefig("figures/" + fig_name + ".png")
