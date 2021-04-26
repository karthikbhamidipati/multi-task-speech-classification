from _pickle import load
from os.path import join

import numpy as np
import torch

from models import run_device
from models.dataset import get_data_loaders
from models.loss_func import get_loss_function
from models.tests.test_models import Config
from models.train import compute_metrics
from utils.config import FEATURES_PATH, MODELS, HYPER_PARAMETERS


def _test(model, test_loader, criterion):
    test_loss, test_metrics = 0.0, np.zeros(2)
    for data, labels in test_loader:
        # move data, labels to run_device
        data = data.to(run_device)
        labels = [label.to(run_device) for label in labels]

        # forward pass without grad to calculate the validation loss
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, labels)

        # calculate validation loss
        test_loss += loss.item()
        test_metrics += compute_metrics(outputs, labels)

    print("Test Loss: {}, Test Metrics: {}"
          .format(test_loss, test_metrics))


def test_model(root_dir, model_key, checkpoint_path):
    features = load(open(join(root_dir, FEATURES_PATH), 'rb'))
    _, model_func = MODELS[model_key]
    config = Config(HYPER_PARAMETERS)

    data_loaders = get_data_loaders(features, config)
    loss_func = get_loss_function(config, features)

    model = torch.load(join(checkpoint_path, model_key + ".pt"))

    _test(model, data_loaders['test'], loss_func)
