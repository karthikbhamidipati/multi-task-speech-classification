from math import ceil
from os.path import join
from pickle import load

import numpy as np
import wandb
from sklearn.metrics import f1_score

import torch
from torch.optim import Adam

from models import run_device
from models.dataset import get_data_loaders
from models.loss_func import get_loss_function
from utils.config import FEATURES_PATH, PROJECT_NAME, HYPER_PARAMETERS, MODELS


def start_wandb_session(run_name):
    wandb.init(project=PROJECT_NAME, config=HYPER_PARAMETERS)
    wandb.run.name = run_name
    return wandb.config


def train_model(root_dir, model_key):
    # get features and init wandb session
    features = load(open(join(root_dir, FEATURES_PATH), 'rb'))
    run_name, model_func = MODELS[model_key]
    config = start_wandb_session(run_name)

    # get data_loaders, loss_func, model, optimizer
    data_loaders = get_data_loaders(features, config)
    loss_func = get_loss_function(config, features)
    model = model_func(config).to(run_device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # set wandb to watch model parameters
    wandb.watch(model, log='all')
    model_save_path = join(wandb.run.dir, "model-" + wandb.run.id + ".pt")

    # begin training
    _train(config.num_epochs, data_loaders, model, optimizer, loss_func, model_save_path)

    # finish wandb session
    wandb.finish()


# helper for computing metrics
def _compute_metrics(predictions, targets):
    metrics = []
    for prediction, target in zip(predictions, targets):
        prediction = prediction.argmax(dim=1).cpu().numpy()
        target = target.cpu().numpy()
        metrics.append(f1_score(target, prediction, average='macro'))
    return metrics


def _train(num_epochs, loaders, model, optimizer, criterion, save_path, min_loss=np.Inf):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    val_loss_min = min_loss
    num_train_iters = ceil(len(loaders['train'].dataset) / loaders['train'].batch_size)
    num_val_iters = ceil(len(loaders['val'].dataset) / loaders['val'].batch_size)

    for epoch in range(1, num_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss, val_loss = 0.0, 0.0
        train_metrics, val_metrics = np.zeros(2), np.zeros(2)

        # training the model
        model.train()
        for batch_idx, (data, labels) in enumerate(loaders['train'], 1):
            # move data, labels to run_device
            data = data.to(run_device)
            labels = [label.to(run_device) for label in labels]

            # forward pass, backward pass and update weights
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            # calculate training loss and metrics
            train_loss += loss.item()
            train_metrics += _compute_metrics(outputs, labels)

        # evaluating the model
        model.eval()
        for batch_idx, (data, labels) in enumerate(loaders['val'], 1):
            # move data, labels to run_device
            data = data.to(run_device)
            labels = [label.to(run_device) for label in labels]

            # forward pass without grad to calculate the validation loss
            with torch.no_grad():
                outputs = model(data)
                loss = criterion(outputs, labels)

            # calculate validation loss
            val_loss += loss.item()
            val_metrics += _compute_metrics(outputs, labels)

        # compute average loss and accuracy
        train_loss /= num_train_iters
        val_loss /= num_val_iters
        train_metrics *= 100 / num_train_iters
        val_metrics *= 100 / num_val_iters

        # logging metrics to wandb
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_fscore_gender': train_metrics[0],
                   'train_fscore_accent': train_metrics[1], 'val_loss': val_loss, 'val_fscore_gender': val_metrics[0],
                   'val_fscore_accent': val_metrics[1]})

        # print training & validation statistics
        print(
            "Epoch: {}\tTraining Loss: {:.6f}\tTraining F-score: {}\tValidation Loss: {:.6f}\tValidation F-score: {}".format(
                epoch, train_loss, train_metrics, val_loss, val_metrics))

        # saving the model when validation loss decreases
        if val_loss <= val_loss_min:
            print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(val_loss_min, val_loss))
            torch.save(model.module, save_path)
            val_loss_min = val_loss
