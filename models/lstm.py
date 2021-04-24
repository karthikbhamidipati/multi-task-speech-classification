import torch
from torch.nn import Module, Sequential, LSTM, BatchNorm2d, Linear, Dropout, init, AdaptiveAvgPool2d

from models import run_device
from models.attention import Attention


class _LSTMModel(Module):
    def __init__(self, config, hidden_size=512, n_layers=8, bidirectional=False, attention=False):
        super(_LSTMModel, self).__init__()
        self.attention = attention

        # lstm layers
        self.lstm = LSTM(64, hidden_size, n_layers, dropout=config.lstm_dropout, bidirectional=bidirectional)

        n_layers *= 2 if bidirectional else 1
        self.hidden_cell = (torch.zeros(n_layers, 256, hidden_size).to(run_device),
                            torch.zeros(n_layers, 256, hidden_size).to(run_device))

        hidden_size *= 2 if bidirectional else 1

        if attention:
            self.att_layer = Attention(hidden_size, (256, hidden_size), batch_first=True)

        self.avg_pooling = AdaptiveAvgPool2d((1, hidden_size))

        # fully connected output layers
        self.gender_out = Sequential(
            Dropout(config.fc_dropout),
            Linear(hidden_size, 3)
        )

        self.accent_out = Sequential(
            Dropout(config.fc_dropout),
            Linear(hidden_size, 16)
        )

        # initialise the network's weights
        self.init_weights()

    def forward(self, x):
        # pass through lstm layers
        x, self.hidden_cell = self.lstm(x, self.hidden_cell)

        if self.attention:
            x, _ = self.att_layer(x)
        else:
            # reshape the input into 4D format
            x = x.unsqueeze(1)
            x = self.avg_pooling(x)

        # flatten the features
        x = x.view(x.shape[0], -1)

        # pass through the multiple output layers
        gender = self.gender_out(x)
        accent = self.accent_out(x)

        return [gender, accent]

    def init_weights(self, layer=None):
        # Method to recursively initialize network weights
        # linear and conv layers, weights are initialized using xavier normal initialization
        # batch norm will have it's weights initialized with 1 and bias with 0
        if not layer or type(layer) == Sequential:
            children = layer.children() if layer else self.children()
            for module in children:
                self.init_weights(module)
        elif type(layer) == Linear:
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0.001)
        elif type(layer) == BatchNorm2d:
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)


def simple_lstm(config):
    return _LSTMModel(config=config)


def bi_lstm(config):
    return _LSTMModel(config=config, bidirectional=True)


def lstm_attention(config):
    return _LSTMModel(config=config, bidirectional=False, attention=True)


def bi_lstm_attention(config):
    return _LSTMModel(config=config, bidirectional=True, attention=True)
