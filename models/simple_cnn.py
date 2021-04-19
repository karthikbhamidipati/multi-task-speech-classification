from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Dropout, init


class _SimpleCNNModel(Module):
    def __init__(self, config):
        super(_SimpleCNNModel, self).__init__()

        # convolutional layers
        self.conv1 = Sequential(
            Conv2d(1, 64, kernel_size=(5, 5), padding=(2, 2)),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(8, 4))
        )

        self.conv2 = Sequential(
            Conv2d(64, 128, kernel_size=(5, 5), padding=(2, 2)),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=(4, 2))
        )

        self.conv3 = Sequential(
            Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        # fully connected output layers
        self.gender_out = Sequential(
            Linear(2048, 256),
            ReLU(),
            Dropout(config.fc_dropout),
            Linear(256, 3)
        )

        self.accent_out = Sequential(
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 16)
        )

        # initialise the network's weights
        self.init_weights()

    def forward(self, x):
        # reshape the input into 4D format by adding an empty dimension at axis 1 for channels
        # (batch_size, channels, time_len, frequency_len)
        x = x.unsqueeze(1)

        # pass through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

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
        elif type(layer) in (Conv2d, Linear):
            init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, 0.001)
        elif type(layer) == BatchNorm2d:
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)


def simple_cnn(config):
    return _SimpleCNNModel(config=config)
