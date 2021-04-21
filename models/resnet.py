from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Dropout, init, AdaptiveAvgPool2d, \
    GroupNorm
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


class _ResNet(Module):
    def __init__(self, config, layers, block, inplanes=64, groups=1, norm_layer=BatchNorm2d):
        super(_ResNet, self).__init__()

        # type checks
        if type(layers) == list and len(layers) != 4 and all(map(lambda x: isinstance(x, int), layers)):
            raise ValueError("layers should be a list of ints with size 4")
        elif block not in (BasicBlock, Bottleneck):
            raise ValueError("invalid block, possible values: <BasicBlock|Bottleneck>")

        # constants
        self.inplanes = inplanes
        self.base_width = inplanes
        self.groups = groups
        self.norm_layer = norm_layer
        self.linear_units = 512 if block == BasicBlock else 2048

        # Initial convolutional layer
        self.conv = Sequential(
            Conv2d(1, self.inplanes, kernel_size=(7, 5), stride=(2, 1), padding=(3, 2), bias=False),
            self.norm_layer(self.inplanes),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))
        )

        # Residual blocks
        self.res1 = self._make_layer(block, 64, layers[0])
        self.res2 = self._make_layer(block, 128, layers[1], stride=2)
        self.res3 = self._make_layer(block, 256, layers[2], stride=2)
        self.res4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = AdaptiveAvgPool2d((1, 1))

        # fully connected output layers
        self.gender_out = Sequential(
            Dropout(config.fc_dropout),
            Linear(self.linear_units, 3)
        )

        self.accent_out = Sequential(
            Dropout(config.fc_dropout),
            Linear(self.linear_units, 16)
        )

        # initialise the network's weights
        self._init_weights()

    def forward(self, x):
        # reshape the input into 4D format by adding an empty dimension at axis 1 for channels 
        # (batch_size, channels, time_len, frequency_len)
        x = x.unsqueeze(1)

        # pass through convolutional layer
        x = self.conv(x)

        # pass through residual blocks, and pooling layers
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avg_pool(x)

        # flatten the features
        x = x.view(x.shape[0], -1)

        # pass through the multiple output layers
        gender = self.gender_out(x)
        accent = self.accent_out(x)

        return [gender, accent]

    def _init_weights(self):
        # Method to iteratively initialize network weights
        # conv layers, weights are initialized using kaiming normal initialization
        # linear layers, weights are initialized using xavier normal initialization
        # batch norm will have it's weights initialized with 1 and bias with 0        
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, Linear):
                init.xavier_normal_(m.weight)
            elif isinstance(m, (BatchNorm2d, GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, Bottleneck):
                init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, 1, norm_layer)]

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, norm_layer=norm_layer))

        return Sequential(*layers)


def resnet18(config):
    return _ResNet(config, [2, 2, 2, 2], block=BasicBlock)


def resnet34(config):
    return _ResNet(config, [3, 4, 6, 3], block=BasicBlock)


def resnet50(config):
    return _ResNet(config, [3, 4, 6, 3], block=Bottleneck)


def resnet101(config):
    return _ResNet(config, [3, 4, 23, 3], block=Bottleneck)