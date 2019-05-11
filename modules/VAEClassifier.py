import torch
import torch.nn as nn
import numpy as np

from modules.ConvBlock import ConvBlock
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.add_module('conv1', ConvBlock(64, 64))
        self.add_module('maxpool1', nn.MaxPool2d(kernel_size=4, return_indices=True))
        # (64)3c-3p-
        self.add_module('conv2', ConvBlock(64, 128))
        self.add_module('maxpool2', nn.MaxPool2d(kernel_size=3, return_indices=True))
        # (128)3c-
        self.add_module('conv3', ConvBlock(128, 128))
        # (128)3c-2p-
        self.add_module('conv4', ConvBlock(128, 256))
        self.add_module('maxpool3', nn.MaxPool2d(kernel_size=2, return_indices=True))
        # (256)3c-
        self.add_module('conv5', ConvBlock(256, 256))
        # (256)3c-
        self.add_module('conv6', ConvBlock(256, 512))
        # (256)3c-
        self.add_module('conv7', ConvBlock(512, 512))
        # (512)3c-
        self.add_module('conv8', ConvBlock(512, 512))
        # (512)3c-
        # 'conv9', ConvBlock(512, 10))
        self.add_module('maxpool4', nn.MaxPool2d(kernel_size=2, return_indices=True))
        # (512)3c-2p-10fc
        self.add_module('fc1', nn.Linear(2048, 10))

        # )

    def forward(self, x):
        x = self._modules['conv1'](x)
        x, indices1 = self._modules['maxpool1'](x)
        x = self._modules['conv2'](x)
        x, indices2 = self._modules['maxpool2'](x)
        x = self._modules['conv3'](x)
        x = self._modules['conv4'](x)
        x, indices3 = self._modules['maxpool3'](x)
        x = self._modules['conv5'](x)
        x = self._modules['conv6'](x)
        x = self._modules['conv7'](x)
        x = self._modules['conv8'](x)
        # x = self['conv9'](x)
        x, indices4 = self._modules['maxpool4'](x)
        #         print(x.shape)
        x = x.view(x.size(0), -1)
        x = self._modules['fc1'](x)
        return x, indices1, indices2, indices3, indices4


class VAEClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(VAEClassifier, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=1)
        self.encoder = Encoder()
        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        self.convFinal = nn.Conv2d(64, 3, kernel_size=1)

    # todo, load encoder weights
    def load(self, path):
        pass

    # this will need debugging
    def forward(self, x):
        x = self.conv0(x)
        x, indices1, indices2, indices3, indices4 = self.encoder(x)
        x = self.classifier(x)

        return x

