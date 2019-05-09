import torch
import torch.nn as nn
import numpy as np

from modules.ConvBlock import ConvBlock
from modules.DeConvBlock import DeConvBlock

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
        # self.add_module('fc1', nn.Linear(10, num_classes))
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

        return x, indices1, indices2, indices3, indices4

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # self.add_module(# (512)3c-2p-10fc 
            # 'fc1', nn.Linear(num_classes, 10))
            # (512)3c-
        self.add_module('unpool1', nn.MaxUnpool2d(kernel_size=2))
        self.add_module('deconv1', DeConvBlock(512, 512))
        # (512)3c-
        self.add_module('deconv2', DeConvBlock(512, 512))
        # (256)3c-
        self.add_module('deconv3', DeConvBlock(512, 512))
        # (256)3c-
        self.add_module('deconv4', DeConvBlock(512, 256))
        # (256)3c-
        self.add_module('deconv5', DeConvBlock(256, 256))
        # (128)3c-2p-
        self.add_module('unpool2', nn.MaxUnpool2d(kernel_size=2))
        self.add_module('deconv6', DeConvBlock(256, 128))
        
        # (128)3c-
        self.add_module('deconv7', DeConvBlock(128, 128))
        # (64)3c-3p-
        self.add_module('deconv8', DeConvBlock(128, 64))
        self.add_module('unpool3', nn.MaxUnpool2d(kernel_size=3))
        # (64)3c-4p-
        self.add_module('deconv9', DeConvBlock(64, 64))
        self.add_module('unpool4', nn.MaxUnpool2d(kernel_size=4))

    def forward(self, x, indices1, indices2, indices3, indices4):
        x = self._modules['unpool1'](x, indices4)
        x = self._modules['deconv1'](x)
        x = self._modules['deconv2'](x)
        x = self._modules['deconv3'](x)
        x = self._modules['deconv4'](x)
        x = self._modules['deconv5'](x)
        x = self._modules['unpool2'](x, indices3)
        x = self._modules['deconv6'](x)
        
        x = self._modules['deconv7'](x)
        x = self._modules['unpool3'](x, indices2)
        x = self._modules['deconv8'](x)
        x = self._modules['unpool4'](x, indices1)
        x = self._modules['deconv9'](x)

        return x

class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_classes=1000):
        super(VariationalAutoEncoder, self).__init__()

        # self.encoder = nn.Sequential(
        self.conv0 = nn.Conv2d(3, 64, kernel_size=1)
        self.encoder = Encoder()
        # self.encoder = {
        #     # (64)3c-4p-
        #     'conv1': ConvBlock(64, 64),
        #     'maxpool1': nn.MaxPool2d(kernel_size=4, return_indices=True),
        #     # (64)3c-3p-
        #     'conv2': ConvBlock(64, 128),
        #     'maxpool2': nn.MaxPool2d(kernel_size=3, return_indices=True),
        #     # (128)3c-
        #     'conv3': ConvBlock(128, 128),
        #     # (128)3c-2p-
        #     'conv4': ConvBlock(128, 256),
        #     'maxpool3': nn.MaxPool2d(kernel_size=2, return_indices=True),
        #     # (256)3c-
        #     'conv5': ConvBlock(256, 256),
        #     # (256)3c-
        #     'conv6': ConvBlock(256, 512),
        #     # (256)3c-
        #     'conv7': ConvBlock(512, 512),
        #     # (512)3c-
        #     'conv8': ConvBlock(512, 512),
        #     # (512)3c-
        #     # 'conv9': ConvBlock(512, 10),
        #     'maxpool4': nn.MaxPool2d(kernel_size=2, return_indices=True),
        #     # (512)3c-2p-10fc 
        #     'fc1': nn.Linear(10, num_classes)
        # # )
        # }

        # self.decoder = nn.Sequential(
        self.decoder = Decoder()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # print(x.shape)
        x = self.conv0(x)
        # print(x.shape)
        # x = self.encoder['conv1'](x)
        # x, indices1 = self.encoder['maxpool1'](x)
        # x = self.encoder['conv2'](x)
        # x, indices2 = self.encoder['maxpool2'](x)
        # x = self.encoder['conv3'](x)
        # x = self.encoder['conv4'](x)
        # x, indices3 = self.encoder['maxpool3'](x)
        # x = self.encoder['conv5'](x)
        # x = self.encoder['conv6'](x)
        # x = self.encoder['conv7'](x)
        # x = self.encoder['conv8'](x)
        # # x = self.encoder['conv9'](x)
        # x, indices4 = self.encoder['maxpool4'](x)
        
        mu_logvar, indices1, indices2, indices3, indices4 = self.encoder(x)
        # mu_logvar = self.encoder(x.view(-1, 1000)).view(-1, 2, d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z, indices1, indices2, indices3, indices4), mu, logvar
        
        # print(indices1.shape)
        # print(indices2.shape)
        # print(indices3.shape)
        # print(indices4.shape)
        # print(x.shape)
        # x = self.avgpool(x
        # x = x.view(x.size(0), -1)
        # # x = x.view(-1, np.prod(x.shape))
        
        
        # x = self.encoder['fc1'](x)

        # x = self.decoder['fc1'](x)
        
        
        # x = self.decoder['unpool1'](x, indices4)
        # x = self.decoder['deconv1'](x)
        # x = self.decoder['deconv2'](x)
        # x = self.decoder['deconv3'](x)
        # x = self.decoder['deconv4'](x)
        # x = self.decoder['deconv5'](x)
        # x = self.decoder['unpool2'](x, indices3)
        # x = self.decoder['deconv6'](x)
        
        # x = self.decoder['deconv7'](x)
        # x = self.decoder['unpool3'](x, indices2)
        # x = self.decoder['deconv8'](x)
        # x = self.decoder['unpool4'](x, indices1)
        # x = self.decoder['deconv9'](x)
        
        # x = self.decoder['fc1'](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    
    @staticmethod
    def loss_function(x_hat, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(
            x_hat, x.view(-1, 1000), reduction='sum'
        )
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KLD