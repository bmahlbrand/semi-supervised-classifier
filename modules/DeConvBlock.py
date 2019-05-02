import torch.nn as nn

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.conv(x)
