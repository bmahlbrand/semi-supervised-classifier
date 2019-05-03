# (64)3c-4p-
# (64)3c-3p-
# (128)3c-
# (128)3c-2p-
# (256)3c-
# (256)3c-
# (256)3c-
# (512)3c-
# (512)3c-
# (512)3c-2p-10fc 

import torch.nn as nn

# vanilla straight from paper
class StackedWhatWhereAutoEncoder(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(StackedWhatWhereAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # (64)3c-4p-
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=4),
            # (64)3c-3p-
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            # (128)3c-
            nn.Conv2d(128, 128, kernel_size=3),
            # (128)3c-2p-
            nn.Conv2d(128, 256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            # (256)3c-
            nn.Conv2d(256, 256, kernel_size=3),
            # (256)3c-
            nn.Conv2d(256, 512, kernel_size=3),
            # (256)3c-
            nn.Conv2d(512, 512, kernel_size=3),
            # (512)3c-
            nn.Conv2d(512, 512, kernel_size=3),
            # (512)3c-
            nn.Conv2d(512, 10, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            # (512)3c-2p-10fc 
            nn.Linear(10, num_classes)
        )

        self.decoder = nn.Sequential(
            # (512)3c-2p-10fc 
            nn.Linear(10, num_classes),
            # (512)3c-
            nn.ConvTranspose2d(512, 10, kernel_size=3),
            nn.MaxUnpool2d(kernel_size=2),
            # (512)3c-
            nn.ConvTranspose2d(512, 512, kernel_size=3),
            # (256)3c-
            nn.ConvTranspose2d(512, 512, kernel_size=3),
            # (256)3c-
            nn.ConvTranspose2d(256, 512, kernel_size=3),
            # (256)3c-
            nn.ConvTranspose2d(256, 256, kernel_size=3),
            # (128)3c-2p-
            nn.ConvTranspose2d(128, 256, kernel_size=3),
            nn.MaxUnpool2d(kernel_size=2),
            # (128)3c-
            nn.ConvTranspose2d(128, 128, kernel_size=3),
            # (64)3c-3p-
            nn.ConvTranspose2d(64, 128, kernel_size=3),
            nn.MaxUnpool2d(kernel_size=3),
            # (64)3c-4p-
            nn.ConvTranspose2d(64, 64, kernel_size=3),
            nn.MaxUnpool2d(kernel_size=4)
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.fc(x)