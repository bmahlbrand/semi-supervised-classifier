import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# (tensor([0.5032, 0.4746, 0.4275]), tensor([0.2268, 0.2225, 0.2256]))

class DLDataset(data.Dataset):
    def __init__(self):
        
        self.transform = transforms.Normalize((0.5032, 0.4746, 0.4275),(0.2268, 0.2225, 0.2256))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass