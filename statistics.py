"""
This script is for getting the mean and std of three channels of the datasets. 
"""

import numpy as np
from Dataset import DLDataset 

def computeStatistics(loader):

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

if __name__ == '__main__':
    print("computing mean and standard deviation of dataset...")

    dataset = DLDataset()
    computeStatistics(dataset)