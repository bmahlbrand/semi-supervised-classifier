"""
This script is for getting the mean and std of three channels of the datasets. 
"""

import numpy as np
from DataLoader import image_loader

import argparse

parser = argparse.ArgumentParser(description='compute std and mean of dataset')
parser.add_argument('--dataset', type=str, default="data", metavar='N', help='input batch size for training (default: ./data)')
args = parser.parse_args()

def computeStatistics(loader):

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, target in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

print("computing mean and standard deviation of dataset...")

data_loader_sup_train, data_loader_sup_val, data_loader_unsup = image_loader(args.dataset, 8)
print(computeStatistics(data_loader_sup_train))