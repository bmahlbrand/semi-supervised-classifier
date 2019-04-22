import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import argparse
import numpy as np

from utils.EarlyStopping import EarlyStopping

from DataLoader import image_loader

from utils.config_utils import load_config
from utils.Timer import Timer
from utils.fs_utils import create_folder
from utils.logger import Logger
import utils.torch_utils as torch_utils


folderPath = 'checkpoints/session_' + Timer.timeFilenameString() + '/'
create_folder(folderPath)

logPath = 'log/log_' + Timer.timeFilenameString()

parser = argparse.ArgumentParser(description='PyTorch training script for unsupervised classifier')



## system training
parser.add_argument('--log-interval', default=100, type=int, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--workers', default=0, type=int, metavar='W',
                    help='workers (default: 0)')

parser.add_argument('--train_dir', default='data', type=str, metavar='PATHT',
                    help='path to latest checkpoint (default: data folder)')

parser.add_argument('--val_dir', default='data', type=str, metavar='PATHV',
                    help='path to latest checkpoint (default: data folder)')                    

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()

def train(epoch, model, optimizer, criterion, loader, device, log_callback):
    pass

def validation(model, criterion, loader, device, log_callback):
    pass


start_epoch = 1

if args.resume:
    start_epoch, model, optimizer, scheduler = torch_utils.load(args.resume, model, optimizer, start_epoch, scheduler)
    # append_line_to_log('resuming ' + args.resume + '... at epoch ' + str(start_epoch))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# put model into the corresponding device
model.to(device)

# append_line_to_log('executing on device: ')
# append_line_to_log(str(device))

# history = {'validation_loss':[]}

# best_val_loss = np.inf

# for epoch in range(start_epoch, args.epochs + 1):
    
#     early_stop = EarlyStopping(0., model)

#     if early_stop.early_stop:
#         # append_line_to_log("Early stopping")
#         break

