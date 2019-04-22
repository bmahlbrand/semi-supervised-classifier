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

## hyperparameters
parser.add_argument('--batch-size', default=8, type=int, metavar='B',
                    help='batch size (default: 8)')

parser.add_argument('--learning-rate', default=1e-4, type=float, metavar='L', help="initial learning rate")

parser.add_argument('--seed', type=int, default=0xDEADBEEF, metavar='S', help='random seed (default: (0xDEADBEEF)')

subparsers = parser.add_subparsers(help='optimizer type')

sgd_parser = subparsers.add_parser("sgd")
adam_parser = subparsers.add_parser("adam")


sgd_parser.add_argument('--dampening', type=float, default=0.1, metavar='DA',
                    help='SGD dampening (default: 0.1)')

sgd_parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='WDE',
                    help='SGD weight decay (default: 0.0005)')

sgd_parser.add_argument('--decay', type=float, default=0.1, metavar='DE',
                    help='SGD learning rate decay (default: 0.1)')

sgd_parser.add_argument('--momentum', type=float, default=0.9, metavar='MO',
                    help='SGD learning rate decay (default: 0.9)')

sgd_parser.add_argument('--nesterov', type=bool, default=False, metavar='NE',
                    help='SGD nesterov momentum formula (default: False)')


adam_parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                    help=' Adam parameter beta1 (default: 0.9)')

adam_parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                    help=' Adam parameter beta2 (default: 0.999)')
                    
adam_parser.add_argument('--epsilon', type=float, default=1e-6, metavar='EL',
                    help=' Adam regularization parameter (default: (1e-6)')


## system
parser.add_argument('--log-interval', default=100, type=int, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--cuda', default=True, type=bool, metavar='C',
                    help='use cuda or not (default: true)')

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

train_loader, val_loader, unsup_loader = image_loader('data', args.batch_size)

if args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

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

