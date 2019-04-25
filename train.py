import argparse
import json
import os
import time

import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

from utils.EarlyStopping import EarlyStopping

from DataLoader import image_loader

from utils.config_utils import load_config
from utils.Timer import Timer
from utils.fs_utils import create_folder
from utils.logger import Logger
from utils.AverageMeter import AverageMeter
from utils.torch_utils import torch_utils

folderPath = 'checkpoints/session_' + Timer.timeFilenameString() + '/'
create_folder(folderPath)

logPath = 'log/log_' + Timer.timeFilenameString()

parser = argparse.ArgumentParser(description='PyTorch training script for unsupervised classifier')

## hyperparameters
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='batch size (default: 8)')
parser.add_argument('--learning-rate', default=1e-4, type=float, metavar='L', help="initial learning rate")
parser.add_argument('--seed', type=int, default=0xDEADBEEF, metavar='S', help='random seed (default: (0xDEADBEEF)')

## scheduler
parser.add_argument("--config", type=str, default='', help='config json file to reload experiments')
subparsers = parser.add_subparsers(help='optimizer type')

sgd_parser = subparsers.add_parser("sgd")
adam_parser = subparsers.add_parser("adam")

sgd_parser.add_argument('--dampening', type=float, default=0.1, metavar='DA', help='SGD dampening (default: 0.1)')
sgd_parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='WDE', help='SGD weight decay (default: 0.0005)')
sgd_parser.add_argument('--decay', type=float, default=0.1, metavar='DE', help='SGD learning rate decay (default: 0.1)')
sgd_parser.add_argument('--momentum', type=float, default=0.9, metavar='MO', help='SGD learning rate decay (default: 0.9)')
sgd_parser.add_argument('--nesterov', type=bool, default=False, metavar='NE', help='SGD nesterov momentum formula (default: False)')

adam_parser.add_argument('--beta1', type=float, default=0.9, metavar='B1', help=' Adam parameter beta1 (default: 0.9)')
adam_parser.add_argument('--beta2', type=float, default=0.999, metavar='B2', help=' Adam parameter beta2 (default: 0.999)')
adam_parser.add_argument('--epsilon', type=float, default=1e-6, metavar='EL', help=' Adam regularization parameter (default: (1e-6)')


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


if args.config:
    with open(args.config, 'r') as f:
        args.__dict__ = json.load(f)

# save parameters of this experiment for reproduction later
with open('experiments/experiment_' + Timer.timeFilenameString() + '.json', 'w') as f:
    config = args.__dict__
    config['logpath'] = logPath
    config['best_model'] = os.path.join(folderPath, 'best_model.cpkt')
    config['checkpoints'] = folderPath

    json.dump(config, f, indent=4)

batch_time = AverageMeter()
data_time = AverageMeter()

def train(epoch, model, optimizer, criterion, loader, device, log_callback):

    end = time.time()
    model.train()

    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']

    # the output of the dataloader is (batch_idx, image, mask, c, v, t)
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        data_time.update(time.time() - end)
        
        # input all the input vectors into the model 
        output = model(data)

        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
 
        # record essential informations into log file.
        if batch_idx % args.log_interval == 0:
            log_callback('Epoch: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1} batches, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1} batches, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'.format(
                epoch, args.log_interval, learning_rate, batch_time=batch_time,
                data_time=data_time))
            
            log_callback('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
            log_callback()
            
            log_callback('Loss{0} = {loss:.8f}\t'
                    .format(1, loss=loss.item()))

            log_callback()
            log_callback("current time: " + Timer.timeString())
            
            batch_time.reset()
            data_time.reset()

    torch_utils.save(folderPath + 'SSL_DLSP19' + str(epoch) + '.cpkt', epoch, model, optimizer, scheduler)

def validation(model, criterion, loader, device, log_callback):
    end = time.time()
    model.eval()

    # return validation_loss, validation_acc
    with torch.no_grad():
        # the output of the dataloader is (batch_idx, image, mask, c, v, t)
        for batch_idx, (data, target) in enumerate(loader):
            target = target.to(device, non_blocking=True)
            data = data.to(device, non_blocking=True)

            output = model(data)
           
            # compute the loss
            loss = criterion(output, target)
        
            batch_time.update(time.time() - end)
            end = time.time()

        # records essential information into log file.
        log_callback('epoch: {0}\t'
                'Time {batch_time.sum:.3f}s / {1} epochs, ({batch_time.avg:.3f})\t'
                'Data load {data_time.sum:.3f}s / {1} epochs, ({data_time.avg:3f})\n'
                'Loss = {loss:.8f}\n'.format(
            epoch, batch_idx, batch_time=batch_time,
            data_time=data_time, loss=loss.item()))
        
        log_callback()
        
        log_callback('Loss{0} = {loss:.8f}\t'
                .format(1, loss=loss.item()))

        log_callback(Timer.timeString())

        batch_time.reset()
         
        return loss.item()

start_epoch = 1

# model =
# optimizer = 
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08)



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

history = { 
            'training_loss': [],
            'validation_loss': [],
            'validation_accuracy': []
          }

best_val_loss = np.inf
best_val_acc = 0.

for epoch in range(start_epoch, args.epochs + 1):

    early_stop = EarlyStopping(0., model)

    if early_stop.early_stop:
        # append_line_to_log("Early stopping")
        break

    training_loss = train(epoch, model, optimizer, criterion, train_loader, device, append_line_to_log)
    val_loss, val_acc = validation(model, criterion, val_loader, device, append_line_to_log)
    
    history['training_loss'].append(training_loss)
    history['validation_loss']
    history['validation_accuracy'].append(val_acc)

    scheduler.step(val_loss)

    # # is_best = val_loss < best_val_loss
    # is_best = val_acc > best_val_acc

    # # best_val_loss = min(val_loss, best_val_loss)
    # best_val_acc = max(val_acc, best_val_acc)

    # if is_best:
    #     best_model_file = 'best_model.pth'
    #     model_file = folderPath + best_model_file
    #     torch.save(model.state_dict(), model_file)
    # model_file = 'model_' + str(epoch) + '.pth'
    # model_file = folderPath + model_file

    # torch.save(model.state_dict(), model_file)
    # append_line_to_log('Saved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` \n')

# plt.plot(range(len(history['losses'])), history['losses'], 'g-')
# plt.xlabel('batch steps')
# plt.ylabel('test loss')
# plt.savefig('test_loss.png')

# plt.plot(range(args.epochs), history['validation_accuracy'], 'r-')
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.savefig('valid_accuracy.png')
