import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

from utils.EarlyStopping import EarlyStopping

from DataLoader import image_loader
import torchvision.transforms as transforms

from utils.Timer import Timer
from utils.fs_utils import create_folder
from utils.logger import Logger
from utils.AverageMeter import AverageMeter
from utils import torch_utils

def append_line_to_log(line = '\n'):
    with open(logPath, 'a') as f:
        f.write(line + '\n')


logPath = 'log/log_' + Timer.timeFilenameString()

parser = argparse.ArgumentParser(description='PyTorch training script for semi-supervised classifier')

parser.add_argument('--network', default='densenet', type=str, metavar='N', help='network architecture to use')
parser.add_argument('--augment', action='store_true', help='dataset augmentation')

## hyperparameters
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='batch size (default: 8)')
parser.add_argument('--learning-rate', default=1e-3, type=float, metavar='L', help="initial learning rate")
parser.add_argument('--seed', type=int, default=0xDEADBEEF, metavar='S', help='random seed (default: (0xDEADBEEF)')
parser.add_argument('--epochs', type=int, default=50, metavar='E', help='training iterations')
# parser.add_argument('--optimizer', type=str, default='sgd', metavar='O', help='which optimizer to use? supported types: [sgd, adam]')

## scheduler
parser.add_argument('--no-scheduler', action='store_true')
parser.add_argument('--mode', type=str, default='min')
parser.add_argument('--factor', type=float, default=0.7)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--threshold', type=float, default=1e-4)
parser.add_argument('--threshold_mode', type=str, default='rel')
parser.add_argument('--cooldown', type=int, default=2)
parser.add_argument('--min_lr', type=float, default=0.0)
parser.add_argument('--eps', type=float, default=1e-08)

## optimizer
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer algorithm')

parser.add_argument('--dampening', type=float, default=0.1, metavar='DA', help='SGD dampening (default: 0.1)')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='WDE', help='SGD weight decay (default: 0.0005)')
parser.add_argument('--decay', type=float, default=0.1, metavar='DE', help='SGD learning rate decay (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='MO', help='SGD learning rate decay (default: 0.9)')
parser.add_argument('--nesterov', type=bool, default=False, metavar='NE', help='SGD nesterov momentum formula (default: False)')

parser.add_argument('--beta1', type=float, default=0.9, metavar='B1', help=' Adam parameter beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='B2', help=' Adam parameter beta2 (default: 0.999)')
parser.add_argument('--epsilon', type=float, default=1e-6, metavar='EL', help=' Adam regularization parameter (default: (1e-6)')

## system
parser.add_argument('--config', type=str, default='', help='config json file to reload experiments')
parser.add_argument('--log-interval', default=100, type=int, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--cuda', action='store_true', help='use cuda or not (default: true)')
parser.add_argument('--pinned-memory', action='store_true', help='use memory pinning or not (default: true)')
parser.add_argument('--gpu-id', default=None, type=int, metavar='G', help='GPU ID to use')
parser.add_argument('--workers', default=0, type=int, metavar='W', help='workers (default: 0)')

parser.add_argument('--train-dir', default='data', type=str, metavar='PATHT', help='path to latest checkpoint (default: data-folder)')
parser.add_argument('--val-dir', default='data', type=str, metavar='PATHV', help='path to latest checkpoint (default: data-folder)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint-path', default='checkpoints', type=str, metavar='PATHC', help='base path to save checkpoints (default: checkpoints)')
parser.add_argument('--checkpoint-interval', default=5, type=int, metavar='C', help='interval to save checkpoints')

args = parser.parse_args()

# print(args)

if args.config:
    with open(args.config, 'r') as f:
        args.__dict__ = json.load(f)

experiment_filename = 'experiments/experiment_' + Timer.timeFilenameString() + '.json'

folderPath = args.checkpoint_path + '/session_' + Timer.timeFilenameString() + '/'


# save parameters of this experiment for reproduction later
with open(experiment_filename, 'w') as f:
    config = args.__dict__
    config['logpath'] = logPath
    config['best_model'] = os.path.join(folderPath, 'best_model.cpkt')
    config['checkpoints'] = folderPath

    json.dump(config, f, indent=4)

create_folder(folderPath)

batch_time = AverageMeter()
data_time = AverageMeter()

def train(epoch, model, optimizer, criterion, loader, device, log_callback):
    
    train_loss = AverageMeter()

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

        train_loss.update(loss.item())

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

    torch_utils.save(folderPath + 'SSL_DLSP19_' + str(epoch) + '.cpkt', epoch, model, optimizer, scheduler)
    return train_loss.avg

def validation(model, criterion, loader, device, log_callback, top_k):
    end = time.time()
    model.eval()

    validation_loss = AverageMeter()
    # correct = 0

    n_samples = 0.
    n_correct_top_1 = 0
    n_correct_top_k = 0

    # return validation_loss, validation_acc
    with torch.no_grad():
        # the output of the dataloader is (batch_idx, image, mask, c, v, t)
        for batch_idx, (data, target) in enumerate(loader):
            target = target.to(device, non_blocking=True)
            data = data.to(device, non_blocking=True)

            output = model(data)
        
            # Top 1 accuracy
            pred_top_1 = torch.topk(output, k=1, dim=1)[1]
            n_correct_top_1 += pred_top_1.eq(target.view_as(pred_top_1)).int().sum().item()

            # Top k accuracy
            pred_top_k = torch.topk(output, k=top_k, dim=1)[1]
            target_top_k = target.view(-1, 1).expand(args.batch_size, top_k)
            n_correct_top_k += pred_top_k.eq(target_top_k).int().sum().item()

            # compute the loss
            loss = criterion(output, target)
            validation_loss.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()
            # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        # Accuracy
        top_1_acc = n_correct_top_1/n_samples
        top_k_acc = n_correct_top_k/n_samples

        # validation_acc  = float(correct) / float(len(val_loader.dataset))
        log_callback('\nValidation set: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.4f}%), Top {} Accuracy: {}{} ({:.4f})\n'.format(
            validation_loss.avg, n_correct_top_1, len(val_loader.dataset),
            100. * top_1_acc, top_k, n_correct_top_k, len(val_loader.dataset), 100. * top_k_acc))
        
        # records essential information into log file.
        log_callback('epoch: {0}\t'
                'Time {batch_time.sum:.3f}s / {1} epochs, ({batch_time.avg:.3f})\t'
                'Data load {data_time.sum:.3f}s / {1} epochs, ({data_time.avg:3f})\n'
                'Average Validation Loss = {loss:.8f}, \nTop 1 Accuracy: {correct_1:3d}/{size:3d} ({acc_1:.4f}%)\n'
                'Top k Accuracy: {correct_k:3d}/{size:3d} ({acc_k:.4f}%)'.format(
            epoch, batch_idx, batch_time=batch_time,
            data_time=data_time, 
            loss=validation_loss.avg, correct_1=n_correct_top_1, size=len(val_loader.dataset),
            acc_1=100. * top_1_acc, 
            correct_k = n_correct_top_k, acc_k=100. * top_k_acc
            ))
        
        log_callback()
        
        # log_callback('Loss{0} = {loss:.8f}\t'
        #         .format(1, loss=loss.item()))

        log_callback(Timer.timeString())

        batch_time.reset()
        
        return validation_loss.avg, top_1_acc, top_k_acc

start_epoch = 1

import torchvision.models as models
# resnet = models.resnet18()
# alexnet = models.AlexNet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet121()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# model = densenet

from modules.VGG import vgg11
from modules.AutoEncoder import AutoEncoder
from modules.DenseNet import DenseNet

if args.network == 'vgg':
    model = vgg11()
elif args.network == 'densenet':
    model = models.densenet121()
    # model = DenseNet()
elif args.network == 'resnet':
    model = models.resnet18()
elif args.network == 'ae':
    model = AutoEncoder()
else:
    print('invalid network architecture specified')
    sys.exit()

augment_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation((-30, 30)),
                                    transforms.ColorJitter(brightness=.15, contrast=.15, hue=.05, saturation=.05)
                                    ])

if args.network in ['vgg', 'densenet', 'resnet']:
    scale_transform = transforms.Resize((224, 224))
else:
    scale_transform = transforms.Compose([])

if args.augment:
    train_loader, val_loader, unsup_loader = image_loader('data', args.batch_size, args.pinned_memory, args.workers, scale_transform=scale_transform, augment_transform=augment_transform)
else:
    train_loader, val_loader, unsup_loader = image_loader('data', args.batch_size, args.pinned_memory, args.workers, scale_transform=scale_transform)

optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.mode, factor=args.factor, patience=args.factor, verbose=args.verbose,
            threshold=args.threshold, threshold_mode=args.threshold_mode, cooldown=args.cooldown, min_lr=args.min_lr, eps=args.eps)

criterion = nn.CrossEntropyLoss()

if args.cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

if args.resume:
    start_epoch, model, optimizer, scheduler = torch_utils.load(args.resume, model, optimizer, start_epoch, scheduler)
    # append_line_to_log('resuming ' + args.resume + '... at epoch ' + str(start_epoch))

# put model into the corresponding device
model.to(device)

for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

if args.cuda and args.gpu_id is not None:
    print("Let's use GPU:", args.gpu_id)
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # Example::
    #
    #     >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    #     >>> output = net(input_var)
    #
    model = nn.DataParallel(model, device_ids=[args.gpu_id])
elif args.cuda and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

append_line_to_log('executing on device: ')
append_line_to_log(str(device))

if 'history' not in config.keys():
    history = { 
                'training_loss': [],
                'validation_loss': [],
                'validation_accuracy_top_1': [],
                'validation_accuracy_top_k': []
            }
else:
    history = config['history']

best_val_loss = np.inf
best_val_acc = 0.

early_stop = EarlyStopping(patience=7)

for epoch in range(start_epoch, args.epochs + 1):

    training_loss = train(epoch, model, optimizer, criterion, train_loader, device, append_line_to_log)
    val_loss, top_1_acc, top_k_acc = validation(model, criterion, val_loader, device, append_line_to_log, top_k=5)
    
    history['training_loss'].append(training_loss)
    history['validation_loss'].append(val_loss)
    history['validation_accuracy_top_1'].append(top_1_acc)
    history['validation_accuracy_top_k'].append(top_k_acc)

    if not args.no_scheduler:
        scheduler.step(val_loss)

    early_stop(val_loss)

    if early_stop.early_stop:
        append_line_to_log("Early stopping")
        break

    is_best = val_loss < best_val_loss
    # is_best = val_acc > best_val_acc

    best_val_loss = min(val_loss, best_val_loss)
    # best_val_acc = max(val_acc, best_val_acc)

    if is_best:
        best_model_file = 'best_model.pth'
        model_file = folderPath + best_model_file
        torch_utils.save_model(model_file, model.state_dict())

    if epoch % args.checkpoint_interval == 0:

        model_file = 'model_' + str(epoch) + '.checkpoint'
        model_file = folderPath + model_file

        # torch.save(model.state_dict(), model_file)
        torch_utils.save(model_file, epoch, model, optimizer, scheduler)
        append_line_to_log('Saved model to ' + model_file + '\n')
    
    with open(experiment_filename, 'r') as f:
        experiment_data = json.load(f)
    
    with open(experiment_filename, 'w') as f:
        experiment_data['history'] = history
        json.dump(experiment_data, f, indent=4)
