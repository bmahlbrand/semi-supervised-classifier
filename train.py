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
import torchvision.transforms as transforms

from utils.config_utils import load_config
from utils.Timer import Timer
from utils.fs_utils import create_folder
from utils.logger import Logger
from utils.AverageMeter import AverageMeter
from utils import torch_utils

def append_line_to_log(line = '\n'):
    with open(logPath, 'a') as f:
        f.write(line + '\n')

folderPath = 'checkpoints/session_' + Timer.timeFilenameString() + '/'
create_folder(folderPath)

logPath = 'log/log_' + Timer.timeFilenameString()

parser = argparse.ArgumentParser(description='PyTorch training script for semi-supervised classifier')

## hyperparameters
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='batch size (default: 8)')
parser.add_argument('--learning-rate', default=1e-4, type=float, metavar='L', help="initial learning rate")
parser.add_argument('--seed', type=int, default=0xDEADBEEF, metavar='S', help='random seed (default: (0xDEADBEEF)')
parser.add_argument('--epochs', type=int, default=50, metavar='E', help='training iterations')
# parser.add_argument('--optimizer', type=str, default='sgd', metavar='O', help='which optimizer to use? supported types: [sgd, adam]')

## scheduler
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
parser.add_argument('--cuda', default=False, type=bool, metavar='C', help='use cuda or not (default: true)')
parser.add_argument('--pinned-memory', default=False, type=bool, metavar='P', help='use memory pinning or not (default: true)')
parser.add_argument('--workers', default=0, type=int, metavar='W', help='workers (default: 0)')
parser.add_argument('--train_dir', default='data', type=str, metavar='PATHT', help='path to latest checkpoint (default: data folder)')
parser.add_argument('--val_dir', default='data', type=str, metavar='PATHV', help='path to latest checkpoint (default: data folder)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

args = parser.parse_args()

print(args)

if args.config:
    with open(args.config, 'r') as f:
        args.__dict__ = json.load(f)

experiment_filename = 'experiments/experiment_' + Timer.timeFilenameString() + '.json'

# save parameters of this experiment for reproduction later
with open(experiment_filename, 'w') as f:
    config = args.__dict__
    config['logpath'] = logPath
    config['best_model'] = os.path.join(folderPath, 'best_model.cpkt')
    config['checkpoints'] = folderPath

    json.dump(config, f, indent=4)

# args = args.__dict__

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
    return train_loss.avg()

def validation(model, criterion, loader, device, log_callback):
    end = time.time()
    model.eval()

    validation_loss = 0.0
    correct = 0

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
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= float(len(val_loader.dataset))
        validation_acc  = float(correct) / float(len(val_loader.dataset))
        log_callback('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * validation_acc))
        
        # records essential information into log file.
        log_callback('epoch: {0}\t'
                'Time {batch_time.sum:.3f}s / {1} epochs, ({batch_time.avg:.3f})\t'
                'Data load {data_time.sum:.3f}s / {1} epochs, ({data_time.avg:3f})\n'
                'Average Validation Loss = {loss:.8f}, Accuracy: {correct:3d}/{size:3d} ({acc:.4f}%)\n'.format(
            epoch, batch_idx, batch_time=batch_time,
            data_time=data_time, 
            loss=loss.item(), correct=correct, size=len(val_loader.dataset),
            acc=100. * validation_acc))
        
        log_callback()
        
        # log_callback('Loss{0} = {loss:.8f}\t'
        #         .format(1, loss=loss.item()))

        log_callback(Timer.timeString())

        batch_time.reset()
         
        return validation_loss, validation_acc

start_epoch = 1

import torchvision.models as models
# resnet18 = models.resnet18()
# alexnet = models.AlexNet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
densenet = models.densenet121()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# model = densenet

from modules.VGG import vgg11

model = vgg11()

optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.mode, factor=args.factor, patience=args.factor, verbose=args.verbose,
            threshold=args.threshold, threshold_mode=args.threshold_mode, cooldown=args.cooldown, min_lr=args.min_lr, eps=args.eps)

criterion = nn.NLLLoss()

if args.resume:
    start_epoch, model, optimizer, scheduler = torch_utils.load(args.resume, model, optimizer, start_epoch, scheduler)
    # append_line_to_log('resuming ' + args.resume + '... at epoch ' + str(start_epoch))

train_loader, val_loader, unsup_loader = image_loader('data', args.batch_size, args.pinned_memory, args.workers, transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5032, 0.4746, 0.4275),(0.2268, 0.2225, 0.2256))]))

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

early_stop = EarlyStopping(patience=7)

for epoch in range(start_epoch, args.epochs + 1):

    training_loss = train(epoch, model, optimizer, criterion, train_loader, device, append_line_to_log)
    val_loss, val_acc = validation(model, criterion, val_loader, device, append_line_to_log)
    
    history['training_loss'].append(training_loss)
    history['validation_loss'].append(val_loss)
    history['validation_accuracy'].append(val_acc)

    scheduler.step(val_loss)

    early_stop(val_loss)

    if early_stop.early_stop:
        append_line_to_log("Early stopping")
        break

    # # is_best = val_loss < best_val_loss
    is_best = val_acc > best_val_acc

    # best_val_loss = min(val_loss, best_val_loss)
    best_val_acc = max(val_acc, best_val_acc)

    if is_best:
        best_model_file = 'best_model.pth'
        model_file = folderPath + best_model_file
        torch.save(model.state_dict(), model_file)

    model_file = 'model_' + str(epoch) + '.pth'
    model_file = folderPath + model_file

    torch.save(model.state_dict(), model_file)
    append_line_to_log('Saved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` \n')
    
    with open(experiment_filename, 'r') as f:
        experiment_data = json.load(f)
    
    with open(experiment_filename, 'w') as f:
        experiment_data['history'] = history
        json.dump(experiment_data, f, indent=4)
# plt.plot(range(len(history['losses'])), history['losses'], 'g-')
# plt.xlabel('batch steps')
# plt.ylabel('test loss')
# plt.savefig('test_loss.png')

# plt.plot(range(args.epochs), history['validation_accuracy'], 'r-')
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.savefig('valid_accuracy.png')