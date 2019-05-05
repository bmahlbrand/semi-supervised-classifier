from collections import OrderedDict

import torch

def save(filename, epoch, model, optimizer, scheduler = None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
        # ...
    }
    torch.save(state, filename)

def load(filename, model, optimizer, epoch, scheduler = None):
    
    state = torch.load(filename)

    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch'] + 1
    scheduler.load_state_dict(state['scheduler'])

    return epoch, model, optimizer, scheduler

#with no code...
def save_model(filename, model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model, filename)

#with no code...
def load_model(filename):
    model = torch.load(filename)
    return model

def fix_state_dict(model, loaded_state_dict):
    own_state = model.state_dict()
    new_state = OrderedDict()
    for name, param in loaded_state_dict.items():
        if name not in own_state:
            na = name.replace("module.", "")
            new_state[na] = param