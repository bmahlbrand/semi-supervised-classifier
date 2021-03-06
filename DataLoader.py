import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5032, 0.4746, 0.4275),(0.2268, 0.2225, 0.2256))])

def image_loader(path, batch_size, pinned = False, workers = 0, scale_transform = None, augment_transform = transforms.Compose([]), sampler = None):

    if scale_transform is None:
        print('invalid scale transform')

    train_transform = transforms.Compose([
        scale_transform,
        augment_transform,
        normalize
    ])

    valid_transform = transforms.Compose([
        scale_transform,
        normalize
    ])

    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=train_transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=valid_transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=train_transform)
    
    if sampler != None:
        shuffle = False
    else:
        shuffle = True
        
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pinned,
        sampler=sampler
    )
    
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pinned,
    )
    
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pinned
    )

    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup

#TODO visualize classes
if __name__ == '__main__':
    print('visualizing classes')