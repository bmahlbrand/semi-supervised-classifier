import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def image_loader(path, batch_size, pinned = False, workers = 0, transform = transforms.Compose([transforms.ToTensor()]), valid_transform=transforms.Compose([transforms.ToTensor()])):

    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=valid_transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=valid_transform)
    
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pinned
    )
    
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pinned
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