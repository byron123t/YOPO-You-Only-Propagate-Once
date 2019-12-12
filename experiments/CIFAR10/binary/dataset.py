import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np


def my_mapper(item):
    if item in [0, 1, 8, 9]:
        return 0
    if item in [2, 3, 4, 5, 6, 7]:
        return 1
    return item


def create_train_dataset(batch_size=128, root='./data'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    print('Full: {}'.format(len(train_set.targets)), train_set.data.shape)

    train_set.targets = torch.tensor(list(map(my_mapper, train_set.targets)))
    print('Mapped: {}'.format(len(train_set.targets)), train_set.data.shape)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def create_test_dataset(batch_size=128, root='./data'):
    transform_test = transforms.Compose([
     transforms.ToTensor(),
    ])

    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    print('Full: {}'.format(len(test_set.targets)), test_set.data.shape)

    test_set.targets = torch.tensor(list(map(my_mapper, test_set.targets)))
    print('Mapped: {}'.format(len(test_set.targets)), test_set.data.shape)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


if __name__ == '__main__':
    print(create_train_dataset())
    print(create_test_dataset())
