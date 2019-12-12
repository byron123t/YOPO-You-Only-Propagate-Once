import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import pickle
import numpy as np


def my_mapper(item):
    indices = []
    labels = []
    map_label = {2: 2, 3: 3, 0: 0, 6: 6, 8: 8, 5: 5}
    
    for i, label in enumerate(item.targets.cpu().tolist()):
        if label in [2, 3, 0, 6, 8, 5]:
            indices.append(i)
            labels.append(map_label[label])

    item.targets = labels
    item.data = np.take(item.data, indices, axis=0)
    return item


def create_train_dataset(batch_size=512, root='./data'):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    print('Full: {}'.format(len(train_set.targets)), train_set.data.shape)

    train_set = my_mapper(train_set)
    print('Mapped: {}'.format(len(train_set.targets)), train_set.data.shape)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader


def create_test_dataset(batch_size=512, root='./data'):
    transform_test = transforms.Compose([
     transforms.ToTensor(),
    ])

    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    print('Full: {}'.format(len(test_set.targets)), test_set.data.shape)

    test_set = my_mapper(test_set)
    print('Mapped: {}'.format(len(test_set.targets)), test_set.data.shape)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


def load_test_dataset(batch_size=512, root='./data', natural=False):
    if natural:
        dbfile = open('../binary/natural_indices', 'rb')
    else:
        dbfile = open('../binary/labeled_indices', 'rb')
    db = pickle.load(dbfile)
    print(db.shape)
    dbfile.close()

    transform_test = transforms.Compose([
     transforms.ToTensor(),
    ])

    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    print('Full: {}'.format(len(test_set.targets)), test_set.data.shape)

    test_set.targets = test_set.targets[db]
    test_set.data = test_set.data[db]
    test_set = my_mapper(test_set)
    print('Mapped: {}'.format(len(test_set.targets)), test_set.data.shape)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


if __name__ == '__main__':
    print(create_train_dataset())
    print(create_test_dataset())
