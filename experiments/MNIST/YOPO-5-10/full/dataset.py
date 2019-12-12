import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data


def create_train_dataset(batch_size=512, root='./data'):

    transform_train = transforms.Compose([
     transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    print('Full: {}'.format(len(train_set.targets)), train_set.data.shape)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def create_test_dataset(batch_size=512, root='./data'):
    transform_test = transforms.Compose([
     transforms.ToTensor(),
    ])

    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    print('Full: {}'.format(len(test_set.targets)), test_set.data.shape)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


if __name__ == '__main__':
    print(create_train_dataset())
    print(create_test_dataset())

