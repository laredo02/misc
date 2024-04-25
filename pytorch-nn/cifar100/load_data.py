
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def getDataLoaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100(root="~/.pytorch/", train=True, download=True, transform=transform)
    Xtrain, ytrain = train_dataset[0]
    print(Xtrain.shape, ytrain)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = datasets.CIFAR100(root="~/.pytorch/", train=False, download=True, transform=transform)
    Xtest, ytest = test_dataset[0]
    print(Xtest.shape, ytest)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader

