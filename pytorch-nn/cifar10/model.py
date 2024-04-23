
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

test_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

train_data = datasets.CIFAR10(root="~/.pytorch/cifar10", train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root="~/.pytorch/cifar10", train=False, download=True, transform=test_transforms)

print("Training Data\n", train_data)
print("Test Data\n", test_data)

trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=True)

data_batch, labels_batch = next(iter(trainloader))





