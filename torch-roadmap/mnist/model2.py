#!../torch-venv/bin/python3

import torch
from torch.nn import Module, Linear, ReLU, Sequential, Flatten, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torchvision import transforms

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
batch_size = 64

transformsv2 = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32)])

train_data = MNIST(
    root="~/datasets/fashion_mnist",
    train=True,
    download=True,
    transform=transforms.ToTensor())

test_data = MNIST(
    root="~/datasets/fashion_mnist",
    train=False,
    download=True,
    transform=transforms.ToTensor())

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class Fmnist(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear_relu_stack = Sequential(
            Linear(28*28, 512), ReLU(),
            Linear(512, 512), ReLU(),
            Linear(512, 512), ReLU(),
            Linear(512, 10), ReLU())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = Fmnist().to(device)
print(model)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

loss_fn = CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i in range(10):
    train(train_dataloader, model, loss_fn, optimizer)


