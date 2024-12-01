#!../torch-venv/bin/python3

import torch
from torch import optim
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(.1, .1))
])

batch_size=32
train_data = datasets.MNIST(root="~/.pytorch/mnist", train=True,  download=True, transform=transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root="~/.pytorch/mnist", train=False, download=True, transform=transforms)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

##########################################################
#fig, axes = plt.subplots(3, 3, figsize=(9, 9))
#axes = axes.flatten()
#for i in range(9):
#    image, label = train_data[i]
#    image = image.squeeze().numpy()
#
#    axes[i].imshow(image, cmap='gray')
#    axes[i].set_title(f'Label: {label}')
#    axes[i].axis('on')
#
#plt.tight_layout()
#plt.show()
##########################################################

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


def train_model(dataloader, model, loss_fn, optim):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if (batch%100==0):
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

model = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optim = optim.SGD(model.parameters(), lr=0.001)
    
epochs = 1
for i in range(epochs):
    print(f"Epoch{i}---------------------------")
    train_model(train_loader, model, loss_fn, optim)
print("Done")



