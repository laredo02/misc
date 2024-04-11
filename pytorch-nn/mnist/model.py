
import torch
from torch import optim
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

batch_size = 64
num_classes = 10
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(.1, .1))
])

train_data = datasets.MNIST(root="~/.pytorch/mnist", train=True,  download=True, transform=transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data  = datasets.MNIST(root="~/.pytorch/mnist", train=False, download=True, transform=transforms)
train_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.flatten()
for i in range(9):
    image, label = train_data[i]
    image = image.squeeze().numpy()

    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('on')

plt.tight_layout()
plt.show()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        

    def forward(self, x):
        
        return x

def train_model(dataloader, model, loss_fn, optim):
    model.train()
    size = len(dataloader.dataset)
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    
model = Model()
loss_fn = nn.CrossEntropyLoss()
optim = optim.SGD(model.parameters(), lr=learning_rate)

train_model(train_loader, model, loss_fn, optim)











