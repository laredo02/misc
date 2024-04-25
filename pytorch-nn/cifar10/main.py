
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def getDataLoaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root="~/.pytorch/", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root="~/.pytorch/", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return  train_loader, test_loader


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,   out_channels=16,  kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,  out_channels=32,  kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32,  out_channels=64,  kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128,  out_channels=256,  kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activ = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.Flatten(),

            #nn.Linear(in_features=128*8*8, out_features=2048),
            nn.Linear(in_features=256*4*4, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=1024, out_features=10),
        )

    def forward(self, x):
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2(x))
        x = self.pool1(x)

        x = self.activ(self.conv3(x))
        x = self.activ(self.conv4(x))
        x = self.pool2(x)

        x = self.activ(self.conv5(x))
        x = self.pool3(x)

        x = self.classifier(x)
        return x


def train_model(dataloader, model, criterion, optim, device, epochs=10):
    model.train()
    for i in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            if batch%100 == 0:
                print(f"{loss.item()}, {batch}")
        print()


def test_model(dataloader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            total_correct += (predicted == y).sum().item()
            total_images += y.size(0)

        average_loss = total_loss/len(dataloader)
        acc = (total_correct/total_images)*100
        print(f"acc = {acc}")
        print(f"avg_loss = {average_loss}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = Model().to(device)
train_loader, test_loader = getDataLoaders(64);

criterion = torch.nn.CrossEntropyLoss()




optim = torch.optim.Adam(model.parameters(), 0.001)
train_model(train_loader, model, criterion, optim, device, 12)
test_model(test_loader, model, criterion, device)












