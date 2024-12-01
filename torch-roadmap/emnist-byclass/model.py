#!../torch-venv/bin/python3

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, EMNIST
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def show_sample(sample):
    image, label = sample
    image = image.permute(2, 1, 0)
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

def load_data(batch_size):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_data = EMNIST(root="~/.pytorch/emnist/", split="byclass", train=True, download=True, transform=transforms)
    test_data = EMNIST(root="~/.pytorch/emnist/", split="byclass", train=False, download=True, transform=transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=3)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=3)
    return (train_loader, test_loader)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.pool(self.bn1(self.conv1(x)))
            x = self.pool(self.bn2(self.conv2(x)))
            flatten_size = x.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 62)
        )

    def forward(self, x):
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def train_classifier(model, dataloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss}")

def test_classifier(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted.eq(labels).sum().item()

    acc = 100*(correct/total)
    print(acc)

def main():
    train_loader, test_loader = load_data(batch_size=512)
    model = Discriminator().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_classifier(model, train_loader, criterion, optimizer, 10, device)
    test_classifier(model, test_loader, criterion, device)
    
    torch.save(model.state_dict(), 'model.pth')
    model = Discriminator().to(device)
    model.load_state_dict(torch.load('model.pth', map_location='cuda:0'))
    test_classifier(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()




