
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size = 64
num_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(.1, .1))
])

train_data = datasets.MNIST(root="~/.pytorch/mnist", train=True,  download=True, transform=transforms)
print(f"Size of training dataset: {len(train_data)}")
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data  = datasets.MNIST(root="~/.pytorch/mnist", train=False, download=True, transform=transforms)
print(f"Size of test dataset: {len(test_data)}")
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
        self.conv1   = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu    = nn.ReLU()
        self.pool1   = nn.MaxPool2d(kernel_size=(2, 2));
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(16*14*14, num_classes);

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.fc1(x)
        return x

def train_model(dataloader, model, loss_fn, optim):
    model.train()
    
model = Model().to(device)







