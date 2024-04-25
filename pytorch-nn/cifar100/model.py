
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,   out_channels=16,  kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,  out_channels=32,  kernel_size=(3, 3), stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32,  out_channels=64,  kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=(3, 3), stride=1, padding=1)

        self.activ = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*8*8, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=100),
        )

    def forward(self, x):
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2(x))
        x = self.pool(x)

        x = self.activ(self.conv3(x))
        x = self.activ(self.conv4(x))
        x = self.pool(x)

        x = self.classifier(x)
        return x


def getModel(device):
    model = Model().to(device)
    return model





