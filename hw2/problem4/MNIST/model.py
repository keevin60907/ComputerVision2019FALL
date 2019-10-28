import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # with 44,426 parameters
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(in_features=256, out_features=120, bias=True)
        self.fc2 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc3 = nn.Linear(in_features=84, out_features=10, bias=True)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def name(self):
        return "ConvNet"

class Fully(nn.Module):
    # with 109,386 parameters
    def __init__(self):
        super(Fully, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=10, bias=True)

    def forward(self, x):
        x = x.view(x.size(0),-1) # flatten input tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def name(self):
        return "Fully"

