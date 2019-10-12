import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from copy import deepcopy



class Net(nn.Module):
    def __init__(self, n_actions, depth):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 3, 48)
        self.fc2 = nn.Linear(84, 24)
        self.fc3 = nn.Linear(24, n_actions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


PROCESSOR_NETWORK = Net(6, 13)