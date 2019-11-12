import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

import numpy as np
from copy import deepcopy

from agent.base import Agent

class Net(nn.Module):
    def __init__(self, n_actions, depth):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 3, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, n_actions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1) #dim=0 --> [1, 1, 1, 1, ..., 1]
        return x

class NNagent(Agent):

    def __init__(self, GG, parent=None, max_steps=1000):
        super(NNagent, self).__init__(GG, max_steps)

        if parent:
            self.nn = deepcopy(parent)
        
        else:
            self.nn = Net(n_actions=self.action_space,
                          depth=self.env.depth)

        self.nn.double()

    def evaluate(self, env=None):
        """Run self agent on current generator level.
        """
        if env == None:
            env = self.env
        return -1*np.sum(super().evaluate(env))

    def mutate(self, mutationRate):
        childGG = self.env.mutate(mutationRate)
        return NNagent(childGG, parent=self.nn)
    
    def get_action(self, state):
        """Select an action by running a tile-input through the neural network.

        :param state: tile-grid; numpy tensor
        :return: int of selected action
        """
        # the grid needs to be part of a 'batch', so we make state the only element in a list.
        inp = Variable(torch.from_numpy(np.array([state])), requires_grad=False)
        outputs = self.nn(inp.double())
        _, predicted = torch.max(outputs, 1)
        # break data out of tensor
        return predicted.data.numpy()[0]

