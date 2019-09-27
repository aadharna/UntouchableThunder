import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


from agent.base import Agent


class Net(nn.Module):
    def __init__(self, n_actions, depth):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(depth, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 3)
        self.fc1 = nn.Linear(16 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_actions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NNagent(Agent):

    def __init__(self, GG, max_steps=250):
        super(NNagent, self).__init__(GG, max_steps)

        self.nn = Net(n_actions=self.action_space,
                      depth=self.env.depth)

        self.nn.double()

    def update(self):
        """Update network. neuroevolution.

        :return:
        """
        pass

    def get_action(self, state):
        """Select an action by running a tile-input through the neural network.

        :param state: tile-grid; numpy tensor
        :return: int of selected action
        """
        # the grid needs to be part of a 'batch', so we make state the only element in a list.
        input = Variable(torch.from_numpy(np.array([state])), requires_grad=False)
        outputs = self.nn(input)
        _, predicted = torch.max(outputs, 1)
        # break data out of tensor
        return predicted.data.numpy()[0]

