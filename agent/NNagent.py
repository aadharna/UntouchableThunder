import torch
import torch.nn as nn
import torch.nn.functional as F


from agent.base import Agent


class Net(nn.Module):
    def __init__(self, n_actions, depth):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=6, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_actions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NNagent(Agent):

    def __init__(self, GG, max_steps=250):
        super(NNagent, self).__init__(GG, max_steps)

        self.nn = Net(n_actions=self.action_space,
                      depth=self.env.depth)


    def update(self):
        """Update network. neuroevolution.

        :return:
        """
        pass

    def get_action(self, input=None):
        return self.nn(input)