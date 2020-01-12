import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

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
        self.fc1 = nn.Linear((32 * 3) + 4, 48)
                        # neurons from conv layer + neurons to enter compass info
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, n_actions)

    def forward(self, x, compass):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(len(x), 32 * 3)
        x = torch.cat([x, compass], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1) #dim = 0 since array was flattened
        return x

class NNagent(Agent):

    def __init__(self, GG=None, parent=None, max_steps=1000, actions=6, depth=13):
        
        self.compass_info = []
        
        # GG exists, use it.
        if GG:
                super(NNagent, self).__init__(GG, max_steps)
                
                if parent:
                    self.nn = deepcopy(parent)
                else:
                    self.nn = Net(n_actions=self.action_space,
                              depth=self.depth)
                    
        
        # GG does not exist, That's okay       
        else:
            
            # we cannot rely on the env to have this information
            self.orientation = np.eye(4, dtype=int)
            self.prev_move = 4
            self.rotating_actions = [1, 2, 3, 4]
            
            if parent:
                self.nn = deepcopy(parent)
                
            # 
            else:    
                self.nn = Net(n_actions=actions,
                                  depth=depth)
       
        self.nn.double()
        self.compass_info = np.array([0, 0, 0, 1])

    def evaluate(self, env=None, rl=False):
        """Run self agent on current generator level.
        """
        if env == None:
            env = self.env
        return np.sum(super().evaluate(env, rl=rl))

    def mutate(self, mutationRate):
        childGG = self.env.mutate(mutationRate)
        return NNagent(childGG, parent=self.nn)
    
    def __getstate__(self):
        base = super().__dict__
        base['state_dict'] = self.nn.state_dict()
        return base
    
    def __setstate__(self, d):
        self.__dict__ = d
        self.nn = Net(d['action_space'], d['depth'])
        self.nn.load_state_dict(d['state_dict'])
        self.nn.double()
        del self.__dict__['state_dict']
        
    
    def get_action(self, state, compass):
        """Select an action by running a tile-input through the neural network.

        :param state: tile-grid; numpy tensor
        :return: int of selected action
        """
  
        with torch.no_grad():
            outputs = self.nn(state, compass).squeeze()
            
            _, predicted = torch.max(outputs, 0)
        
        # break data out of tensor
        action = predicted.data.numpy()
        
        # update orientation
        if not hasattr(self, '_env') and action != self.prev_move and action in self.rotating_actions:
            self.prev_move = action
        
        return action, _, _
   
    def rl_get_action(self, state, compass):
        """Select an action by running a tile-input through the neural network.

        :param state: tile-grid; numpy tensor
        :return: int of selected action
        """

        logits = self.nn(state, compass)
        
        probs = Categorical(logits=logits)
        action = probs.sample()
        
        if state.shape[0] == 1:
            a2 = action.item()

            # update orientation
            if not hasattr(self, '_env') and a2 != self.prev_move:
                if a2 in self.rotating_actions:
                    self.prev_move = action
                else:
                    pass
                self.compass_info = self.orientation[self.prev_move - 1]

            elif hasattr(self, '_env'):
                self.compass_info = self.env.orientation[self.env.prev_move - 1]
        
        return action, -probs.log_prob(action), probs.entropy()        
