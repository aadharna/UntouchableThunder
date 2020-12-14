import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

import os

import numpy as np
from copy import deepcopy

from agent.base import Agent
from agent.models import Net
# from agent.models import Actor

class NNagent(Agent):

    def __init__(self, time_stamp, GG=None, parent=None, prefix='.', actions=6, depth=13, master=True):
        
        self.compass_info = []
        
        # GG exists, use it.
        if GG:
                super(NNagent, self).__init__(GG, time_stamp, prefix=prefix, master=master)
                
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
        return NNagent(time_stamp=self.unique_run_id,
                       prefix=self.prefix,
                       GG=childGG,
                       parent=self.nn)
    
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
            outputs = self.nn(state, compass).flatten()
            
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
                
                self.compass_info = self.orientation[self.prev_move - 1]

            elif hasattr(self, '_env'):
                self.compass_info = self.env.orientation[self.env.prev_move - 1]
        
        else:
            actions = action.data.numpy()
            for act in actions:
                pass
            
        return action, -probs.log_prob(action), probs.entropy()        
