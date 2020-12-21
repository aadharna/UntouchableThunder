import torch
from torch.distributions.categorical import Categorical

import numpy as np
from copy import deepcopy

from agent.base import BaseAgent
from agent.models import Net
from utils.utils import add_noise


class OrientationConditionedNNagent(BaseAgent):

    def __init__(self, time_stamp, GG=None, parent=None, prefix='.', actions=6, depth=13, master=True):
        
        self.compass_info = []
        
        # GG exists, use it.
        if GG:
                super(OrientationConditionedNNagent, self).__init__(GG, time_stamp, prefix=prefix, master=master)
                
                if parent:
                    self.nn = deepcopy(parent)
                else:
                    self.nn = Net(n_actions=self.action_space,
                              depth=self.depth)
                    
        
        # GG does not exist, That's okay
        # This is useful for vectorized environment evolution evaluation
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

        """Run self agent on current generator level. 
                """
        self.images = []
        # print("evaluating agent")
        done = False
        rewards = []
        state = add_noise(env.reset()) if self.noisy else env.reset()
        while not done:
            c = env.orientation[env.prev_move - 1]
            state = torch.DoubleTensor(np.array([state]))
            c = torch.DoubleTensor(np.array([c]))

            action, nlogpob, ent = self.get_action(state, c) if not rl else self.rl_get_action(state, c)

            # todo: FIGURE OUT HOW TO MAKE IT SO THAT I DON'T HAVE TO INT(ACTION).
            state, reward, done, info = env.step(int(action))
            if self.noisy:
                state = add_noise(state)
            # print(f"action: {action}, done: {done}, reward: {reward}")
            # state is a grid world here since we're using GridGame class
            rewards.append(reward)
            if self.vis:
                self.images.append(info['pic'])
                self.vis(env.env, action, image=info['pic'])

        self.won = info['won']

        # self.update_score(np.sum(rewards))
        # print("evaluated")
        # print(len(rewards))
        # if the user wants to do another noisy trial,
        # let them request it again.
        self.noisy = False
        return np.sum(rewards)

    def fitness(self, noisy=False, fn=None, rl=False):
        """run this agent through the current generator env once and store result into
        """
        self.noisy = noisy
        self.vis = fn
        return self.env.fitness(self, rl=rl)

    def mutate(self, mutationRate):
        childGG = self.env.mutate(mutationRate)
        return OrientationConditionedNNagent(time_stamp=self.unique_run_id,
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
        
        return action, 1, 0
   
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
