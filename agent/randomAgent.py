import os
import numpy as np
from utils.utils import add_noise
from agent.base import BaseAgent

import torch
from torch.distributions import Categorical


class RandomAgent(BaseAgent):
    """
    Wrap each env with a game-playing agent
    """
    agent_count = 0

    def __init__(self, GG, time_stamp, prefix='.', master=True):
        """Wrap environment with a game-playing agent

        :param GG: GridGame Class (contains gym_gvgai env and a level generator)

        """
        super(RandomAgent, self).__init__(GG=GG,
                                          time_stamp=time_stamp,
                                          prefix=prefix,
                                          master=master)

        return

    def evaluate(self, env):
        """Run self agent on current generator level.
        """
        self.images = []
        # print("evaluating agent")
        done = False
        rewards = []
        state = add_noise(env.reset()) if self.noisy else env.reset()
        while not done:
            state = torch.DoubleTensor(np.array([state]))

            action, nlogpob, ent = self.get_action(state)

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
        return rewards

    def mutate(self, mutationRate):
        childGG = self.env.mutate(mutationRate)
        return RandomAgent(childGG)

    def get_action(self, state, c):
        # randomly for now
        probs = Categorical(probs=torch.Tensor([1 / self.action_space for _ in range(self.action_space)]))
        action = probs.sample()
        return action, -probs.log_prob(action), probs.entropy()

    def rl_get_action(self, state, c):
        return self.get_action(state, c)

    def fitness(self, noisy=False, fn=None, rl=False):
        """run this agent through the current generator env once and store result into
        """
        self.noisy = noisy
        self.vis = fn
        return self.env.fitness(self, rl=rl)
