import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import neat
import os

from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.recurrent_net import RecurrentNet

import numpy as np
from copy import deepcopy


from agent.base import Agent


class Net(nn.Module):
    def __init__(self, n_actions, depth):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=depth, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
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



# def make_env():
#     return gym.make("CartPole-v0")


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs[:, 0] > 0.5




class NNagent(Agent):

    def __init__(self, GG, parent=None, max_steps=250):
        super(NNagent, self).__init__(GG, max_steps)

        if parent:
            self.nn = deepcopy(parent)
        
        else:
            self.nn = Net(n_actions=self.action_space,
                          depth=self.env.depth)

        self.nn.double()



    def update(self):
        """Update network. neuroevolution.

        :return:
        """
        # Load the config file, which is assumed to live in
        # the same directory as this script.
        config_path = os.path.join(
            '/home/aadharna/PycharmProjects/ThesisResearch/UntouchableThunder/agent/', "neat.cfg")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        evaluator = MultiEnvEvaluator(
            make_net, activate_net, max_env_steps=self.max_steps, envs=[self._env]
        )

        def eval_genomes(genomes, config):
            for _, genome in genomes:
                genome.fitness = evaluator.eval_genome(genome, config)

        pop = neat.Population(config)
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        reporter = neat.StdOutReporter(True)
        pop.add_reporter(reporter)
        logger = LogReporter("neat.log", evaluator.eval_genome)
        pop.add_reporter(logger)

        best = pop.run(eval_genomes, 5)
        self.current_genome = best

        self.neat = make_net(best, config, 1)



    def mutate(self, mutationRate):
        childGG = self.env.mutate(mutationRate)
        return NNagent(childGG, parent=self.nn)
    
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

