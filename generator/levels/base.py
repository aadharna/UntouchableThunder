import numpy as np
from pprint import pprint
import pandas as pd

CHARS = ['#',  # ground
         'G',  # goomba
         'C',  # coin
         'P',  # piranha plant
         ' ']  # gap
GROUND_CHARS = ['G', 'P']

IS_VIABLE_SCORE = 100
# STATIC_WORLD_TOP = np.array([['']*WIDE]*3, ndmin=2)
# STATIC_WORLD_TOP[:, -1] = 'F' # flag/end

class Generator:
    def __init__(self, world, mechanics=None):
        self._length = world.shape[0]
        self._height = world.shape[1]
        self._world = world

        self.chars = np.unique(np.unique(self.world).tolist() + mechanics)
        self.mechanics = mechanics

        self._fit = 0

    @property
    def world(self):
        return self._world

    def fitness(self, gym, agent):
        """Score agent by having it try to complete the level.
        :param gym: gvgai_gym env.
        :param agent: NN-agent
        :return:
        """
        # gym.unwrapped.setLevel(self.world)
        # gym.reset()
        score = agent.evalute_in(gym)
        self._fit = score / IS_VIABLE_SCORE

    def mutate(self, mutationRate):
        """randomly edit parts of the level!
        :param mutationRate:
        :return: n/a
        """
        for i in range(self._length): 
            height = np.random.choice(np.arange(self._height))
            if np.random.rand() < mutationRate:
                self._world[i][height] = np.random.choice(self.chars)

        # self.apply_gravity()

        # add static elements (end goal/mario starting position)
        # self._world[:, -1] = 'F'  # flag/end
        # self._world[-2][0] = 'M'

    def apply_gravity(self):
        """ensure goombas, piranha plants etc are not floating
        :return:
        """
        for j in range(self._length):
            for i in range(self._height - 2):
                if self._world[i][j] in GROUND_CHARS and not self._world[i+1][j] in CHARS:
                    self._world[i+1][j] = self._world[i][j]
                    self._world[i][j] = ''

    def crossOver(self, parent):
        """Edit levels via crossover rather than mutation
        :param self: parent level A
        :param parent: parent level B
        :return: child level
        """

        child = Generator(self.world, self.mechanics)
        for i in range(len(child._world)):
            for j in range(len(child._world[i])):
                if np.random.choice([0, 1]):
                    child._world[i][j] = self._world[i][j]
                else:
                    child._world[i][j] = parent._world[i][j]

        return child

    def __str__(self):
        stringrep = ""
        for i in range(len(self._world)):
            for j in range(len(self._world[i])):
                stringrep += self._world[i][j]
                if j == (len(self._world[i]) - 1):
                    stringrep += '\n'
        return stringrep