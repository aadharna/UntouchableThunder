import numpy as np
from pprint import pprint
import pandas as pd

CHARS = ['#',  # ground
         'G',  # goomba
         'C',  # coin
         'P',  # piranha plant
         ' ']  # gap

IS_VIABLE_SCORE = 100
WIDE = 25
TALL = 5
# STATIC_WORLD_TOP = np.array([['']*WIDE]*3, ndmin=2)
# STATIC_WORLD_TOP[:, -1] = 'F' # flag/end

class generator:
    def __init__(self):
        self._length = WIDE
        self._height = TALL
        self._world = np.array([['']*self._length]*self._height, ndmin=2)
        self._fit = 0

    @property
    def world(self):
        return self._world
        # return np.concatenate((STATIC_WORLD_TOP, self._world), axis=0)

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
            height = np.random.choice(np.arange(5))
            if np.random.rand() < mutationRate:
                self._world[height][i] = np.random.choice(CHARS)

        self.apply_gravity()

        # add static elements (end goal/mario starting position)
        self._world[:, -1] = 'F'  # flag/end
        self._world[-2][0] = 'M'

    def apply_gravity(self):
        """ensure goombas, piranha plants etc are not floating

        :return:
        """
        for j in range(self._length):
            for i in range(self._height - 2):
                if self._world[i][j] in CHARS:
                    self._world[i+1][j] = self._world[i][j]
                    self._world[i][j] = ''

    def crossOver(self, parent):
        """Edit levels via crossover rather than mutation

        :param self: parent level A
        :param parent: parent level B
        :return: child level
        """
        pass

x = generator()

x._world[-1][:] = CHARS[0] # set ground
x.mutate(0.25)
print(pd.DataFrame(x.world))
