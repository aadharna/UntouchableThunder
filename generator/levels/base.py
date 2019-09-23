import numpy as np
import os
from pprint import pprint
import pandas as pd

from gym_gvgai import dir

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
    def __init__(self, tile_world, path=dir+'envs/games/zelda_v0/', mechanics=[], generation=0):
        """

        :param tile_world: 2d numpy array of map
        :param path: gym_gvgai.dir
        :param mechanics: list of sprites you would like to be able to mutate into
        """
        self._length = tile_world.shape[0]
        self._height = tile_world.shape[1]
        self._tile_world = tile_world
        self.mechanics = mechanics
        #make folder in levels folder
        self.base_path = path
        if not os.path.exists(os.path.join(self.base_path, 'levels')):
            os.mkdir(os.path.join(self.base_path, 'levels'))
        self.base_path = os.path.join(self.base_path, 'levels')

        self.chars = np.unique(np.unique(self.tile_world).tolist() + self.mechanics)
        self.mechanics = mechanics

        self._fit = 0
        self.generation = generation


    @property
    def tile_world(self):
        return self._tile_world

    def cleanup(self):
        """remove generated/saved files.
        SHOULD ONLY BE CALLED ONCE (which actually tells me that this shouldn't live in the Generator class).

        :return:
        """
        for fname in os.listdir(self.base_path):
            os.remove(os.path.join(self.base_path, fname))
        os.rmdir(self.base_path)


    def to_file(self, env_id, game='zelda'):
        path = os.path.join(self.base_path, f"{game}_id:{env_id}_g:{self.generation}.txt")
        with open(path, 'w+') as fname:
            fname.write(str(self))
            self.path_to_file = path
            # np.save(f"{path.split('.')[0]}.npy", self.tile_world)
        return path

    def mutate(self, mutationRate):
        """randomly edit parts of the level!
        :param mutationRate:
        :return: n/a
        """
        # self.generation += 1
        
        for i in range(self._length): 
            height = np.random.choice(np.arange(self._height))
            if np.random.rand() < mutationRate:
                self._tile_world[i][height] = np.random.choice(self.chars)

    def crossOver(self, parent):
        """Edit levels via crossover rather than mutation
        :param self: parent level A
        :param parent: parent level B
        :return: child level
        """

        child = Generator(tile_world= self.tile_world, 
                          mechanics = self.mechanics, 
                          generation= self.generation + 1)
        
        for i in range(len(child._tile_world)):
            for j in range(len(child._tile_world[i])):
                if np.random.choice([0, 1]):
                    child._tile_world[i][j] = self._tile_world[i][j]
                else:
                    child._tile_world[i][j] = parent._world[i][j]

        return child

    def __str__(self):
        stringrep = ""
        for i in range(len(self._tile_world)):
            for j in range(len(self._tile_world[i])):
                stringrep += self._tile_world[i][j]
                if j == (len(self._tile_world[i]) - 1):
                    stringrep += '\n'
        return stringrep


def _initialize(path):
    """build numpy array of level from txt file

    :param path: path to txt file representation of level
    :return:
    """
    f = open(path, 'r')
    f = f.readlines()
    rep = []
    for l in f:
        rep.append(l[:-1]) #cut off '\n'
    mat = []
    for r in rep:
        for s in r:
            mat.append(s)
    npa = np.array(mat).reshape((9, -1)) # make into numpy array 9x13
    return npa