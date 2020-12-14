import os
import time
import numpy as np
from utils.utils import load_obj
from utils.loader import load_from_yaml
from generator.levels.base import BaseGenerator, _initialize
from generator.levels.zelda2018NeuripsGenerator import *


class IlluminatingGenerator(BaseGenerator):
    id = 0

    def __init__(self,
                 shape,
                 args_file='./args.yml',
                 path='./levels',
                 generation=0,
                 run_folder='..',
                 **kwargs):
        """

        :param tile_world: 2d numpy array of map
        :param path: gym_gvgai.dir
        :param mechanics: list of sprites you would like to be able to mutate into
        :param generation: int
        """
        super().__init__()

        self.args_file = args_file

        self.args = load_from_yaml(args_file)
        self.floor = self.args.floor[0]

        self.game = self.args.game
        self.shape = shape

        self._length = shape[0]
        self._height = shape[1]

        self.jsGame = 'zelda' if self.game == 'dzelda' else self.game

        self.BOUNDARY = load_obj(path, f'{self.game}_boundary.pkl')

        self.mechanics = self.args.mechanics
        # make folder in levels folder
        self.base_path = path
        self._path = os.path.join(self.base_path, f'{self.game}_poet_levels')
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self.generation = generation

        self.id = IlluminatingGenerator.id
        IlluminatingGenerator.id += 1

        self.script = os.path.dirname(os.path.realpath(__file__)) + '/../../ext/a2c_gvgai/lib/gvgai_generator/app_v3.js'

        self.num_samples = 0
        self.path_to_file = None

        self.diff = kwargs['diff'] if 'diff' in kwargs else 0.05

        self.run_folder = run_folder
        self.env_id = 0
        self.new = True

        if not self.jsGame == 'zelda':
            # import from neurips generator
            raise ValueError("Please use Zelda for illuminating generator. Other games are being tested still")

        self.m = Maze()
        self.gen_Z = Zelda(self.m, charMap)

    def to_file(self, netId, game):
        self.path_to_file = os.path.join(self._path, f"{game}_{netId}_{int(time.time())}.txt")
        with open(self.path_to_file, 'w+') as fname:
            fname.write(self.string)
        return self.path_to_file

    def generate(self, params=[], **kwargs):

        self.string = self.gen_Z.newLevel(round(params[0], 2), self._height, self._length)
        return self.string

    def update_from_lvl_string(self, new_lvl):
        return

    def mutate(self, mutationRate, minimal=False, r=None, **kwargs):

        return self.generate(params=[r], difficulty=True, **kwargs)

    def __str__(self):
        diff = self.diff if self.diff else np.random.rand()
        return self.generate(params=[diff], env_id=self.env_id)
