import os
import time
import numpy as np
from utils.utils import load_obj
from utils.loader import load_from_yaml
from generator.levels.base import BaseGenerator, _initialize


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

        # self.generate(params=[self.diff], difficulty=True, env_id=self.env_id)

        # self.chars = np.unique(np.unique(self.tile_world).tolist() + self.mechanics)
        # self.chars = list(set(self.chars) - {'A'}) # do not place more agents

    def to_file(self, netId, game):
        # level = _initialize(path)
        # with open(path, 'w+') as fname:
        #     level[level == '1'] = '3'
        #     level[level == '2'] = '3'
        #     fname.write(self.to_string(level))
        # print(self.path_to_file)
        return self.path_to_file

    def generate(self, params=[], **kwargs):

        self.num_samples += 1
        env_id = kwargs['env_id'] if 'env_id' in kwargs else self.id
        # prefix = kwargs['path'] if 'path' in kwargs else self._path
        # print(self.prefix)
        name = os.path.join(self.run_folder, str(env_id), 'levels', f"sample:{self.num_samples}")
        if self.new:
            name += f"_{int(time.time())}"
            self.new = False
        if params:
            name += "_dif:" + str(round(params[0], 2))
            params = ["difficulty"] + params + [self._height, self._length]
        else:
            params = [self._height, self._length] + params
        params = [str(param) for param in params]
        param_str = " ".join(params)
        file = name + ".txt"
        # print(file)
        os.system("node " + self.script + " " + self.jsGame + " " + file + " " + param_str)
        time.sleep(0.1)
        while not os.path.exists(file):
            time.sleep(0.2)
        path = os.path.abspath(file)

        # set path to file.
        self.path_to_file = str(path)
        level = _initialize(path)
        with open(path, 'w+') as fname:
            level[level == '1'] = '3'
            level[level == '2'] = '3'
            fname.write(self.to_string(level))

        self.string = self.to_string(level)
        return path

    def update_from_lvl_string(self, new_lvl):
        return

    def mutate(self, mutationRate, minimal=False, r=None, **kwargs):

        return self.generate(params=[r], difficulty=True, **kwargs)

    def to_string(self, tile_world):
        stringrep = ""
        for i in range(len(tile_world)):
            for j in range(len(tile_world[i])):
                stringrep += tile_world[i][j]
                if j == (len(tile_world[i]) - 1):
                    stringrep += '\n'

        return stringrep

    def __str__(self):
        diff = self.diff if self.diff else np.random.rand()
        self.generate(params=[diff], env_id=self.env_id)
        return self.string
