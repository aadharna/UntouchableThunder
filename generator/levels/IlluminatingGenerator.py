import os
from utils.utils import load_obj
from utils.loader import load_from_yaml
from generator.levels.base import BaseGenerator


class IlluminatingGenerator(BaseGenerator):
    id = 0

    def __init__(self, tile_world,
                 shape,
                 args_file='./args.yml',
                 path='/envs/games/zelda_v0/',
                 mechanics=[],
                 generation=0,
                 locations={},
                 game='dzelda'):
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
        self._length = shape[0]
        self._height = shape[1]

        self.BOUNDARY = load_obj(path, f'{self.game}_boundary.pkl')

        self._tile_world = tile_world

        self.mechanics = self.args.mechanics
        # make folder in levels folder
        self.base_path = path
        self._path = os.path.join(self.base_path, f'{self.game}_poet_levels')
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self.generation = generation

        self.id = IlluminatingGenerator.id
        IlluminatingGenerator.id += 1

        self.script = os.path.dirname(os.path.realpath(__file__)) + '/lib/gvgai_generator/app_v3.js'

        self.num_samples = 0

        # self.chars = np.unique(np.unique(self.tile_world).tolist() + self.mechanics)
        # self.chars = list(set(self.chars) - {'A'}) # do not place more agents

    def generate(self, params=[], difficulty=None, **kwargs):

        self.num_samples += 1
        env_id = kwargs['kwargs']['env_id'] if 'env_id' in kwargs['kwargs'] else self.id
        name = f"{self.game}_sample:{self.num_samples}_id:{env_id}"
        if difficulty is not None:
            name += "_dif:" + str(round(params[0], 2))
            params = ["difficulty"] + params + [self._length, self._height]
        else:
            params = [self._length, self._height] + params
        params = [str(param) for param in params]
        param_str = " ".join(params)
        file = os.path.join(self.base_path, name + ".txt")
        os.system("node " + self.script + " " + self.game + " " + file + " " + param_str)
        path = os.path.abspath(file)
        return path

    def mutate(self, mutationRate, minimal=False, r=None, **kwargs):

        return self.generate(params=[r], difficulty=True, kwargs=kwargs)