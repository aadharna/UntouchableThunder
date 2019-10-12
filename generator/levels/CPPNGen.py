import os
import numpy as np


import neat

from gym_gvgai import dir
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.activations import tanh_activation
from utils.utils import eval_cppn_genome


from generator.levels.base import Generator



def make_net(genome, config, input_coords, output_coords, input_names, output_names):
    return AdaptiveLinearNet.create(
                                    genome,
                                    config,
                                    input_coords=input_coords,
                                    output_coords=output_coords,
                                    input_names=input_names,
                                    output_names=output_names,
                                    weight_threshold=0.4,
                                    batch_size=1,
                                    activation=tanh_activation,
                                    output_activation=tanh_activation,
                                    device="cpu",
                                )



class CPPNGen(Generator):
    def __init__(self, tile_world, path=dir+'/envs/games/zelda_v0/', mechanics=[], generation=0, locations={}):
        super(CPPNGen, self).__init__(tile_world=tile_world,
                                      path=path,
                                      mechanics=mechanics,
                                      generation=generation,
                                      locations=locations)

        self.config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
        g = neat.DefaultGenome

        self.config = neat.Config(
                                    g,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    self.config_path,
                                )
        # Create the population, which is the top-level object for a NEAT run.
        self.p = neat.Population(self.config)

        self.p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.p.add_reporter(stats)
        self.p.add_reporter(neat.Checkpointer(5))

        self.genome = self.p.run(eval_cppn_genome, 1)

        def one_hot(y):
            """one hot encode the sprites

            :param y: list of sprites
            :return:
            """
            return np.eye(y.shape[0])

        self.sprites = np.array(list(set(self.locations)))# - {'A'})
        self.hot_sprites = one_hot(self.sprites)


        self.nn = make_net(self.genome,
                           self.config,
                           [[1, 0], [0, 1]],
                           self.hot_sprites,
                           ['x_in', 'y_in'],
                           self.sprites)

    @property
    def tile_world(self):
        # numpy array
        npa = np.array([['0'] * self._height] * self._length, dtype=str)
        for i in range(1, self._length - 1):
            for j in range(1, self._height - 1):
                output = self.nn([(i, j)])
                cha = self._output_to_char(output)
                npa[i][j] = cha

        for pos in self.BOUNDARY['w']:
            npa[pos[0]][pos[1]] = 'w'
        return npa

    def mutate(self, mutationRate):
        if np.random.rand() < mutationRate:
            self.genome = self.p.run(eval_cppn_genome, 10)

        self.nn = make_net(self.genome,
                           self.config,
                           [[1, 0], [0, 1]],
                           self.hot_sprites,
                           ['x_in', 'y_in'],
                           self.sprites)

    def _output_to_char(self, output):
        return np.random.choice(self.sprites)