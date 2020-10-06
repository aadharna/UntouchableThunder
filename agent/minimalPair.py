import os
from copy import deepcopy
from agent.models import Net
from generator.levels.EvolutionaryGenerator import EvolutionaryGenerator
from generator.levels.IlluminatingGenerator import IlluminatingGenerator

class MinimalPair():
    id = 0
    def __init__(self, unique_run_id,
                 game='dzelda',
                 generatorType='evolutionary',
                 generator=None,
                 parent=None,
                 prefix='..',
                 actions=6,
                 depth=13):

        self.unique_run_id = unique_run_id
        self.game = game

        if parent:
            self.nn = deepcopy(parent)
        else:
            self.nn = Net(n_actions=actions,
                          depth=depth)

        self.generatorType = generatorType
        self.generator = generator

        self.id = MinimalPair.id
        MinimalPair.id += 1

        run_folder = f'{prefix}/results_{self.unique_run_id}/'

        if not os.path.exists(run_folder):
            os.mkdir(run_folder)
        self.agent_folder = os.path.join(run_folder, str(self.id))
        if not os.path.exists(self.agent_folder):
            os.mkdir(self.agent_folder)
        with open(f'{self.agent_folder}/lvl{self.generator.id}.txt', 'w+') as fname:
            fname.write(str(self.generator))

    def mutate(self, mutationRate, minimal, r):
        """
        NOTE: r is an overloaded term depending on which generator we're using.
        If the evolutionary generator, r is the mutation radius. It didn't do anything at all, so we're going to
        overload it.

        If illuminating, r will represent the difficulty parameter.
        """
        if self.generatorType == "evolutionary":
            new_map, shp = self.generator.mutate(mutationRate=mutationRate,
                                                 minimal=minimal,
                                                 r=r)
            gen = EvolutionaryGenerator(game=self.game,
                                         args_file=self.generator.args_file,
                                         tile_world=None,
                                         shape=shp,
                                         path=self.generator.base_path,
                                         mechanics=self.generator.mechanics,
                                         generation=self.generator.generation + 1,
                                         locations=new_map)

            gen.to_file(gen.id, self.game)
            return gen

        else:
            level = self.generator.mutate(mutationRate=mutationRate,
                                          r=r,
                                          env_id=self.id,
                                          path=self.agent_folder)

            gen = IlluminatingGenerator(shape=self.generator.shape,
                                        args_file=self.generator.args_file,
                                        path=self.generator.base_path,
                                        generation=self.generator.generation + 1,
                                        lvl_path=level)

            return gen
