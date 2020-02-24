import os
from copy import deepcopy
from agent.models import Net
from generator.levels.base import Generator

class MinimalPair():
    id = 0
    def __init__(self, unique_run_id,
                 game='dzelda',
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

        self.generator = generator

        self.id = MinimalPair.id
        MinimalPair.id += 1

        run_folder = f'{prefix}/results_{self.unique_run_id}/'

        if not os.path.exists(run_folder):
            os.mkdir(run_folder)
        agent_folder = os.path.join(run_folder, str(self.id))
        if not os.path.exists(agent_folder):
            os.mkdir(agent_folder)
        with open(f'{agent_folder}/lvl{self.generator.id}.txt', 'w+') as fname:
            fname.write(str(self.generator))

    def mutate(self, mutationRate, minimal, r):
        new_map, shp = self.generator.mutate(mutationRate=mutationRate,
                                             minimal=minimal,
                                             r=r)
        gen = Generator(tile_world=None,
                            shape=shp,
                            path=self.generator.base_path,
                            mechanics=self.generator.mechanics,
                            generation=self.generator.generation + 1,
                            locations=new_map)

        gen.to_file(gen.id, self.game)
        return gen

