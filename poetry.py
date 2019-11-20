import argparse

import numpy as np

from agent.NNagent import NNagent

from generator.env_gen_wrapper import GridGame

from optimization.Optimizer import PyTorchObjective



parser = argparse.ArgumentParser()
parser.add_argument("max_children", type=int, default=10)
parser.add_argument('max_generations', type=int, default=2)
parser.add_argument('game', type=str, default='zelda')
parser.add_argument('length', type=int, default=1000)
args = parser.parse_args()


class POET:
    def __init__(self, args):
        self.args = args
        self.population = []

        # initial population
        self.population.append(NNagent(GridGame(game=args.game,
                                                play_length=args.length,
                                                path='./levels',
                                                lvl_name='start.txt',
                                                mechanics=['1', '2', '3', '+', 'g', 'w'],
                                                # monsters, key, door, wall
                                                )
                                       )
                               )



    def optimize(self):
        """This function is responsible for 'main'."""
        pass

    def pass_mc(self, score):
        if 1 <= score <= 2:
            return True
        else:
            return False

    def create_children(self, parents, max_children=8):
        pass

    def evaluate_transer(self):
        pass

    def evaluate(self):
        pass

    def evaluate_and_pick_children(self, all_children):
        pass

    def compute_novelty_vs_archive(self, child):
        pass

    def remove_oldest(self):
        pass

    def transfer(self):
        pass





if __name__ == "__main__":
    # do something
    pass