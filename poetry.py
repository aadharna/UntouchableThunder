import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("max_children", type=int, default=10)
parser.add_argument('max_generations', type=int, default=2)
args = parser.parse_args()


class POET:
    def __init__(self):
        pass

    def optimize(self):
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