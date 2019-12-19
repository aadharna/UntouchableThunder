import numpy as np
from gym.spaces import Box, Discrete

zelda_spaces = (Box(low=0, high=1, shape=(13, 9, 13), dtype=np.float32), Discrete(6))

def add_noise(grid):
    """add noise to the grid. Let that noise be flip the bit from 0 to 1 or vice-versa
    :param grid: np.array grid (in the case of zelda, shape=(13, 9, 13)
    :return: noisy_grid
    """
    def flip_bit(fiber, ind):
        # switch 1 to 0, or 0 to 1.
        new_fiber = np.zeros(len(fiber))
        new_fiber[ind] = 1
        return new_fiber

    noisy_grid = np.array(grid)

    for p1 in range(grid.shape[1]):
        for p2 in range(grid.shape[2]):
            # flip a bit in every other tensor fiber
            if np.random.rand() < 0.5:
                # slice all depths, at this row/col,
                # then pick which element of fiber p2 inside fiber_p1_p2
                fiber = noisy_grid[:, p1, p2]
                ind = np.random.choice(np.arange(len(fiber)))
                noisy_grid[:, p1, p2] = flip_bit(fiber, ind)

    return noisy_grid


def eval_cppn_genome(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = np.random.rand()