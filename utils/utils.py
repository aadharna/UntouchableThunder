import numpy as np

def add_noise(grid):
    """add noise to the grid. Let that noise be flip the bit from 0 to 1 or vice-versa
    :param grid: np.array grid (in the case of zelda, shape=(13, 9, 13)
    :return: noisy_grid
    """
    def flip_bit(fiber, ind):
        fiber[ind] = (fiber[ind] + 1)%2

    noisy_grid = np.array(grid)

    for p1 in range(13):
        for p2 in range(9):
            # there are 13 channels in the fiber, so why not start with
            # that much noise.
            if np.random.rand() < (1/13):
                # slice all row, all column, fibers p1 then pick exactly depth p2
                # then pick which element of fiber p2 inside fiber_p1_p2
                fiber = noisy_grid[:, :, p1].T[p2]
                ind = np.random.choice(np.arange(len(fiber)))
                flip_bit(fiber, ind)

    return noisy_grid

