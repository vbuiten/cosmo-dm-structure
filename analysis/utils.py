import numpy as np
from numba import jit

@jit(nopython=True)
def distanceFromPositions(positions, pos0):

    dim = positions.shape[-1]

    if dim == 2:
        dist_squared = (positions[:,0] - pos0[0])**2 + (positions[:,1] - pos0[1])**2

    else:
        dist_squared = (positions[:,0] - pos0[0])**2 + (positions[:,1] - pos0[1])**2 + (positions[:,2] - pos0[2])**2

    distance = np.sqrt(dist_squared)

    return distance


@jit(nopython=True)
def distanceBetweenPositionSets(positions1, positions2):

    distances = np.zeros((len(positions1), len(positions2)))

    for i, pos1 in enumerate(positions1):
        distances[i] = distanceFromPositions(positions2, pos1)

    return distances


def randomPositions(n_particles, dim, box_size):

    random_pos = np.random.uniform(0., box_size, size=(n_particles, dim))

    return random_pos


def countPairs():
    pass