import numpy as np
from numba import jit

@jit(nopython=True)
def distanceFromPositions(positions, pos0):
    '''
    Calculate the distance between a number of positions and a single reference point.

    Args:
        positions: ndarray of shape (n_particles, dim)
            Positions for which to calculate the distance to pos0.
        pos0: ndarray of shape (dim,)
            Reference point to which the distances are calculated.

    Returns:
        distance: ndarray of shape (n_particles,)
            The distance to pos0 for each particle.
    '''

    dim = positions.shape[-1]

    if dim == 2:
        dist_squared = (positions[:,0] - pos0[0])**2 + (positions[:,1] - pos0[1])**2

    else:
        dist_squared = (positions[:,0] - pos0[0])**2 + (positions[:,1] - pos0[1])**2 + (positions[:,2] - pos0[2])**2

    distance = np.sqrt(dist_squared)

    return distance


@jit(nopython=True)
def distanceBetweenPositionSets(positions1, positions2):
    '''
    Calculate the distance between each particle specified by position1 and each particle specified by position2.
    Note that this function counts each pair twice. This is no problem if it is compared to other pair counts using the
    same procedure, where pairs are doubly counted.

    Args:
        positions1: ndarray of shape (n_particles1, dim)
            Positions of the first set of particles.
        positions2: ndarray of shape (n_particles2, dim)
            Positions of the second set of particles.

    Returns:
        distances: ndarray of shape (n_particles1, n_particles2)
    '''

    distances = np.zeros((len(positions1), len(positions2)))

    for i, pos1 in enumerate(positions1):
        distances[i] = distanceFromPositions(positions2, pos1)

    return distances


def randomPositions(n_particles, dim, box_size):
    '''
    Generate positions from a uniform random distribution such that each point in the grid is equally likely to be
    populated by a particle, i.e. to generate a set of completely uncorrelated particles inside the specified box.

    Args:
        n_particles: int
            Number of particles to generate.
        dim: int
            Number of dimensions. Must be 2 or 3 to be consistent with other simulation code.
        box_size: float
            Dimensionless linear size of the box.

    Returns:
        random_pos: ndarray of shape (n_particles, dim)
            Set of random positions.
    '''

    random_pos = np.random.uniform(0., box_size, size=(n_particles, dim))

    return random_pos


def countPairs(positions1, positions2, bin_edges):
    '''
    Count the number of pairs within specific bins of distance from each other. Uses numpy's histogram function.

    Args:
        positions1: ndarray of shape (n_particles1, dim)
            Positions of the first set of particles.
        positions2: ndarray of shape (n_particles2, dim)
            Positions of the second set of particles.
        bin_edges: ndarray of shape (n_bins + 1,)
            Edges of the distance bins to use.

    Returns:
        counts: ndarray of shape (n_bins,)
            The number of pairs counted in each bin.
        bin_mids: ndarray of shape (n_bins,)
            The midpoints of the distance bins used in the histogram.
    '''

    distances = distanceBetweenPositionSets(positions1, positions2)

    counts, bin_edges = np.histogram(distances, bins=bin_edges)
    widths = bin_edges[1:] - bin_edges[:-1]
    bin_mids = bin_edges[:-1] + 0.5 * widths

    return counts, bin_mids


def corrFuncLandySzalay(counts_data, counts_random, counts_data_random, floor_value=0.):
    '''
    Compute the Landy-Szalay estimator [Landy & Szalay (1993)] for the pair correlation function from a given set of
    data-data, random-random and data-random histograms.

    Args:
        counts_data: ndarray of shape (n_bins,)
            Number of data-data pairs found in each distance bin.
        counts_random: ndarray of shape (n_bins,)
            Number of random-random pairs found in each distance bin.
        counts_data_random: ndarray of shape (n_bins,)
            Number of data-random pairs found in each distance bin.
        floor_value: float
            (Small) term added to the random-random counts in the denominator such that the estimator doesn't fail if
            no random-random pairs have been found in a certain distance bin. To be used with much care!

    Returns:
        corr_func: ndarray of shape (n_bins,)
            Estimated correlation function evaluated for each distance bin.
    '''

    numerator = counts_data - 2 * counts_data_random + counts_random
    corr_func = numerator / (counts_random + floor_value)

    return corr_func