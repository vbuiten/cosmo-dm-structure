import numpy as np
from numba import jit


#@jit(nopython=True)
def countParticlesInCell(positions, cellmids):

    dim = len(cellmids)

    if dim == 2:
        xmids, ymids = cellmids
    else:
        xmids, ymids, zmids = cellmids

    density = np.zeros((xmids.size, ymids.size))

    if dim == 2:

        for i, x in enumerate(xmids):
            for j, y in enumerate(ymids):

                in_cell = (positions[:,0] > (x - 0.5)) & (positions[:,0] < (x + 0.5)) \
                    & (positions[:,1] > (y - 0.5)) & (positions[:,1] < (y + 0.5))

                density[i,j] = np.sum(in_cell)

    return density


#@jit(nopython=True)
def nearestGridPointDensity(positions, meshgrid):

    '''
    if not isinstance(positions, np.ndarray):
        raise TypeError("'positions' must be an ndarray.")

    if not isinstance(meshgrid, list):
        raise TypeError("'meshgrid' must be a list of length 2 or 3.")

    else:
        dim = len(meshgrid)

        if not (dim == 2 or dim == 3):
            raise TypeError("'meshgrid' must be a list of length 2 or 3.")
    '''

    # this bit of code currently doesn't work with numba
    n_particles = len(positions)
    x1d = meshgrid[0][0,:]
    y1d = meshgrid[1][:,0]

    dim = len(meshgrid)

    if dim == 3:
        z1d = meshgrid[2].ravel()

    density = countParticlesInCell(positions, [x1d, y1d])

    # normalise the density so that the mean is 1
    normdensity = density / np.mean(density)

    return normdensity