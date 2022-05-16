import numpy as np
from numba import jit


@jit(nopython=True)
def countParticlesInCell(positions, cellmids):

    dim = len(cellmids)

    if dim == 2:
        xmids, ymids = cellmids
    else:
        xmids, ymids, zmids = cellmids

    if dim == 2:

        density = np.zeros((xmids.size, ymids.size))

        for j, x in enumerate(xmids):
            for i, y in enumerate(ymids):

                in_cell = (positions[:,0] > (x - 0.5)) & (positions[:,0] < (x + 0.5)) \
                    & (positions[:,1] > (y - 0.5)) & (positions[:,1] < (y + 0.5))

                density[i,j] = np.sum(in_cell)

    else:
        density = np.zeros((xmids.size, ymids.size, zmids.size))

        for k, x in enumerate(xmids):
            for j, y in enumerate(ymids):
                for i, z in enumerate(zmids):

                    in_cell = (positions[:,0] > (x - 0.5)) & (positions[:,0] < (x + 0.5)) \
                              & (positions[:,1] > (y - 0.5)) & (positions[:,1] < (y + 0.5)) \
                              & (positions[:,2] > (z - 0.5)) & (positions[:,2] < (z + 0.5))

                    density[i,j,k] = np.sum(in_cell)

    return density


@jit(nopython=True)
def nearestGridPointDensity(positions, mids_tuple):

    density = countParticlesInCell(positions, mids_tuple)

    # normalise the density so that the mean is 1
    #normdensity = density / np.mean(density)

    #return normdensity
    return density


def cloudInCellDensity(positions, mids_tuple):

    dim = len(mids_tuple)

    density = np.zeros([el.size for el in mids_tuple])

    if dim == 2:
        for pos in positions:
            parent_cell = (abs(pos[0] - mids_tuple[0]) < 0.5) & (abs(pos[1] - mids_tuple[1]) < 0.5)


    else:
        for pos in positions:
            parent_cell = (abs(pos[0] - mids_tuple[0]) < 0.5) & (abs(pos[1] - mids_tuple[1]) < 0.5) \
                          & (abs(pos[2] - mids_tuple[2]) < 0.5)