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


@jit(nopython=False)
def cloudInCellDensity(positions, mids_tuple):

    dim = len(mids_tuple)

    if dim == 2:
        density = np.zeros((len(mids_tuple[0]), len(mids_tuple[1])))
    #density = np.zeros((len(mids_tuple[i]) for i in range(dim)))

    # for each particle we need to identify the (Cartesian!) index of its parent cell
    # and of all neighbouring parent cells it affects

    if dim == 2:
        for pos in positions:
            #parent_cell = (abs(pos[0] - mids_tuple[0]) < 0.5) & (abs(pos[1] - mids_tuple[1]) < 0.5)
            dist_x = pos[0] - mids_tuple[0]
            dist_y = pos[1] - mids_tuple[1]
            idx_x_parent = abs(dist_x) < 0.5
            idx_y_parent = abs(dist_y) < 0.5
            density[idx_y_parent, idx_x_parent] += (1 - dist_x[idx_x_parent]) * (1 - dist_y[idx_y_parent])


    else:
        for pos in positions:
            parent_cell = (abs(pos[0] - mids_tuple[0]) < 0.5) & (abs(pos[1] - mids_tuple[1]) < 0.5) \
                          & (abs(pos[2] - mids_tuple[2]) < 0.5)


    return density