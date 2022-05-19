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


#@jit(nopython=False)
def cloudInCellDensity(positions, mids_tuple, size):

    dim = len(mids_tuple)

    if dim == 2:

        X_mids, Y_mids = np.meshgrid(mids_tuple[0], mids_tuple[1])

        density = np.zeros((len(mids_tuple[0]), len(mids_tuple[1])))

    else:
        X_mids, Y_mids, Z_mids = np.meshgrid(mids_tuple[0], mids_tuple[1], mids_tuple[2])
        density = np.zeros((len(mids_tuple[0]), len(mids_tuple[1]), len(mids_tuple[2])))

    # find the 4 (2D) or 8 (3D) nearest gridpoints
    # calculate the distance to these gridpoints

    if dim == 2:
        for pos in positions:

            abs_dist_x = np.abs((pos[0] - X_mids)) % (size - 1)
            abs_dist_y = np.abs((pos[1] - Y_mids)) % (size - 1)

            nearest_cells = (abs_dist_x < 1.) & (abs_dist_y < 1.)
            density[nearest_cells] += ((abs_dist_x[nearest_cells] + 0.5) % 1) * ((abs_dist_y[nearest_cells] + 0.5) % 1)


    else:
        for pos in positions:

            abs_dist_x = np.abs((pos[0] - X_mids)) % (size - 1)
            abs_dist_y = np.abs((pos[1] - Y_mids)) % (size - 1)
            abs_dist_z = np.abs((pos[2] - Z_mids)) % (size - 1)

            nearest_cells = (abs_dist_x < 1.) & (abs_dist_y < 1.) & (abs_dist_z < 1.)
            density[nearest_cells] += ((abs_dist_x[nearest_cells] + 0.5) % 1) \
                                      * ((abs_dist_y[nearest_cells] + 0.5) % 1) \
                                      * ((abs_dist_z[nearest_cells] + 0.5) % 1)


    return density