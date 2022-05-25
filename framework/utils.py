import numpy as np
from numba import jit


@jit(nopython=True)
def countParticlesInCell(positions, cellmids):
    '''
    Count the number of particles in each cell on the grid.

    :param positions: ndarray of shape (n_particles, dim)
        Positions of the particles (in Cartesian coordinates).
    :param cellmids: tuple of shape (size, size) or (size, size, size)
        Tuple of 1D arrays indicating the grid cell midpoints in all dimensions.

    :return:
        counts: ndarray of shape (size, size) or (size, size, size)
            Number of particles contained in each grid cell.
    '''

    dim = len(cellmids)

    if dim == 2:
        xmids, ymids = cellmids
    else:
        xmids, ymids, zmids = cellmids

    if dim == 2:

        counts = np.zeros((xmids.size, ymids.size))

        for j, x in enumerate(xmids):
            for i, y in enumerate(ymids):

                in_cell = (positions[:,0] > (x - 0.5)) & (positions[:,0] < (x + 0.5)) \
                    & (positions[:,1] > (y - 0.5)) & (positions[:,1] < (y + 0.5))

                counts[i,j] = np.sum(in_cell)

    else:
        counts = np.zeros((xmids.size, ymids.size, zmids.size))

        for k, x in enumerate(xmids):
            for j, y in enumerate(ymids):
                for i, z in enumerate(zmids):

                    in_cell = (positions[:,0] > (x - 0.5)) & (positions[:,0] < (x + 0.5)) \
                              & (positions[:,1] > (y - 0.5)) & (positions[:,1] < (y + 0.5)) \
                              & (positions[:,2] > (z - 0.5)) & (positions[:,2] < (z + 0.5))

                    counts[i,j,k] = np.sum(in_cell)

    return counts


@jit(nopython=True)
def nearestGridPointDensity(positions, mids_tuple):
    '''
    Calculate the discretised density field using the nearest grid point (NGP) method. This method amounts to counting
    the particles in each grid cell and normalising.

    :param positions: ndarray of shape (n_particles, dim)
        Positions of the particles (in Cartesian coordinates).
    :param mids_tuple: tuple of shape (size, size) or (size, size, size)
        Tuple of 1D arrays indicating the grid cell midpoints in all dimensions.

    :return:
        normdensity: ndarray of shape (size, size) or (size, size, size)
            Density in the field. The density is normalised such that mean is 1.
    '''

    density = countParticlesInCell(positions, mids_tuple)

    # normalise the density so that the mean is 1
    normdensity = density / np.mean(density)

    return normdensity


def cloudInCellDensity(positions, mids_tuple, size):
    '''
    Determine the discretised density field from the given positions using the cloud-in-cell (CIC) method. This method
    amounts to assuming the particles are squares/cubes of uniform density.

    :param positions: ndarray of shape (n_particles, dim)
        Positions of the particles (in Cartesian coordinates).
    :param mids_tuple: tuple of shape (size, size) or (size, size, size)
        Tuple of 1D arrays indicating the grid cell midpoints in all dimensions.
    :param size: int
        Linear size of the box in terms of the number of grid cells.

    :return:
        density: ndarray of shape (size, size) or (size, size, size)
            Discretised density field evaluated on the grid.
    '''

    dim = len(mids_tuple)

    if dim == 2:

        X_mids, Y_mids = np.meshgrid(mids_tuple[0], mids_tuple[1])

        density = np.zeros((len(mids_tuple[0]), len(mids_tuple[1])))

    else:
        X_mids, Y_mids, Z_mids = np.meshgrid(mids_tuple[0], mids_tuple[1], mids_tuple[2])
        density = np.zeros((len(mids_tuple[0]), len(mids_tuple[1]), len(mids_tuple[2])))

    # find the 4 (2D) or 8 (3D) nearest gridpoints
    # calculate the linear distance to these gridpoints

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