import numpy as np

class ParticleGrid:
    '''
    Particle Mesh grid containing the density of each grid cell.

    '''

    def __init__(self, size, dim=3):
        '''
        :param size: int
                Linear size of the box in terms of grid cells on each side
        '''

        if isinstance(size, int):
            self.size = size
        else:
            raise TypeError("size must be an integer.")

        if isinstance(dim ,int) and (dim == 2 or dim == 3):
            self.dim = dim
        else:
            raise ValueError("dim must be an integer with value 2 or 3.")

        mids = np.arange(0.5, size+0.5)

        if self.dim == 2:
            self.x_mids, self.y_mids = np.meshgrid(mids, mids)

        else:
            self.x_mids, self.y_mids, self.z_mids = np.meshgrid(mids, mids, mids)

        # initialise a universe with uniform density 1
        self._densities = np.ones_like(self.x_mids)
        self._overdensities = self._densities - 1

        # for now use nearest grid point density assignment
        # i.e. particles are point-like


    @property
    def densities(self):
        return self._densities

    @densities.setter
    def densities(self, dens):

        if isinstance(dens, np.ndarray) and dens.shape == self.x_mids.shape:
            self._densities = dens
            self._overdensities = dens - 1

        else:
            raise TypeError("Given density should contain the density evaluated at each grid midpoint.")

    @property
    def overdensities(self):
        return self._overdensities

    @overdensities.setter
    def overdensities(self, overdens):

        if isinstance(overdens, np.ndarray) and overdens.shape == self.x_mids.shape:
            self._overdensities = overdens
            self._densities = overdens + 1

        else:
            raise TypeError("Given overdensity should contain the density evaluated at each grid midpoint.")


    def randomDensities(self, std=0.001):

        rng = np.random.default_rng()
        self.densities = rng.normal(1, std, size=self._densities.shape)