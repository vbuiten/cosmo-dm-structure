import numpy as np

class Grid:
    '''
    Grid containing the density of each grid cell.

    Attributes:
        size: int
            Linear size of the box in terms of grid cells on each side.
        dim: int
            Dimensions of the box.
        densities: ndarray of shape (size, size, size) or (size, size), depending on the dimensions.
            Density field in the box.
        overdensities: ndarray of shape (size, size, size) or (size, size), depending on the dimensions.
            Overdensity field in the box.

    Methods:
        potential(Om0, scale_factor):
            Computes the potential at each point in the grid.
        randomDensities(std=0.001):
            Sets random densities following a Gaussian distribution of mean one and standard deviation std.

    '''

    def __init__(self, size, dim=3):
        '''
        :param size: int
                Linear size of the box in terms of grid cells on each side
        :param dim: int
                Dimensions of the box. Should be either 2 or 3.
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
        '''
        Getter for the densities property. Gives the density at each point in the grid.

        :return:
            self._densities: ndarray of shape (size, size, size) or (size, size)
                The density field.
        '''
        return self._densities

    @densities.setter
    def densities(self, dens):
        '''
        Setter for the densities property.

        :param dens: ndarray of shape (size, size, size) or (size, size) (depending on dimension)
                The density field.
        :return:
        '''

        if isinstance(dens, np.ndarray) and dens.shape == self.x_mids.shape:
            self._densities = dens
            self._overdensities = dens - 1

        else:
            raise TypeError("Given density should contain the density evaluated at each grid midpoint.")

    @property
    def overdensities(self):
        '''
        Getter for the overdensity at each point in the grid.

        :return:
            self._overdensities: ndarray of shape (size, size, size) or (size, size), depending on the dimension.
                The overdensity field.
        '''

        return self._overdensities

    @overdensities.setter
    def overdensities(self, overdens):
        '''
        Setter for the overdensities property.

        :param overdens: ndarray of shape (size, size, size) or (size, size) (depending on dimension)
                The overdensity field.
        :return:
        '''

        if isinstance(overdens, np.ndarray) and overdens.shape == self.x_mids.shape:
            self._overdensities = overdens
            self._densities = overdens + 1

        else:
            raise TypeError("Given overdensity should contain the density evaluated at each grid midpoint.")


    def potential(self, Om0, scale_factor):
        '''
        Compute the potential at each grid cell using FFT and IFFT.

        :param Om0: float
        :param scale_factor: float
        :return:
            potential_ifft: ndarray of shape (size, size, size) or (size, size), depending on the dimension.
                The computed potential field.
        '''

        overdens = np.fft.ifftshift(self.overdensities)
        overdens_fft = np.fft.fftn(overdens)
        overdens_fft = np.fft.fftshift(overdens_fft)

        print ("Shape of FFT: {}".format(overdens_fft.shape))
        prefactor = -3/2 * Om0 / scale_factor

        # compute the square of the k-vector of grid cell in k-space
        freqs = 2 * np.pi * np.fft.fftfreq(overdens_fft.shape[0], 1.)

        # inefficient code; see if we can improve it with numpy later
        if self.dim == 2:
            k_squared = np.zeros_like(overdens_fft.shape)
            for i in range(freqs):
                for j in range(freqs):
                    k_squared[i,j] = freqs[i]**2 + freqs[j]**2

        else:
            k_squared = np.zeros_like(overdens_fft.shape)
            for i in range(freqs):
                for j in range(freqs):
                    for l in range(freqs):
                        k_squared[i,j,l] = freqs[i]**2 + freqs[j]**2 + freqs[l]**2

        potential_fft = prefactor / k_squared * overdens_fft

        # manually set the potential (in k-space) to 0 in places where k^2 = 0
        potential_fft[k_squared == 0] = 0

        # compute the inverse FFT to get the potential in real space
        potential_fft = np.fft.ifftshift(potential_fft)
        potential_ifft = np.fft.ifftn(potential_fft)
        potential_ifft = np.fft.fftshift(potential_ifft)

        return potential_ifft


    def randomDensities(self, std=0.001):
        '''
        Randomly set initial densities in the grid. The density values are drawn from a Gaussian distribution with
        mean 1 and standard deviation std.

        :param std: float
                Standard deviation of the Gaussian. Default is 0.001.
        :return:
        '''

        rng = np.random.default_rng()
        self.densities = rng.normal(1, std, size=self._densities.shape)