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
        mids1d: ndarray of shape (size,)
            Midpoints of the grid cells, given as a 1D array.
        mids_tuple: tuple of shape (dim,)
            Midpoints of the grid cells in all dimensions.
        x_mids: ndarray of shape (size, size) or (size, size, size)
            x-array for numpy meshgrid specifying the grid.
        y_mids: ndarray of shape (size, size) or (size, size, size)
            y-array for numpy meshgrid specifying the grid.
        z_mids: ndarray of shape (size, size, size)
            z-array for numpy meshgrid specifying the grid. Only exists if dim == 3.

    Methods:
        potential(Om0, scale_factor):
            Computes the potential at each point in the grid.
        randomDensities(std=0.001):
            Sets random densities following a Gaussian distribution of mean one and standard deviation std.
        exponentialDensityProfile(scale_length):
            Sets an exponential density profile with scale parameter scale_length.

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
        self.mids1d = mids

        if self.dim == 2:
            self.mids_tuple = (mids, mids)
            self.x_mids, self.y_mids = np.meshgrid(*self.mids_tuple)

        else:
            self.mids_tuple = (mids, mids, mids)
            self.x_mids, self.y_mids, self.z_mids = np.meshgrid(*self.mids_tuple)

        # initialise a universe with uniform density 1
        self._densities = np.ones_like(self.x_mids)


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
            self._overdensities = dens/np.mean(dens) - 1

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

        # compute the Green's function
        prefactor = -3/8 * Om0 / scale_factor

        # compute the square of the k-vector of grid cell in k-space
        freqs = 2 * np.pi * np.fft.fftfreq(overdens_fft.shape[0], 1.)

        # inefficient code; see if we can improve it with numpy later
        if self.dim == 2:
            k_squared = np.zeros_like(overdens_fft)
            sine_terms = np.zeros_like(overdens_fft)
            for i in range(freqs.size):
                for j in range(freqs.size):
                    k_squared[i,j] = freqs[i]**2 + freqs[j]**2
                    sine_terms[i,j] = np.sin(freqs[i] / self.size)**2 + np.sin(freqs[j] / self.size)**2

        else:
            k_squared = np.zeros_like(overdens_fft)
            sine_terms = np.zeros_like(overdens_fft)
            for i in range(freqs.size):
                for j in range(freqs.size):
                    for l in range(freqs.size):
                        k_squared[i,j,l] = freqs[i]**2 + freqs[j]**2 + freqs[l]**2
                        sine_terms[i,j,l] = np.sin(freqs[i] / self.size)**2 + np.sin(freqs[j] / self.size)**2 + np.sin(freqs[l] / self.size)**2

        potential_fft = prefactor / sine_terms * overdens_fft

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
        
        
    def exponentialDensityProfile(self, scale_length):
        '''
        Set an exponential density profile centred at x = y = z = size/2. The profile is normalised such that the mean
        density in the field is one.

        :param scale_length: float
                Scale length setting the "width" of the exponential profile.
        :return:
        '''
        
        mid = self.size / 2
        
        if self.dim == 2:
            distance_squared = (self.x_mids - mid)**2 + (self.y_mids - mid)**2
            
        else:
            distance_squared = (self.y_mids - mid)**2 + (self.y_mids - mid)**2 + (self.z_mids - mid)**2

        distance = np.sqrt(distance_squared)
        densities = np.exp(-distance / scale_length)

        # normalise so that the average is 1
        self.densities = densities / np.mean(densities)